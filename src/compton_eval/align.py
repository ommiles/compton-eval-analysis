"""Judge alignment: does the judge agree with a human, and what is the true rate?

:mod:`compton_eval.reliability` answers whether a judge agrees with *itself*.
That is necessary and nowhere near sufficient. A judge can be perfectly
self-consistent and consistently wrong, and ICC will call it excellent. This
module answers the question that actually matters: how often does the judge
agree with a human, and what does the true failure rate look like once you
correct for the judge's errors?

The procedure follows Husain & Shankar, *Evals for AI Engineers*, ch. 5:

1. Split labeled traces into train / dev / test. Refine the judge prompt
   against dev only. Read test exactly once, after freezing the prompt.
2. On the frozen test set, measure the judge's true positive rate and true
   negative rate against human labels.
3. Run the judge over a larger unlabeled batch and observe its raw pass
   rate, ``p_obs``.
4. Correct ``p_obs`` for judge bias, then bootstrap the labeled test set to
   put an interval on the corrected estimate.

**Why TPR and TNR rather than precision and recall.** The goal is estimating
the true pass rate, and a judge can only get that wrong two ways: by missing
real passes, or by passing real fails. TPR and TNR name those two error modes
directly. Precision and recall are not wrong to compute, they just do not line
up with the quantity being estimated.

**Why the correction matters.** Counting the judge's "pass" labels over a
production batch gives a biased estimate, and the bias does not shrink as the
batch grows. More unlabeled data buys precision around the wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Below this, ``TPR + TNR - 1`` is too close to zero for the correction to
#: mean anything: the judge is near chance and the denominator explodes.
MIN_INFORMEDNESS = 0.05

#: The book's suggested stopping point for judge refinement.
GOOD_RATE = 0.90

#: Dev and test each want this many of *both* classes to pin the rates down.
MIN_PER_CLASS = 30


@dataclass(frozen=True)
class Alignment:
    """Judge accuracy against human labels on a frozen test set."""

    tpr: float
    tnr: float
    n_pass: int
    n_fail: int
    true_pass: int
    true_fail: int

    @property
    def informedness(self) -> float:
        """Youden's J: ``TPR + TNR - 1``. Zero means chance-level."""
        return self.tpr + self.tnr - 1.0

    @property
    def usable(self) -> bool:
        """Whether the correction can be applied at all."""
        return np.isfinite(self.informedness) and self.informedness > MIN_INFORMEDNESS

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.n_pass < MIN_PER_CLASS:
            out.append(
                f"only {self.n_pass} human-labeled Pass examples "
                f"(want >={MIN_PER_CLASS}); TPR is imprecise"
            )
        if self.n_fail < MIN_PER_CLASS:
            out.append(
                f"only {self.n_fail} human-labeled Fail examples "
                f"(want >={MIN_PER_CLASS}); TNR is imprecise"
            )
        if self.tpr < GOOD_RATE:
            out.append(f"TPR {self.tpr:.2f} below {GOOD_RATE:.0%} — keep refining the prompt")
        if self.tnr < GOOD_RATE:
            out.append(f"TNR {self.tnr:.2f} below {GOOD_RATE:.0%} — keep refining the prompt")
        if not self.usable:
            out.append(
                f"informedness {self.informedness:.3f} is at or near chance; "
                "bias correction is not valid — fix the judge, not the arithmetic"
            )
        return out

    def __str__(self) -> str:
        return (
            f"TPR={self.tpr:.3f} ({self.true_pass}/{self.n_pass})  "
            f"TNR={self.tnr:.3f} ({self.true_fail}/{self.n_fail})  "
            f"informedness={self.informedness:.3f}"
        )


@dataclass(frozen=True)
class CorrectedRate:
    """Bias-corrected true pass rate with a bootstrap interval."""

    observed: float
    corrected: float
    low: float
    high: float
    alignment: Alignment
    n_unlabeled: int
    confidence: float = 0.95
    note: str = ""

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def shift(self) -> float:
        """How far the correction moved the estimate."""
        return self.corrected - self.observed

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return (
            f"observed {self.observed:.3f} → corrected {self.corrected:.3f} "
            f"[{self.low:.3f}, {self.high:.3f}] ({pct}% CI, "
            f"m={self.n_unlabeled} unlabeled)"
        )


def _rates(human: np.ndarray, judge: np.ndarray) -> tuple[float, float, int, int, int, int]:
    """TPR, TNR and their supporting counts. Pass is the positive class."""
    n_pass = int(np.count_nonzero(human))
    n_fail = int(human.size - n_pass)
    true_pass = int(np.count_nonzero(human & judge))
    true_fail = int(np.count_nonzero(~human & ~judge))

    tpr = true_pass / n_pass if n_pass else float("nan")
    tnr = true_fail / n_fail if n_fail else float("nan")
    return tpr, tnr, n_pass, n_fail, true_pass, true_fail


def measure_alignment(
    human_labels: np.ndarray | list[bool | int],
    judge_labels: np.ndarray | list[bool | int],
) -> Alignment:
    """Compare a frozen judge against human labels on the test split.

    Both arguments are binary, where truthy means Pass. Binary is not a
    simplification for convenience — TPR and TNR are undefined on a Likert
    scale, so a 1-5 judge cannot be validated this way at all. That is the
    main practical argument for binary evaluators.
    """
    h = np.asarray(human_labels).astype(bool).ravel()
    j = np.asarray(judge_labels).astype(bool).ravel()
    if h.shape != j.shape:
        raise ValueError(f"label arrays must match: {h.shape} vs {j.shape}")
    if h.size == 0:
        raise ValueError("no labeled examples")

    tpr, tnr, n_pass, n_fail, true_pass, true_fail = _rates(h, j)
    if n_pass == 0 or n_fail == 0:
        raise ValueError(
            "test set has only one class; both Pass and Fail examples are "
            "needed to measure TPR and TNR"
        )
    return Alignment(tpr, tnr, n_pass, n_fail, true_pass, true_fail)


def correct_rate(p_obs: float, tpr: float, tnr: float) -> float:
    """Rogan-Gladen (1978) correction for an imperfect classifier.

        theta = (p_obs + TNR - 1) / (TPR + TNR - 1)

    Clipped to [0, 1]. Returns NaN when ``TPR + TNR - 1`` is not comfortably
    positive.

    The book's reference implementation rejects ``denom <= 0``. This uses a
    small positive floor instead, for two reasons. A judge with informedness
    of 0.01 passes a ``> 0`` test but divides by 0.01, so the "correction"
    amplifies noise by 100x and returns a confident-looking number built from
    nothing. And a judge *below* chance (negative informedness) must be
    rejected outright rather than sign-flipped into a plausible answer — the
    fix there is a better judge, not arithmetic.
    """
    informedness = tpr + tnr - 1.0
    if not np.isfinite(informedness) or informedness <= MIN_INFORMEDNESS:
        return float("nan")
    return float(np.clip((p_obs + tnr - 1.0) / informedness, 0.0, 1.0))


def corrected_success_rate(
    human_labels: np.ndarray | list[bool | int],
    judge_labels: np.ndarray | list[bool | int],
    *,
    n_unlabeled: int,
    n_judged_pass: int,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    include_batch_uncertainty: bool = False,
    seed: int = 0,
) -> CorrectedRate:
    """True pass rate over an unlabeled batch, corrected and bounded.

    Parameters
    ----------
    human_labels, judge_labels:
        The frozen test split. Binary, truthy = Pass.
    n_unlabeled:
        ``m``, the size of the unlabeled batch the judge ran over.
    n_judged_pass:
        ``k``, how many of those the judge called Pass.
    include_batch_uncertainty:
        The book's procedure holds ``p_obs`` fixed across bootstrap
        iterations and resamples only the labeled test set, so the interval
        reflects uncertainty in the judge's measured accuracy alone. That is
        the right call when ``m`` is large, since ``p_obs`` is then pinned
        down tightly. When ``m`` is small the binomial noise in ``p_obs``
        is not negligible, and setting this True resamples it too, widening
        the interval accordingly. Default False to match the book.
    """
    align = measure_alignment(human_labels, judge_labels)
    h = np.asarray(human_labels).astype(bool).ravel()
    j = np.asarray(judge_labels).astype(bool).ravel()

    if n_unlabeled <= 0:
        raise ValueError("n_unlabeled must be positive")
    if not 0 <= n_judged_pass <= n_unlabeled:
        raise ValueError(f"n_judged_pass ({n_judged_pass}) outside [0, {n_unlabeled}]")

    p_obs = n_judged_pass / n_unlabeled
    point = correct_rate(p_obs, align.tpr, align.tnr)

    if not align.usable or not np.isfinite(point):
        return CorrectedRate(
            observed=p_obs,
            corrected=float("nan"),
            low=float("nan"),
            high=float("nan"),
            alignment=align,
            n_unlabeled=n_unlabeled,
            confidence=confidence,
            note=(
                "judge is at or near chance (informedness "
                f"{align.informedness:.3f}); correction not applied"
            ),
        )

    rng = np.random.default_rng(seed)
    n = h.size
    idx = rng.integers(0, n, size=(n_resamples, n))

    thetas = np.empty(n_resamples, dtype=float)
    if include_batch_uncertainty:
        k_star = rng.binomial(n_unlabeled, p_obs, size=n_resamples)
        p_star = k_star / n_unlabeled
    else:
        p_star = np.full(n_resamples, p_obs)

    for b in range(n_resamples):
        hb, jb = h[idx[b]], j[idx[b]]
        tpr_b, tnr_b, np_b, nf_b, _, _ = _rates(hb, jb)
        # A resample can lose a class entirely; that draw carries no
        # information about the missing rate, so it is dropped rather than
        # silently treated as 0 or 1.
        thetas[b] = (
            correct_rate(p_star[b], tpr_b, tnr_b)
            if np_b and nf_b
            else np.nan
        )

    good = thetas[np.isfinite(thetas)]
    dropped = n_resamples - good.size
    if good.size < n_resamples // 10:
        return CorrectedRate(
            p_obs, point, float("nan"), float("nan"), align, n_unlabeled, confidence,
            note=(
                f"{dropped}/{n_resamples} bootstrap draws were unusable — the "
                "test set is too small or too imbalanced to bound this estimate"
            ),
        )

    alpha = (1 - confidence) / 2
    low, high = np.quantile(good, [alpha, 1 - alpha])

    note = ""
    if dropped:
        note = f"{dropped}/{n_resamples} bootstrap draws dropped (a class went missing)"

    return CorrectedRate(
        observed=p_obs,
        corrected=point,
        low=float(low),
        high=float(high),
        alignment=align,
        n_unlabeled=n_unlabeled,
        confidence=confidence,
        note=note,
    )


def split_labeled(
    n: int,
    *,
    labels: np.ndarray | list[bool | int] | None = None,
    train: float = 0.15,
    dev: float = 0.425,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / dev / test indices, stratified by label when one is given.

    Defaults follow the book: roughly 10-20% train (few-shot candidates only),
    with the rest split evenly between dev and test. Unlike ordinary supervised
    learning the training pool is the *smallest* slice, because in-context
    learning saturates after a handful of well-chosen examples and the data is
    better spent on measurement.

    Returns index arrays. Test is meant to be read once, after the prompt is
    frozen; nothing here enforces that, but reusing it inflates TPR and TNR.
    """
    if not 0 < train < 1 or not 0 < dev < 1 or train + dev >= 1:
        raise ValueError(f"invalid split: train={train}, dev={dev}")

    rng = np.random.default_rng(seed)

    def cut(pool: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pool = rng.permutation(pool)
        a = int(round(train * pool.size))
        b = a + int(round(dev * pool.size))
        return pool[:a], pool[a:b], pool[b:]

    if labels is None:
        return cut(np.arange(n))

    y = np.asarray(labels).astype(bool).ravel()
    if y.size != n:
        raise ValueError(f"labels length {y.size} != n {n}")

    tr_p, dv_p, te_p = cut(np.flatnonzero(y))
    tr_f, dv_f, te_f = cut(np.flatnonzero(~y))
    return (
        np.sort(np.concatenate([tr_p, tr_f])),
        np.sort(np.concatenate([dv_p, dv_f])),
        np.sort(np.concatenate([te_p, te_f])),
    )
