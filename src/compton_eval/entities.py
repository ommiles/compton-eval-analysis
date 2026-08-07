"""Deterministic proper-name checking against an entity vocabulary.

Round-1 error analysis found `proper_name_misspelled` in 50% of summaries
(18 of 36), evenly spread across prompt variants. It is an ASR-rooted
failure that no prompt change touches and that four scored dimensions
never saw. It is also the one failure mode that needs no LLM judge: the
correct spellings live in the database.

Why this checks a vocabulary and not sibling outputs. All three variants of
one meeting agreed on "Norseman" for the firm Nossaman. The garble entered
upstream in transcription and every prompt faithfully reproduced it, so
cross-variant comparison sees consensus and reports nothing. Only an
external source of truth catches that class.

Two detection tiers, kept separate because their error profiles differ:

**Known garbles** (`confidence="high"`). Exact lookup against a curated
garble→correct map. No false positives by construction: something is only
here because a human recorded that it is wrong.

**Near misses** (`confidence="review"`). A capitalized token close to a
canonical name but not equal to it. This finds garbles nobody has recorded
yet, which is how the map grows, but it guesses and must be reviewed.
Guarded hard: a token that exactly matches any known-good entity is never
flagged, because "Sanders Roberts" the law firm must not be reported as a
misspelling of Treasurer "Sanders".
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

#: Shipped export of officials, aliases, asr_corrections, staff and vendors.
DEFAULT_VOCAB = Path(__file__).parent / "data" / "compton_entities.json"

#: Minimum token length before near-miss matching is attempted. Short tokens
#: are edit-distance neighbours of far too much.
MIN_FUZZY_LEN = 5

#: Maximum edit distance for a near-miss. One is a transposition or a single
#: substitution; two starts producing unrelated pairs at these lengths.
MAX_FUZZY_DISTANCE = 2

#: Tokens that look like names but are structural English or civic boilerplate.
_STOPWORDS = frozenset("""
    the and for with from that this these those council member members city
    compton meeting agenda item items approved approval contract contracts
    consent department director general fund funds public comment committee
    commission board resolution resolutions ordinance amendment services
    service january february march april may june july august september
    october november december monday tuesday wednesday thursday friday
    street streets avenue boulevard road project projects program programs
    million thousand budget report reports vote votes motion second
    mayor attorney manager clerk treasurer controller chief captain
    inc llc lp llp corporation company incorporated
    penal code section article phase fiscal annual settlement lawsuit
    litigation grant grants funding measure ordinance article
""".split())

#: Titles that mark the next capitalized token as a person's name. A
#: distance-2 near-miss is only trusted inside this context.
_PERSON_TITLES = frozenset("""
    councilmember councilman councilwoman mayor attorney chief captain
    director commissioner treasurer clerk controller manager mr mrs ms dr
""".split())


def _norm(s: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().casefold()


def _distance(a: str, b: str, cap: int = MAX_FUZZY_DISTANCE) -> int:
    """Levenshtein distance, abandoning early once it exceeds ``cap``."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class NameFinding:
    """One suspected proper-name error."""

    found: str
    suggested: str
    category: str
    confidence: str
    evidence: str
    count: int = 1

    def __str__(self) -> str:
        n = f" x{self.count}" if self.count > 1 else ""
        return f"{self.found} -> {self.suggested}{n}  [{self.confidence}: {self.evidence}]"


@dataclass
class EntityVocabulary:
    """Known-correct names plus recorded garbles."""

    #: normalized garble -> (correct form, category, evidence)
    garbles: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    #: normalized correct token/phrase -> canonical display form
    canonical: dict[str, str] = field(default_factory=dict)
    #: normalized surname -> canonical full name, for person near-misses
    surnames: dict[str, str] = field(default_factory=dict)
    #: every normalized string that is legitimately some entity's name
    known_good: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_VOCAB) -> EntityVocabulary:
        raw = json.loads(Path(path).read_text())
        v = cls()

        for c in raw.get("corrections") or []:
            g, corrected = c["garbled"], c["corrected"]
            if _norm(g) == _norm(corrected):
                continue  # self-mapping placeholder row, carries no correction
            v.garbles[_norm(g)] = (corrected, c.get("category", "other"),
                                   "asr_corrections")
            v.known_good.add(_norm(corrected))
            for tok in _norm(corrected).split():
                if len(tok) >= MIN_FUZZY_LEN and tok not in _STOPWORDS:
                    v.known_good.add(tok)

        for o in raw.get("officials") or []:
            full = o["full_name"]
            v.canonical[_norm(full)] = full
            v.known_good.add(_norm(full))
            parts = [p for p in _norm(full).split()
                     if len(p) >= MIN_FUZZY_LEN and p not in _STOPWORDS]
            if parts:
                v.surnames.setdefault(parts[-1], full)
            for p in parts:
                v.known_good.add(p)
            # Aliases are treated as known-good vocabulary only, never as
            # garbles. The column mixes legitimate short forms ("Mayor",
            # "Madam City Clerk", "Councilmember Darden") with recorded
            # misspellings ("Tamara Binns") and marks neither. An earlier
            # version guessed by testing whether the alias shared the
            # canonical surname; that classified the bare title "Mayor" as a
            # misspelling of "Emma Sharif" and fired on four summaries whose
            # only crime was the word mayor. asr_corrections is the one place
            # a human asserted "this spelling is wrong", so it is the only
            # garble source. Genuine alias-column garbles are already
            # duplicated there.
            for alias in o.get("aliases") or []:
                na = _norm(alias)
                if not na:
                    continue
                v.known_good.add(na)
                for t in na.split():
                    if len(t) >= MIN_FUZZY_LEN and t not in _STOPWORDS:
                        v.known_good.add(t)

        for s in raw.get("staff") or []:
            full = s.get("full_name")
            if not full:
                continue
            v.canonical[_norm(full)] = full
            v.known_good.add(_norm(full))
            for p in _norm(full).split():
                if len(p) >= MIN_FUZZY_LEN and p not in _STOPWORDS:
                    v.known_good.add(p)

        for vendor in raw.get("vendors") or []:
            if not vendor:
                continue
            nv = _norm(vendor)
            v.canonical.setdefault(nv, vendor)
            v.known_good.add(nv)
            for tok in nv.split():
                if len(tok) >= MIN_FUZZY_LEN and tok not in _STOPWORDS:
                    v.known_good.add(tok)

        return v

    def add_garble(self, garbled: str, corrected: str, *, category: str = "person",
                   evidence: str = "manual") -> None:
        self.garbles[_norm(garbled)] = (corrected, category, evidence)
        self.known_good.add(_norm(corrected))

    @property
    def size(self) -> dict[str, int]:
        return {"garbles": len(self.garbles), "canonical": len(self.canonical),
                "surnames": len(self.surnames), "known_good": len(self.known_good)}


@lru_cache(maxsize=4)
def load_vocabulary(path: str | Path = DEFAULT_VOCAB) -> EntityVocabulary:
    return EntityVocabulary.load(path)


_CAP_TOKEN = re.compile(r"\b[A-Z][a-zA-Z'\-]{2,}\b")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def check_names(
    text: str,
    vocab: EntityVocabulary | None = None,
    *,
    include_near_misses: bool = True,
) -> list[NameFinding]:
    """Find suspected proper-name errors in one summary.

    Returns high-confidence known garbles first, then near misses.
    """
    vocab = vocab or load_vocabulary()
    if not text:
        return []

    plain = _MD_LINK.sub(r"\1", text)
    normalized = _norm(plain)
    findings: dict[str, NameFinding] = {}

    # Tier 1: recorded garbles, longest phrase first so a multi-word entry
    # wins over any single-word entry nested inside it.
    for g in sorted(vocab.garbles, key=len, reverse=True):
        n = len(re.findall(rf"(?<!\w){re.escape(g)}(?!\w)", normalized))
        if not n:
            continue
        corrected, category, evidence = vocab.garbles[g]
        findings[g] = NameFinding(g, corrected, category, "high", evidence, n)

    if not include_near_misses:
        return list(findings.values())

    # Tier 2: capitalized tokens close to a canonical surname.
    seen: set[str] = set()
    for match in _CAP_TOKEN.finditer(plain):
        tok = match.group(0)
        nt = _norm(tok)
        if (nt in seen or len(nt) < MIN_FUZZY_LEN or nt in _STOPWORDS
                or nt in vocab.known_good or nt in findings):
            continue
        seen.add(nt)
        # Skip tokens already covered by a Tier 1 phrase hit.
        if any(nt in g.split() for g in findings):
            continue

        # A token that is one word of a known multi-word entity present in
        # the text is not a misspelling. "Roberts" alone sits distance 1 from
        # "Robert Ahn", but in "Sanders Roberts" it is half a law firm.
        start, end = match.span()
        before = plain[:start].split()
        left = _norm(before[-1]) if before else ""
        after = plain[end:].split()
        right = _norm(after[0]) if after else ""
        if (left and f"{left} {nt}" in vocab.known_good) or (
            right and f"{nt} {right}" in vocab.known_good
        ):
            continue

        best, best_d, best_surname = None, MAX_FUZZY_DISTANCE + 1, ""
        for surname, full in vocab.surnames.items():
            d = _distance(nt, surname)
            if d < best_d:
                best, best_d, best_surname = full, d, surname
        # Distance 2 is only trusted after a title. Without that context it
        # pairs Penal/Bernal and Bredney/Bradley.
        limit = MAX_FUZZY_DISTANCE if left in _PERSON_TITLES else 1
        if best and 0 < best_d <= limit:
            findings[nt] = NameFinding(
                tok, best, "person", "review",
                f"near-miss on '{best_surname}' (distance {best_d})",
                len(re.findall(rf"(?<!\w){re.escape(nt)}(?!\w)", normalized)),
            )

    order = {"high": 0, "review": 1}
    return sorted(findings.values(), key=lambda f: (order[f.confidence], -f.count))


def score_trace(text: str, vocab: EntityVocabulary | None = None,
                *, high_confidence_only: bool = True) -> tuple[int, list[NameFinding]]:
    """Binary evaluator: 1 if the summary misspells a proper name, else 0.

    Defaults to high-confidence findings only. Near misses are for growing
    the vocabulary, not for gating an automated metric.
    """
    findings = check_names(text, vocab)
    if high_confidence_only:
        findings = [f for f in findings if f.confidence == "high"]
    return (1 if findings else 0), findings
