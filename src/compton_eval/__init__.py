"""Statistical analysis for LLM summarization eval runs.

Turns a variant-comparison table of raw means into a defensible claim about
which differences are real.
"""

from .bootstrap import Interval, bootstrap_mean, bootstrap_paired_delta
from .compare import cliffs_delta, compare_run, holm_bonferroni
from .load import DIMENSIONS, EvalRun, load_run, load_runs
from .power import PowerCurve, power_for_run, power_table, simulate_power
from .reliability import Reliability, check_repeatability, icc_2_1, krippendorff_alpha

__version__ = "0.1.0"

__all__ = [
    "DIMENSIONS", "EvalRun", "Interval", "PowerCurve", "Reliability",
    "bootstrap_mean", "bootstrap_paired_delta", "check_repeatability",
    "cliffs_delta", "compare_run", "holm_bonferroni", "icc_2_1",
    "krippendorff_alpha", "load_run", "load_runs", "power_for_run",
    "power_table", "simulate_power",
]
