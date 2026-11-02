"""Exact small-sample statistics for decoy ensembles, with the standard library only.

The ensemble steps compare rates (a register formed, a backbone closed) between two sets of decoys and
scores between groups.  Their scripts run where MDATK may come without SciPy, so everything here uses
``math`` and ``statistics`` only; the tests check each function against SciPy when it is installed.

* :func:`fisher_exact` - two-sided Fisher exact test of a 2 x 2 table (the hypergeometric tables no
  more likely than the observed one, with a relative tolerance of 1e-7 for ties, as R and SciPy do).
* :func:`clopper_pearson` - the exact binomial interval.  The binomial tails are summed in log space
  (``lgamma``), outward from the mode until the terms vanish, so large ensembles neither overflow
  (``math.comb(n, k) * p**k`` does past a few thousand decoys) nor cost O(n) per evaluation.
* :func:`wilson` - the Wilson score interval.
* :func:`fisher_power` / :func:`detectable_rates` - the power of the two-sided Fisher test for two true
  rates, exact (enumerating the outcomes margin by margin) up to :data:`EXACT_POWER_LIMIT` decoys in
  both arms, otherwise from the normal approximation of the unpooled two-proportion test; and the rates
  of the second arm that the test would detect with a given power.  Both arms' sampling enters, so the
  uncertainty of the baseline rate is not ignored.
* :func:`mann_whitney` - the Mann-Whitney U test, exact (no ties, a small sample) or by the normal
  approximation with tie and continuity corrections; the same choices as ``scipy.stats.mannwhitneyu``.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

FISHER_TIE = 1e-7  # relative tolerance: tables this close to the observed probability count as ties
LOG_TAIL = 46.0  # binomial terms below exp(-46) (1e-20) of the largest are left out of tail sums
EXACT_POWER_LIMIT = 4000  # n1 + n2 up to which fisher_power enumerates the outcomes
_NORMAL = NormalDist()


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _check_counts(*values: int) -> None:
    for v in values:
        if int(v) != v or v < 0:
            raise ValueError(f"counts must be whole numbers >= 0, got {v!r}")


# --- Fisher exact test ----------------------------------------------------------------------------------

def _hypergeom_logpmf(x: int, n: int, r1: int, c1: int) -> float:
    """log P(a = x) for a 2 x 2 table with first-row total r1, first-column total c1 and n in all."""
    return _log_comb(r1, x) + _log_comb(n - r1, c1 - x) - _log_comb(n, c1)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value of the table [[a, b], [c, d]]."""
    _check_counts(a, b, c, d)
    n = a + b + c + d
    r1, c1 = a + b, a + c
    if n == 0 or r1 in (0, n) or c1 in (0, n):
        return 1.0
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    observed = _hypergeom_logpmf(a, n, r1, c1)
    limit = observed + math.log1p(FISHER_TIE)
    terms = [_hypergeom_logpmf(x, n, r1, c1) for x in range(lo, hi + 1)]
    kept = [t for t in terms if t <= limit]
    return min(1.0, math.fsum(math.exp(t) for t in kept))


# --- binomial tails and intervals -----------------------------------------------------------------------

def binom_logpmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return _log_comb(n, k) + k * math.log(p) + (n - k) * math.log1p(-p)


def _tail_sum(start: int, stop: int, step: int, n: int, p: float) -> float:
    """Sum of the binomial pmf from ``start`` towards ``stop`` (inclusive) in steps of +-1, assuming the
    terms decrease in that direction (``start`` on the mode's side); stops once they vanish."""
    first = binom_logpmf(start, n, p)
    if first == -math.inf:
        return 0.0
    out = []
    k = start
    while (k <= stop) if step > 0 else (k >= stop):
        t = binom_logpmf(k, n, p)
        if t < first - LOG_TAIL:
            break
        out.append(t - first)
        k += step
    return math.exp(first) * math.fsum(math.exp(t) for t in out)


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Bin(n, p), summed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    mode = int(math.floor((n + 1) * p))
    if k < mode:  # terms rise towards k: sum downwards from k
        return min(1.0, _tail_sum(k, 0, -1, n, p))
    return max(0.0, 1.0 - _tail_sum(k + 1, n, 1, n, p))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Bin(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    mode = int(math.floor((n + 1) * p))
    if k > mode:  # terms fall from k upwards
        return min(1.0, _tail_sum(k, n, 1, n, p))
    return max(0.0, 1.0 - _tail_sum(k - 1, 0, -1, n, p))


def _bisect(f, lo: float, hi: float, iterations: int = 200) -> float:
    """Root of an increasing function f on [lo, hi]."""
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) 1 - alpha interval of a binomial proportion k / n, as fractions."""
    _check_counts(k, n)
    if k > n:
        raise ValueError(f"k = {k} is larger than n = {n}")
    if n == 0:
        return 0.0, 1.0
    half = alpha / 2.0
    # lower: P(X >= k | p) = alpha/2 (increasing in p); upper: P(X <= k | p) = alpha/2 (decreasing)
    low = 0.0 if k == 0 else _bisect(lambda p: binom_sf(k, n, p) - half, 0.0, 1.0)
    high = 1.0 if k == n else _bisect(lambda p: half - binom_cdf(k, n, p), 0.0, 1.0)
    return low, high


def wilson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score 1 - alpha interval of k / n, as fractions."""
    _check_counts(k, n)
    if n == 0:
        return 0.0, 1.0
    z = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    ph = k / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1.0 - ph) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


# --- power and detectable differences -------------------------------------------------------------------

def _binom_pmfs(n: int, p: float) -> List[float]:
    return [math.exp(binom_logpmf(k, n, p)) for k in range(n + 1)]


def _support(pmf: Sequence[float], eps: float = 1e-14) -> Tuple[int, int]:
    keep = [k for k, v in enumerate(pmf) if v > eps]
    return (keep[0], keep[-1]) if keep else (0, len(pmf) - 1)


def _hypergeom_terms(n: int, r1: int, c1: int) -> Dict[int, float]:
    """{x: log P(a = x)} over the part of the support within exp(-LOG_TAIL) of the mode (the rest cannot
    change a p-value compared with a level such as 0.05)."""
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    mode = min(hi, max(lo, int(math.floor((c1 + 1) * (r1 + 1) / (n + 2)))))
    top = _hypergeom_logpmf(mode, n, r1, c1)
    out = {mode: top}
    for step in (-1, 1):
        x = mode + step
        while lo <= x <= hi:
            t = _hypergeom_logpmf(x, n, r1, c1)
            if t < top - LOG_TAIL:
                break
            out[x] = t
            x += step
    return out


_REJECTION: Dict[Tuple[int, int, float, int], Tuple[int, int]] = {}


def _rejection(n1: int, n2: int, alpha: float, m: int) -> Tuple[int, int]:
    """(L, U): with the column total m, the two-sided Fisher test at level alpha rejects for x <= L or
    x >= U (x = successes of arm 1).  The hypergeometric pmf is unimodal and a table's p-value only falls
    as its probability falls, so the rejection region is the two tails; it does not depend on the true
    rates, so it is computed once per (n1, n2, alpha, m)."""
    key = (n1, n2, alpha, m)
    hit = _REJECTION.get(key)
    if hit is not None:
        return hit
    n = n1 + n2
    logs = _hypergeom_terms(n, n1, m)  # beyond this band every table is rejected
    ordered = sorted(logs.values())
    cum, total = [], 0.0
    for t in ordered:
        total += math.exp(t)
        cum.append(total)

    def rejected(x: int) -> bool:
        limit = logs[x] + math.log1p(FISHER_TIE)
        k_lo, k_hi = 0, len(ordered)  # the p-value: every kept table at most as likely
        while k_lo < k_hi:
            mid = (k_lo + k_hi) // 2
            if ordered[mid] <= limit:
                k_lo = mid + 1
            else:
                k_hi = mid
        return (cum[k_lo - 1] if k_lo else 0.0) <= alpha * (1 + 1e-12)

    xs = sorted(logs)
    mode = max(logs, key=logs.get)
    low = xs[0] - 1  # every x below the band
    for x in range(mode, xs[0] - 1, -1):
        if rejected(x):
            low = x
            break
    high = xs[-1] + 1
    for x in range(mode, xs[-1] + 1):
        if rejected(x):
            high = x
            break
    if len(_REJECTION) > 200000:
        _REJECTION.clear()
    _REJECTION[key] = (low, high)
    return low, high


def fisher_power_exact(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05) -> float:
    """Exact power of the two-sided Fisher test at level ``alpha`` when arm 1 (n1 decoys) has the true rate
    p1 and arm 2 (n2) the rate p2: the probability of the outcomes (x1, x2) whose p-value is <= alpha."""
    b1, b2 = _binom_pmfs(n1, p1), _binom_pmfs(n2, p2)
    lo1, hi1 = _support(b1)
    lo2, hi2 = _support(b2)
    power = []
    for m in range(lo1 + lo2, hi1 + hi2 + 1):  # m = x1 + x2, the column total
        low, high = _rejection(n1, n2, alpha, m)
        for x in range(max(lo1, m - hi2), min(hi1, m - lo2) + 1):
            if x <= low or x >= high:
                power.append(b1[x] * b2[m - x])
    return min(1.0, math.fsum(power))


def power_normal(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05) -> float:
    """Power of the two-sided unpooled two-proportion z test (the large-sample approximation)."""
    if n1 <= 0 or n2 <= 0:
        return 0.0
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return 1.0 if p1 != p2 else 0.0
    z = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    shift = abs(p1 - p2) / se
    return _NORMAL.cdf(shift - z) + _NORMAL.cdf(-shift - z)


def fisher_power(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05) -> Tuple[float, str]:
    """(power, method): exact for n1 + n2 <= EXACT_POWER_LIMIT, the normal approximation above."""
    if n1 <= 0 or n2 <= 0:
        return 0.0, "none"
    if n1 + n2 <= EXACT_POWER_LIMIT:
        return fisher_power_exact(p1, n1, p2, n2, alpha), "exact"
    return power_normal(p1, n1, p2, n2, alpha), "normal"


def detectable_rates(p0: float, n0: int, n1: int, alpha: float = 0.05, power: float = 0.8
                     ) -> Dict[str, object]:
    """Rates of arm 1 (n1 decoys) that a two-sided Fisher test against arm 0 (rate p0, n0 decoys) detects
    with probability ``power``: {'lower': rate or None, 'upper': rate or None, 'method': ...}.  A side is
    None when no rate on it reaches the power (e.g. below a baseline of 0, or with too few decoys)."""
    out: Dict[str, object] = {"lower": None, "upper": None, "method": "none", "power": power,
                              "alpha": alpha}
    if n0 <= 0 or n1 <= 0:
        return out
    method = fisher_power(p0, n0, p0, n1, alpha)[1]

    def pw(p):
        return fisher_power(p0, n0, p, n1, alpha)[0]

    if p0 < 1.0 and pw(1.0) >= power:
        out["upper"] = _bisect(lambda p: pw(p) - power, p0, 1.0, iterations=30)
    if p0 > 0.0 and pw(0.0) >= power:
        out["lower"] = _bisect(lambda p: power - pw(p), 0.0, p0, iterations=30)
    out["method"] = method
    return out


# --- Mann-Whitney U ------------------------------------------------------------------------------------

def _ranks(values: Sequence[float]) -> Tuple[List[float], List[int]]:
    """Average ranks (1-based) and the sizes of the groups of ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    ties = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def _u_counts(n1: int, n2: int) -> List[int]:
    """Number of arrangements giving U = 0 .. n1*n2 (no ties): the coefficients of the Gaussian binomial
    prod_{i=1..n1} (1 - q^(n2+i)) / (1 - q^i).  Terms above degree n1*n2 are dropped on the way; neither
    the multiplication nor the division (a running sum) reads a higher coefficient than it writes."""
    deg = n1 * n2
    poly = [1] + [0] * deg
    for i in range(1, n1 + 1):
        s = n2 + i
        for k in range(deg, s - 1, -1):  # times (1 - q^s)
            poly[k] -= poly[k - s]
        for k in range(i, deg + 1):  # divided by (1 - q^i)
            poly[k] += poly[k - i]
    return poly


def mann_whitney(x: Sequence[float], y: Sequence[float], method: str = "auto") -> Tuple[float, float, str]:
    """Two-sided Mann-Whitney U test: (U of ``x``, p-value, method used).

    ``method``: 'exact' (no ties: the exact null distribution), 'asymptotic' (normal approximation with
    the tie correction and a continuity correction of 0.5) or 'auto' (exact when one sample has at most
    8 values and there are no ties, as SciPy chooses)."""
    n1, n2 = len(x), len(y)
    if not n1 or not n2:
        raise ValueError("both samples need at least one value")
    ranks, ties = _ranks(list(x) + list(y))
    u1 = math.fsum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    if method == "auto":
        method = "asymptotic" if (n1 > 8 and n2 > 8) or ties else "exact"
    if method == "exact":
        if ties:
            raise ValueError("the exact Mann-Whitney test needs samples without ties")
        counts = _u_counts(n1, n2)
        total = math.fsum(counts)
        u = int(round(max(u1, u2)))
        p = 2.0 * math.fsum(counts[u:]) / total
        return u1, min(1.0, p), "exact"
    if method != "asymptotic":
        raise ValueError(f"unknown method {method!r}")
    n = n1 + n2
    mu = n1 * n2 / 2.0
    tie_term = sum(t ** 3 - t for t in ties) / (n * (n - 1)) if n > 1 else 0.0
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term))
    if sigma == 0:
        return u1, 1.0, "asymptotic"
    z = (max(u1, u2) - mu - 0.5) / sigma
    return u1, min(1.0, 2.0 * (1.0 - _NORMAL.cdf(z))), "asymptotic"


# --- small helpers for reports --------------------------------------------------------------------------

def median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(v for v in values if v == v)  # NaN dropped
    if not vals:
        return None
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else 0.5 * (vals[m - 1] + vals[m])


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile (NumPy's default); NaN dropped; None for no values."""
    vals = sorted(v for v in values if v == v)
    if not vals:
        return None
    pos = (len(vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
