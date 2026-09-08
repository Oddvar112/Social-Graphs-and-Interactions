"""
Heavy-tail fitting the Clauset-Shalizi-Newman way, for integer degrees.

Three steps, as the course Goodies describe them:
  1. estimate the exponent by maximum likelihood, not by a line through a log-log plot;
  2. choose k_min by the Kolmogorov-Smirnov distance between fitted and empirical CCDF;
  3. test the fit with a semi-parametric bootstrap (p small = reject the power law),
     and compare against alternatives (log-normal, exponential) with a Vuong
     likelihood-ratio test on the same tail, because "plausible" is not "best".

Everything here is discrete: degrees are integers, and the discrete power law
P(k) = k^-alpha / zeta(alpha, k_min) uses the Hurwitz zeta from scipy. No
dependency on the powerlaw package, so the numbers are reproducible with the
same four libraries as week 1 plus scipy.

Only numpy and scipy. Imported by analyse_week2.py; run directly for a self-test.
"""

import math

import numpy as np
from scipy import optimize, special, stats

KMAX = 200_000          # truncation for numerical normalisation of the alternatives


# ---------------------------------------------------------------- power law
def _pl_loglik(alpha, tail, kmin):
    return -alpha * np.log(tail).sum() - len(tail) * np.log(special.zeta(alpha, kmin))


def fit_powerlaw(tail, kmin):
    """MLE of alpha for the discrete power law on data >= kmin."""
    tail = np.asarray(tail, dtype=float)
    res = optimize.minimize_scalar(lambda a: -_pl_loglik(a, tail, kmin), bounds=(1.01, 8.0), method="bounded")
    alpha = float(res.x)
    # Asymptotic standard error of the discrete MLE (Clauset et al. eq. 3.6 analogue).
    se = (alpha - 1) / math.sqrt(len(tail)) if alpha > 1 else float("nan")
    return alpha, se, -float(res.fun)


def pl_cdf(alpha, kmin, ks):
    """P(K <= k) for the discrete power law, at every integer in ks (ascending)."""
    ks = np.asarray(ks)
    # P(K >= k) = zeta(alpha, k) / zeta(alpha, kmin)
    z = special.zeta(alpha, ks + 1) / special.zeta(alpha, kmin)
    return 1.0 - z


def ks_distance(alpha, kmin, tail):
    tail = np.sort(np.asarray(tail))
    ks = np.arange(kmin, tail.max() + 1)
    emp = np.searchsorted(tail, ks, side="right") / len(tail)      # empirical P(K <= k)
    return float(np.max(np.abs(emp - pl_cdf(alpha, kmin, ks))))


def fit_tail(data, min_tail=15, kmin_candidates=None):
    """Choose k_min by KS, fit alpha there. Returns a dict."""
    data = np.asarray([d for d in data if d > 0], dtype=int)
    cands = kmin_candidates if kmin_candidates is not None else np.unique(data)
    best = None
    for kmin in cands:
        tail = data[data >= kmin]
        if len(tail) < min_tail:
            break
        alpha, se, ll = fit_powerlaw(tail, kmin)
        d = ks_distance(alpha, kmin, tail)
        if best is None or d < best["ks"]:
            best = {"kmin": int(kmin), "alpha": alpha, "alpha_se": se, "ks": d,
                    "n_tail": int(len(tail)), "loglik": ll}
    return best


def sample_powerlaw(alpha, kmin, size, rng):
    """Exact discrete draws by inverting the CDF on a table."""
    ks = np.arange(kmin, KMAX)
    pmf = ks ** (-alpha)
    pmf /= pmf.sum()
    cdf = np.cumsum(pmf)
    u = rng.uniform(size=size)
    return ks[np.searchsorted(cdf, u)]


def bootstrap_pvalue(data, fit, n_boot=500, seed=0, min_tail=15):
    """Semi-parametric bootstrap of Clauset et al.: synthetic datasets from the
    fitted model above k_min and from the empirical body below it, refitted
    from scratch (k_min included). p = fraction with KS at least as bad as the
    real fit. Small p rejects the power law."""
    rng = np.random.default_rng(seed)
    data = np.asarray([d for d in data if d > 0], dtype=int)
    body = data[data < fit["kmin"]]
    n, ntail = len(data), fit["n_tail"]
    worse = 0
    for _ in range(n_boot):
        n_from_tail = rng.binomial(n, ntail / n)
        synth = np.concatenate([
            sample_powerlaw(fit["alpha"], fit["kmin"], n_from_tail, rng),
            rng.choice(body, size=n - n_from_tail, replace=True) if len(body) else np.array([], dtype=int),
        ])
        f = fit_tail(synth, min_tail=min_tail)
        if f is None or f["ks"] >= fit["ks"]:
            worse += 1
    return (worse + 1) / (n_boot + 1)


# ------------------------------------------------------------ alternatives
def _discrete_logpmf(logf, kmin):
    """Normalise an unnormalised log-density over integers k >= kmin."""
    ks = np.arange(kmin, KMAX, dtype=float)
    lw = logf(ks)
    lz = special.logsumexp(lw)
    return lambda k: logf(np.asarray(k, dtype=float)) - lz


def fit_lognormal(tail, kmin):
    tail = np.asarray(tail, dtype=float)
    lt = np.log(tail)

    def nll(p):
        mu, sig = p
        if sig <= 0.01:
            return 1e18
        lp = _discrete_logpmf(lambda k: -((np.log(k) - mu) ** 2) / (2 * sig ** 2) - np.log(k), kmin)
        return -lp(tail).sum()

    res = optimize.minimize(nll, x0=[lt.mean(), max(lt.std(), 0.3)], method="Nelder-Mead")
    mu, sig = res.x
    lp = _discrete_logpmf(lambda k: -((np.log(k) - mu) ** 2) / (2 * sig ** 2) - np.log(k), kmin)
    return {"mu": float(mu), "sigma": float(sig), "loglik": -float(res.fun), "logpmf": lp}


def fit_exponential(tail, kmin):
    tail = np.asarray(tail, dtype=float)
    lam = 1.0 / max(tail.mean() - kmin + 0.5, 0.05)          # closed form is close enough to start

    def nll(l):
        if l <= 1e-4:
            return 1e18
        lp = _discrete_logpmf(lambda k: -l * k, kmin)
        return -lp(tail).sum()

    res = optimize.minimize_scalar(nll, bounds=(1e-4, 5.0), method="bounded")
    lp = _discrete_logpmf(lambda k: -res.x * k, kmin)
    return {"lambda": float(res.x), "loglik": -float(res.fun), "logpmf": lp}


def fit_truncated_powerlaw(tail, kmin):
    tail = np.asarray(tail, dtype=float)

    def nll(p):
        a, l = p
        if l < 0 or a < 0:
            return 1e18
        lp = _discrete_logpmf(lambda k: -a * np.log(k) - l * k, kmin)
        return -lp(tail).sum()

    res = optimize.minimize(nll, x0=[1.5, 0.05], method="Nelder-Mead")
    a, l = res.x
    lp = _discrete_logpmf(lambda k: -a * np.log(k) - l * k, kmin)
    return {"alpha": float(a), "lambda": float(l), "loglik": -float(res.fun), "logpmf": lp}


def vuong(tail, logpmf_a, logpmf_b):
    """Normalised log-likelihood ratio R and its two-sided p-value.
    R > 0 favours model a. p large: the data cannot tell them apart."""
    tail = np.asarray(tail, dtype=float)
    diff = logpmf_a(tail) - logpmf_b(tail)
    r = float(diff.sum())
    sd = float(diff.std(ddof=1))
    if sd == 0 or len(diff) < 2:
        return r, 0.0, 1.0
    rn = r / (sd * math.sqrt(len(diff)))
    p = float(special.erfc(abs(rn) / math.sqrt(2)))
    return r, rn, p


def compare(data, fit):
    """Power law against three alternatives, all fitted on the same tail."""
    data = np.asarray([d for d in data if d > 0], dtype=int)
    tail = data[data >= fit["kmin"]]
    kmin, alpha = fit["kmin"], fit["alpha"]
    pl_logpmf = lambda k: -alpha * np.log(np.asarray(k, dtype=float)) - np.log(special.zeta(alpha, kmin))
    out = {}
    ln = fit_lognormal(tail, kmin)
    ex = fit_exponential(tail, kmin)
    tp = fit_truncated_powerlaw(tail, kmin)
    for name, alt in (("lognormal", ln), ("exponential", ex)):
        r, rn, p = vuong(tail, pl_logpmf, alt["logpmf"])
        out[name] = {"R": round(r, 3), "R_norm": round(rn, 3), "p": round(p, 3),
                     **{k: round(v, 4) for k, v in alt.items() if k not in ("logpmf", "loglik")}}
    # Nested: power law is the truncated one with lambda = 0, so a chi-square with 1 dof.
    lr = 2 * (tp["loglik"] - fit["loglik"])
    out["truncated_powerlaw"] = {"LR": round(float(lr), 3), "p": round(float(stats.chi2.sf(max(lr, 0), 1)), 4),
                                 "alpha": round(tp["alpha"], 3), "lambda": round(tp["lambda"], 4)}
    return out


def ccdf(values):
    """x sorted ascending, P(K >= x). Every data point, no binning."""
    v = np.sort(np.asarray(values))
    xs = np.unique(v)
    ps = 1.0 - np.searchsorted(v, xs, side="left") / len(v)
    return xs, ps


if __name__ == "__main__":
    # Self-test: recover a known exponent and reject a Poisson sample.
    rng = np.random.default_rng(1)
    synth = sample_powerlaw(2.5, 3, 2000, rng)
    body = rng.integers(1, 3, size=500)
    f = fit_tail(np.concatenate([synth, body]))
    print("power law, true alpha 2.5, kmin 3 ->", {k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()})
    print("   bootstrap p (should be large):", bootstrap_pvalue(np.concatenate([synth, body]), f, n_boot=100, seed=1))
    print("   alternatives:", compare(np.concatenate([synth, body]), f))
    pois = rng.poisson(9.5, 303)
    f = fit_tail(pois)
    print("Poisson(9.5) n=303 ->", {k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()})
    print("   bootstrap p:", bootstrap_pvalue(pois, f, n_boot=100, seed=2))
    print("   alternatives:", compare(pois, f))
