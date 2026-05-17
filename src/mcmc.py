"""
Aether MCMC — Metropolis-Hastings sampler
==========================================

Why MCMC instead of rejection sampling?
  Rejection sampling throws random darts at the prior and keeps the ones
  that match the observations. With one variable it works fine. With three
  variables and a tight observation, the probability of a random dart landing
  in the right region drops exponentially — you'd need millions of samples
  to get a handful of accepted ones.

  Metropolis-Hastings solves this by *walking* through probability space
  instead of throwing independent darts. Each step proposes a small move
  from the current position and accepts or rejects based on how much more
  (or less) probable the new position is. This guarantees the chain
  converges to the posterior distribution — regardless of dimension.

Algorithm overview:
  1. Initialize latent variables by sampling from their priors.
  2. Propose a new state by perturbing each latent variable slightly.
  3. Compute the log acceptance ratio: log p(proposed) - log p(current).
  4. Accept the proposal with probability min(1, exp(log_accept_ratio)).
  5. Discard the first `warmup` samples (the chain hasn't found the
     high-probability region yet — these samples are biased).
  6. Return the remaining samples as draws from the posterior.

Key concepts:
  - log_joint: log p(data | params) + log p(params) — the quantity being
    maximized implicitly by the chain. We use logs for numerical stability.
  - warmup: also called burn-in. The chain starts from a random prior sample
    and needs time to find the high-probability region. Warmup samples are
    discarded because they don't represent the posterior.
  - step_size: controls how far each proposal moves. Too small: slow mixing,
    high acceptance, chain gets stuck. Too large: big jumps, low acceptance,
    chain also gets stuck. Target: 20-70% acceptance rate.
  - R-hat: convergence diagnostic. Compares variance within the chain to
    variance between two halves of the chain. R-hat ≈ 1.0 means converged.
    R-hat > 1.1 means the chain hasn't mixed — run longer or debug the model.
"""

import math
import random
from collections import defaultdict


# ─────────────────────────────────────────────
#  Log-probability densities
#
#  This is what MCMC needs that rejection sampling doesn't.
#  We need to evaluate HOW PROBABLE a value is under a distribution,
#  not just sample from it.
#
#  We use log-probabilities throughout to avoid numerical underflow.
#  Multiplying many small probabilities together quickly reaches 0 in
#  floating point — adding their logs is numerically stable.
#
#  Each function returns log p(x | params), or -inf if x is outside
#  the support of the distribution (which causes the proposal to be
#  rejected with probability 1).
# ─────────────────────────────────────────────

def _log_prob_normal(x, mean=0.0, std=1.0, **_):
    """
    Log of the Normal density: log N(x | mean, std²)
    = -0.5 * ((x - mean) / std)² - log(std) - 0.5 * log(2π)
    """
    if std <= 0: return -math.inf
    return -0.5 * ((x - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi)

def _log_prob_bernoulli(x, prob=0.5, **_):
    """
    Log of the Bernoulli PMF: log Bernoulli(x | prob)
    Clamp prob to (ε, 1-ε) to avoid log(0).
    """
    prob = max(1e-10, min(1 - 1e-10, prob))
    if x == 1:   return math.log(prob)
    elif x == 0: return math.log(1 - prob)
    return -math.inf  # x is not 0 or 1 — invalid

def _log_prob_beta(x, a=1.0, b=1.0, **_):
    """
    Log of the Beta density: log Beta(x | a, b)
    Support is (0, 1) — returns -inf outside.
    """
    if x <= 0 or x >= 1: return -math.inf
    return (a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - _log_beta_fn(a, b)

def _log_beta_fn(a, b):
    """Log of the Beta function: log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a+b)"""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def _log_prob_uniform(x, low=0.0, high=1.0, **_):
    """
    Log of the Uniform density: log U(x | low, high)
    Constant within [low, high], -inf outside.
    """
    if low >= high: return -math.inf
    if x < low or x > high: return -math.inf
    return -math.log(high - low)

def _log_prob_poisson(x, lam=1.0, **_):
    """
    Log of the Poisson PMF: log Poisson(x | lam)
    = x * log(lam) - lam - log(x!)
    Uses lgamma for log factorial: log(x!) = lgamma(x+1)
    """
    if x < 0 or not float(x).is_integer(): return -math.inf
    k = int(x)
    return k * math.log(lam) - lam - math.lgamma(k + 1)

def _log_prob_categorical(x, probs=None, **_):
    """
    Log of the Categorical PMF: log p[x]
    Returns the log probability of the chosen category.
    """
    if probs is None: probs = [0.5, 0.5]
    k = int(x)
    if k < 0 or k >= len(probs): return -math.inf
    p = max(1e-10, probs[k])
    return math.log(p)

# Registry: maps distribution names to their log-probability functions.
# Must contain every distribution supported by the interpreter.
LOG_PROB = {
    "normal":      _log_prob_normal,
    "bernoulli":   _log_prob_bernoulli,
    "beta":        _log_prob_beta,
    "uniform":     _log_prob_uniform,
    "poisson":     _log_prob_poisson,
    "categorical": _log_prob_categorical,
}

# ─────────────────────────────────────────────
#  Samplers (duplicated from interpreter.py)
#
#  Design decision: mcmc.py is intentionally self-contained.
#  We duplicate the samplers here rather than importing from interpreter.py
#  to avoid a circular dependency (interpreter imports mcmc, mcmc would
#  import interpreter). The duplication is small and the independence
#  makes the module easier to test and reuse.
# ─────────────────────────────────────────────

def _sample_normal(mean=0.0, std=1.0, **_):    return random.gauss(mean, std)
def _sample_bernoulli(prob=0.5, **_):          return 1 if random.random() < prob else 0
def _sample_beta(a=1.0, b=1.0, **_):
    if a <= 0 or b <= 0:
        raise RuntimeError(f"\n  ✗  Type error: beta(a=..., b=...) requires a > 0 and b > 0, got a={a}, b={b}\n     Hint: both shape parameters must be positive")
    return random.betavariate(a, b)
def _sample_uniform(low=0.0, high=1.0, **_):  return random.uniform(low, high)
def _sample_poisson(lam=1.0, **_):
    L, k, p = math.exp(-lam), 0, 1.0
    while p > L: k += 1; p *= random.random()
    return k - 1
def _sample_categorical(probs=None, **_):
    if probs is None: probs = [0.5, 0.5]
    r, cum = random.random(), 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum: return i
    return len(probs) - 1

SAMPLERS = {
    "normal": _sample_normal, "bernoulli": _sample_bernoulli,
    "beta": _sample_beta, "uniform": _sample_uniform,
    "poisson": _sample_poisson, "categorical": _sample_categorical,
}

# Discrete distributions need a different proposal strategy (see propose()).
DISCRETE_DISTS = {"bernoulli", "poisson", "categorical"}


# ─────────────────────────────────────────────
#  Model executor
#
#  Given a model body (list of AST nodes) and a complete variable
#  assignment (env dict), compute the log joint probability of
#  that assignment under the model.
#
#  This is the core of MCMC: we need to score any proposed state.
#  log_joint = Σ log p(latent_var | prior) + Σ log p(observed | params)
# ─────────────────────────────────────────────

def execute_model(body, env_in, eval_expr_fn):
    """
    Score a complete variable assignment against the model.

    Args:
        body:         list of AST statement nodes (the model body)
        env_in:       dict mapping variable names to their current values
        eval_expr_fn: function(node, env) -> value from the interpreter

    Returns:
        (env, log_joint) where log_joint is the log probability of this
        assignment under the model's joint distribution.
    """
    env = dict(env_in)
    log_joint = 0.0

    for stmt in body:
        kind = stmt[0]

        if kind == "sample":
            name, dist_node = stmt[1], stmt[2]
            dist_name, kwargs_nodes = dist_node[1], dist_node[2]
            kwargs = {k: eval_expr_fn(v, env) for k, v in kwargs_nodes.items()}

            # If the variable already has a value (from the current state),
            # score it under the prior. Otherwise sample from the prior.
            val = env.get(name)
            if val is None:
                fn = SAMPLERS.get(dist_name)
                if not fn: raise RuntimeError(f"Unknown distribution: '{dist_name}'")
                val = fn(**kwargs)
                env[name] = val

            lp_fn = LOG_PROB.get(dist_name)
            if not lp_fn: raise RuntimeError(f"No log_prob for '{dist_name}'")
            log_joint += lp_fn(val, **kwargs)  # add log prior

        elif kind == "assign":
            # Deterministic — no probability contribution.
            name, expr = stmt[1], stmt[2]
            env[name] = eval_expr_fn(expr, env)

        elif kind == "observe":
            # Score observed value(s) under the likelihood.
            # For multiple observations, score each one independently
            # (assuming i.i.d. — each observation contributes additively
            # to the log joint).
            name, expr = stmt[1], stmt[2]
            observed_val = eval_expr_fn(expr, env)
            observations = observed_val if isinstance(observed_val, list) else [observed_val]
            dist_node = _find_dist(body, name)
            if dist_node is None:
                env[name] = observations[0]; continue
            dist_name, kwargs_nodes = dist_node[1], dist_node[2]
            kwargs = {k: eval_expr_fn(v, env) for k, v in kwargs_nodes.items()}
            lp_fn = LOG_PROB.get(dist_name)
            if not lp_fn: raise RuntimeError(f"No log_prob for '{dist_name}'")
            for obs in observations:
                log_joint += lp_fn(obs, **kwargs)  # add log likelihood
            env[name] = observations[0]

    return env, log_joint


def _find_dist(body, var_name):
    """Find the distribution node for a variable by scanning the model body."""
    for stmt in body:
        if stmt[0] == "sample" and stmt[1] == var_name:
            return stmt[2]
    return None


# ─────────────────────────────────────────────
#  Proposal distribution
#
#  The proposal decides WHERE to move next in parameter space.
#  A good proposal balances exploration (moving far) and exploitation
#  (staying near high-probability regions).
#
#  We use two strategies:
#    Continuous variables: Gaussian random walk around the current value.
#      The step size is adapted to the variable's scale (its std or range).
#      Constraints (beta must stay in (0,1)) are enforced by clamping.
#
#    Discrete variables: resample from the prior.
#      Random walk doesn't make sense for {0,1} or category indices.
#      Resampling from prior is simple and works for low-cardinality vars.
#      (For high-cardinality discrete models, more sophisticated proposals
#      like Gibbs sampling would be better — a future improvement.)
# ─────────────────────────────────────────────

def propose(name, current_val, dist_name, kwargs, step_size=0.3):
    """
    Generate a proposed new value for a latent variable.

    Args:
        name:        variable name (unused, kept for future debugging)
        current_val: the variable's current value in the chain
        dist_name:   the distribution this variable was sampled from
        kwargs:      the distribution's parameter values
        step_size:   scale of the Gaussian random walk (relative to variable scale)

    Returns:
        A new proposed value.
    """
    if dist_name in DISCRETE_DISTS:
        # Discrete: resample from prior — simple but effective
        fn = SAMPLERS.get(dist_name)
        return fn(**kwargs) if fn else current_val
    else:
        # Continuous: Gaussian random walk
        # Scale the step to the variable's natural scale to avoid tiny
        # or enormous steps regardless of parameter magnitude.
        scale = kwargs.get("std", kwargs.get("high", 1.0) - kwargs.get("low", 0.0)) or 1.0
        proposal = current_val + random.gauss(0, step_size * max(abs(scale), 0.1))

        # Enforce support constraints to keep proposals valid
        if dist_name == "beta":
            # Beta must be strictly inside (0, 1)
            proposal = max(1e-6, min(1 - 1e-6, proposal))
        elif dist_name == "uniform":
            lo, hi = kwargs.get("low", 0.0), kwargs.get("high", 1.0)
            proposal = max(lo + 1e-10, min(hi - 1e-10, proposal))

        return proposal


# ─────────────────────────────────────────────
#  Metropolis-Hastings sampler
# ─────────────────────────────────────────────

def run_mcmc(body, eval_expr_fn, samples=2000, warmup=500, step_size=0.3):
    """
    Run Metropolis-Hastings MCMC on an Aether model.

    The algorithm guarantees convergence to the true posterior distribution
    under mild conditions (the chain must be able to reach any state from
    any other state in finite steps — satisfied here by Gaussian proposals).

    Args:
        body:         model body (list of AST nodes from the parser)
        eval_expr_fn: expression evaluator from the interpreter
        samples:      number of post-warmup samples to collect
        warmup:       number of initial samples to discard
        step_size:    proposal step size (tune for 20-70% acceptance rate)

    Returns:
        dict with keys: samples, accepted, total, latent_vars
    """

    # Identify latent variables: those that are sampled (~) but not observed.
    # Observed variables are conditioned on — they're not inferred.
    observed_vars = {stmt[1] for stmt in body if stmt[0] == "observe"}
    latent_vars = []
    dist_map = {}   # var_name -> (dist_name, kwargs_nodes)

    for stmt in body:
        if stmt[0] == "sample":
            name = stmt[1]
            dist_node = stmt[2]
            dist_name, kwargs_nodes = dist_node[1], dist_node[2]
            dist_map[name] = (dist_name, kwargs_nodes)
            if name not in observed_vars:
                latent_vars.append(name)

    if not latent_vars:
        print("  ✗  No latent variables to infer.")
        return {}

    # Initialize the chain by sampling all variables from their priors.
    # The chain will move toward the posterior during warmup.
    current_env = {}
    for stmt in body:
        if stmt[0] == "sample":
            name, dist_node = stmt[1], stmt[2]
            dist_name, kwargs_nodes = dist_node[1], dist_node[2]
            kwargs = {k: eval_expr_fn(v, current_env) for k, v in kwargs_nodes.items()}
            fn = SAMPLERS.get(dist_name)
            current_env[name] = fn(**kwargs) if fn else 0.0
        elif stmt[0] == "assign":
            current_env[stmt[1]] = eval_expr_fn(stmt[2], current_env)

    # Score the initial state
    _, current_log_prob = execute_model(body, current_env, eval_expr_fn)

    # Main MCMC loop
    all_samples = []
    accepted_count = 0
    total_steps = samples + warmup

    for step in range(total_steps):
        # Propose a new state by perturbing each latent variable
        proposed_env = dict(current_env)
        for name in latent_vars:
            dist_name, kwargs_nodes = dist_map[name]
            kwargs = {k: eval_expr_fn(v, current_env) for k, v in kwargs_nodes.items()}
            proposed_env[name] = propose(
                name, current_env[name], dist_name, kwargs, step_size
            )

        # Score the proposed state
        _, proposed_log_prob = execute_model(body, proposed_env, eval_expr_fn)

        # Metropolis acceptance criterion:
        # Accept if proposed is more probable than current (log_accept > 0).
        # Accept with probability exp(log_accept) if less probable.
        # This asymmetry allows the chain to explore lower-probability regions
        # and is what guarantees convergence to the correct posterior.
        log_accept = proposed_log_prob - current_log_prob
        if math.log(random.random() + 1e-300) < log_accept:
            current_env = proposed_env
            current_log_prob = proposed_log_prob
            if step >= warmup:
                accepted_count += 1

        # Only collect post-warmup samples
        if step >= warmup:
            snapshot = {k: v for k, v in current_env.items() if not k.startswith("_")}
            all_samples.append(snapshot)

    return {
        "samples":     all_samples,
        "accepted":    accepted_count,
        "total":       samples,
        "latent_vars": latent_vars,
    }


# ─────────────────────────────────────────────
#  Diagnostics
#
#  Two diagnostics are computed for every variable:
#
#  1. Posterior statistics: mean, std, median, 90% credible interval.
#     The credible interval [lo, hi] means the true value lies in that
#     range with 90% posterior probability — different from a frequentist
#     confidence interval.
#
#  2. R-hat (Gelman-Rubin convergence diagnostic):
#     We split the chain in half and treat each half as a separate chain.
#     If the chain has mixed well, both halves should look like draws from
#     the same distribution — their between-half variance should match their
#     within-half variance. R-hat measures this ratio.
#     R-hat ≈ 1.0 → converged.  R-hat > 1.1 → run longer or debug model.
# ─────────────────────────────────────────────

def compute_stats(samples, var_name):
    """Compute posterior statistics for a single variable across all samples."""
    vals = [s[var_name] for s in samples if var_name in s]
    if not vals: return None

    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(variance)
    sorted_vals = sorted(vals)
    lo = sorted_vals[int(n * 0.05)]   # 5th percentile
    hi = sorted_vals[int(n * 0.95)]   # 95th percentile
    median = sorted_vals[n // 2]

    return {"mean": mean, "std": std, "median": median, "lo": lo, "hi": hi, "n": n}


def compute_rhat(samples, var_name):
    """
    Compute the Gelman-Rubin R-hat diagnostic using a split-chain approach.
    Splits the chain in half, then compares between-half and within-half variance.
    R-hat = sqrt(var_hat / W) where var_hat is a pooled variance estimate.
    """
    vals = [s[var_name] for s in samples if var_name in s]
    if len(vals) < 4: return None

    mid = len(vals) // 2
    chains = [vals[:mid], vals[mid:]]
    n = mid

    chain_means = [sum(c) / len(c) for c in chains]
    grand_mean = sum(chain_means) / 2

    # Between-chain variance B
    B = n * sum((cm - grand_mean) ** 2 for cm in chain_means)

    # Within-chain variance W
    W = sum(
        sum((v - cm) ** 2 for v in c) / (n - 1)
        for c, cm in zip(chains, chain_means)
    ) / 2

    if W < 1e-10: return 1.0   # chain is constant — degenerate case

    # Pooled posterior variance estimate
    var_hat = (1 - 1/n) * W + B/n
    return math.sqrt(var_hat / W)


# ─────────────────────────────────────────────
#  ASCII Histogram
#
#  Visualizes the posterior distribution of a variable.
#  For discrete variables: counts per unique value.
#  For continuous variables: counts in evenly spaced bins.
#
#  Design decision: ASCII output works in any terminal, any platform,
#  and is copy-pasteable into documentation. A matplotlib plot would
#  be richer but adds a dependency and breaks in headless environments.
# ─────────────────────────────────────────────

def ascii_histogram(vals, bins=10, width=28):
    """Render a horizontal ASCII histogram of posterior samples."""
    if not vals: return

    # Detect discrete vs continuous by checking if all values are small integers
    is_discrete = all(float(v).is_integer() and abs(v) < 100 for v in vals)

    if is_discrete:
        counts = {}
        for v in vals:
            k = int(v)
            counts[k] = counts.get(k, 0) + 1
        sorted_keys = sorted(counts.keys())
        buckets = [(str(k), counts[k]) for k in sorted_keys]
    else:
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-10:
            print(f"    {'█' * width}  (constant)")
            return
        step = (hi - lo) / bins
        bucket_counts = [0] * bins
        for v in vals:
            idx = min(int((v - lo) / step), bins - 1)
            bucket_counts[idx] += 1
        buckets = [
            (f"{lo + i*step:.3f}", bucket_counts[i])
            for i in range(bins)
        ]

    max_count = max(c for _, c in buckets) or 1
    n = len(vals)

    for label, count in buckets:
        filled = round(count / max_count * width)
        empty  = width - filled
        bar    = "█" * filled + "░" * empty
        pct    = count / n * 100
        print(f"    {label:>8}  {bar}  {pct:4.1f}%")


# ─────────────────────────────────────────────
#  ASCII Trace Plot
#
#  Shows the trajectory of the MCMC chain over iterations.
#  This is the primary tool for diagnosing whether the sampler
#  is working correctly.
#
#  Good chain: values jump freely across the full range of the posterior.
#  Bad chain (stuck): values stay flat for long stretches — the chain is
#    trapped in a local mode. Fix: reduce step_size or run longer warmup.
#  Bad chain (random walk): values drift slowly — the chain hasn't mixed.
#    Fix: increase step_size.
#
#  We downsample to `width` columns so the plot fits the terminal.
#  The y-axis shows min, midpoint, and max of the sampled range.
# ─────────────────────────────────────────────

def ascii_trace(vals, width=48, height=6):
    """Render a vertical ASCII trace plot of the MCMC chain trajectory."""
    if not vals or len(vals) < 4:
        return

    n = len(vals)
    lo, hi = min(vals), max(vals)

    if abs(hi - lo) < 1e-10:
        print(f"    [chain constant at {lo:.4f} — check model]")
        return

    # Downsample: pick one sample per display column
    step = max(1, n // width)
    sampled = [vals[i] for i in range(0, n, step)][:width]
    cols = len(sampled)

    # Build a 2D character grid
    grid = [[" "] * cols for _ in range(height)]
    for col, val in enumerate(sampled):
        # Map value to row index; flip so high values appear at top
        row = int((val - lo) / (hi - lo) * (height - 1))
        row = height - 1 - row
        grid[row][col] = "·"

    # Print with labeled y-axis
    for i, row in enumerate(grid):
        if i == 0:
            label = f"{hi:7.3f} │"
        elif i == height - 1:
            label = f"{lo:7.3f} │"
        elif i == height // 2:
            label = f"{(hi+lo)/2:7.3f} │"
        else:
            label = f"        │"
        print(f"    {label}{''.join(row)}")

    print(f"    {'':7s} └{'─' * cols}")
    print(f"    {'':7s}  iter 1{' ' * (cols - 12)}iter {n:,}")


# ─────────────────────────────────────────────
#  Output formatter
#
#  Combines all diagnostics into a readable terminal report.
#  For each latent variable:
#    1. Posterior histogram — shape of the distribution
#    2. Summary statistics — mean, std, median, 90% CI, R-hat
#    3. Trace plot (continuous vars only) — chain health
# ─────────────────────────────────────────────

def print_mcmc_results(model_name, result, warmup):
    samples     = result["samples"]
    accepted    = result["accepted"]
    total       = result["total"]
    latent_vars = result["latent_vars"]

    acceptance_rate = accepted / total * 100 if total > 0 else 0

    print(f"\n  ⟁  Aether MCMC — '{model_name}'")
    print(f"  {'─'*50}")
    print(f"  Algorithm  : Metropolis-Hastings")
    print(f"  Samples    : {total:,}  (warmup discarded: {warmup:,})")
    print(f"  Acceptance : {acceptance_rate:.1f}%", end="")

    # Acceptance rate guidance:
    # < 10%: step_size too large — proposals land in low-probability regions
    # > 80%: step_size too small — chain moves slowly, poor mixing
    # 20-70%: healthy range for Metropolis-Hastings
    if acceptance_rate < 10:
        print("  ⚠  very low — try smaller step_size")
    elif acceptance_rate > 80:
        print("  ⚠  very high — try larger step_size")
    else:
        print("  ✓")

    print()

    for var in latent_vars:
        stats = compute_stats(samples, var)
        if not stats: continue

        rhat = compute_rhat(samples, var)
        rhat_str = ""
        if rhat is not None:
            marker = "✓" if rhat < 1.1 else "⚠ not converged"
            rhat_str = f"   R-hat = {rhat:.3f} {marker}"

        vals = [s[var] for s in samples if var in s]

        print(f"  {var}  — posterior distribution")
        print()
        ascii_histogram(vals)
        print()
        print(f"    mean   = {stats['mean']:.4f}   std = {stats['std']:.4f}   median = {stats['median']:.4f}")
        print(f"    90% CI = [{stats['lo']:.4f}, {stats['hi']:.4f}]{rhat_str}")
        print()

        # Trace plot only for continuous variables.
        # Discrete variables (0/1, counts) produce unreadable trace plots
        # because they only occupy a handful of y positions.
        is_discrete = all(float(v).is_integer() and abs(v) < 100 for v in vals)
        if not is_discrete:
            print(f"  {var}  — trace plot  (good chain: values jump freely)")
            print()
            ascii_trace(vals)
            print()
