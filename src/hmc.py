"""
Aether HMC — Hamiltonian Monte Carlo sampler
=============================================

Why HMC over Metropolis-Hastings?
  MH proposes random steps in any direction — it walks blind.
  HMC simulates physics: a ball rolling on a surface shaped like the
  log-probability landscape. The gradient tells the ball which way is
  downhill. The ball rolls for L leapfrog steps, then we accept/reject.

  Result: HMC explores the posterior in far fewer steps, handles high-
  dimensional models (hierarchical models with many variables) much better,
  and achieves much higher acceptance rates (typically 60-90%).

The algorithm:
  1. Sample a random momentum p ~ N(0, I) for each latent variable.
  2. Simulate Hamiltonian dynamics for L leapfrog steps:
       p ← p - (ε/2) * ∇U(q)       # half step for momentum
       q ← q + ε * p                # full step for position
       p ← p - (ε/2) * ∇U(q)       # half step for momentum
     where U(q) = -log_joint(q) is the potential energy.
  3. Accept the proposal with probability min(1, exp(-ΔH))
     where H = U(q) + K(p) is the total Hamiltonian energy.

Gradient computation:
  - With JAX: exact automatic differentiation. Fast and numerically precise.
  - Without JAX: central finite differences. ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / 2ε
    Slower (2N model evaluations per gradient) but mathematically correct.

Usage in .aeth:
  infer MyModel using hmc(samples=2000, warmup=500, step_size=0.1, steps=10)
"""

import math
import random
from collections import defaultdict

# ─────────────────────────────────────────────
#  JAX detection
#
#  Try to import JAX for automatic differentiation.
#  If unavailable, fall back to numerical gradients.
#  This keeps Aether zero-dependency for basic use while enabling
#  full performance for users who have JAX installed.
# ─────────────────────────────────────────────

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# ─────────────────────────────────────────────
#  Import shared infrastructure from mcmc.py
# ─────────────────────────────────────────────

from mcmc import (
    LOG_PROB, SAMPLERS, DISCRETE_DISTS,
    execute_model, _find_dist, _find_indexed_dist,
    _collect_latent_vars, _init_env,
    compute_stats, compute_rhat,
    ascii_histogram, ascii_trace,
)


# ─────────────────────────────────────────────
#  Log joint probability function
#
#  Wraps execute_model to return a scalar log-probability
#  given a flat vector of latent variable values.
#  This is what we differentiate to get gradients.
# ─────────────────────────────────────────────

def make_log_joint_fn(body, latent_vars, eval_expr_fn, current_env):
    """
    Build a function: values_dict -> log_joint_scalar
    Used by both JAX autodiff and numerical gradient.

    Args:
        body:         model body (AST nodes)
        latent_vars:  list of latent variable keys
        eval_expr_fn: expression evaluator from interpreter
        current_env:  current environment (for non-latent variables)

    Returns:
        A callable that takes a dict {var: value} and returns log_joint.
    """
    def log_joint(values):
        env = dict(current_env)
        env.update(values)
        _, lp = execute_model(body, env, eval_expr_fn)
        return lp
    return log_joint


# ─────────────────────────────────────────────
#  Gradient computation
#
#  Two implementations depending on JAX availability.
#  Both return a dict {var_name: gradient_value}.
# ─────────────────────────────────────────────

def compute_gradient_numerical(log_joint_fn, values, eps=1e-4):
    """
    Compute gradient via central finite differences.

    ∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)

    This requires 2N evaluations of the model (N = number of latent vars).
    Less efficient than JAX but works with zero dependencies.

    Args:
        log_joint_fn: callable(dict) -> float
        values:       dict {var: current_value}
        eps:          finite difference step size

    Returns:
        dict {var: gradient}
    """
    gradients = {}
    base_lp = log_joint_fn(values)

    for var in values:
        # Perturb positively
        values_plus = dict(values)
        values_plus[var] = values[var] + eps
        lp_plus = log_joint_fn(values_plus)

        # Perturb negatively
        values_minus = dict(values)
        values_minus[var] = values[var] - eps
        lp_minus = log_joint_fn(values_minus)

        # Central difference
        gradients[var] = (lp_plus - lp_minus) / (2 * eps)

    return gradients


def compute_gradient_jax(log_joint_fn, values):
    """
    Compute gradient via JAX automatic differentiation.

    JAX traces the computation graph and computes exact derivatives.
    This is equivalent to symbolic differentiation but works for any
    Python function that uses JAX-compatible operations.

    Requires: pip install jax jaxlib

    Args:
        log_joint_fn: callable(dict) -> float (must use jnp operations)
        values:       dict {var: current_value}

    Returns:
        dict {var: gradient}
    """
    # Convert to JAX arrays
    jax_values = {k: jnp.array(float(v)) for k, v in values.items()}

    # Build a function that takes a flat array for grad computation
    var_names = list(values.keys())

    def flat_log_joint(flat_vals):
        d = {var_names[i]: flat_vals[i] for i in range(len(var_names))}
        return log_joint_fn(d)

    flat_vals = jnp.array([float(values[k]) for k in var_names])
    grads = grad(flat_log_joint)(flat_vals)

    return {var_names[i]: float(grads[i]) for i in range(len(var_names))}


def compute_gradient(log_joint_fn, values):
    """
    Compute gradient using JAX if available, numerical otherwise.
    This is the main entry point for gradient computation in HMC.
    """
    if JAX_AVAILABLE:
        try:
            return compute_gradient_jax(log_joint_fn, values)
        except Exception:
            # JAX failed (e.g. non-differentiable operation) — fall back
            return compute_gradient_numerical(log_joint_fn, values)
    else:
        return compute_gradient_numerical(log_joint_fn, values)


# ─────────────────────────────────────────────
#  Constraint handling
#
#  HMC operates in unconstrained space. Variables with bounded support
#  (beta: [0,1], uniform: [low, high]) must be transformed to (-∞, +∞)
#  so the leapfrog integrator can move freely.
#
#  We use the logit transform for [0,1] variables:
#    unconstrained = log(x / (1 - x))   (logit)
#    constrained   = 1 / (1 + exp(-u))  (sigmoid)
#
#  The log-jacobian of the transform must be added to the log-joint
#  to account for the change of variables.
# ─────────────────────────────────────────────

def to_unconstrained(val, dist_name, kwargs):
    """Transform a constrained value to unconstrained space."""
    if dist_name == "beta":
        # [0,1] -> (-inf, inf) via logit
        val = max(1e-6, min(1 - 1e-6, val))
        return math.log(val / (1 - val))
    elif dist_name == "uniform":
        lo, hi = kwargs.get("low", 0.0), kwargs.get("high", 1.0)
        val = max(lo + 1e-10, min(hi - 1e-10, val))
        # [lo, hi] -> (-inf, inf) via scaled logit
        p = (val - lo) / (hi - lo)
        p = max(1e-6, min(1 - 1e-6, p))
        return math.log(p / (1 - p))
    else:
        # Unconstrained (normal, poisson treated as continuous here)
        return val


def to_constrained(u, dist_name, kwargs):
    """Transform an unconstrained value back to constrained space."""
    if dist_name == "beta":
        # (-inf, inf) -> (0, 1) via sigmoid
        return 1.0 / (1.0 + math.exp(-u))
    elif dist_name == "uniform":
        lo, hi = kwargs.get("low", 0.0), kwargs.get("high", 1.0)
        p = 1.0 / (1.0 + math.exp(-u))
        return lo + p * (hi - lo)
    else:
        return u


def log_jacobian(u, dist_name, kwargs):
    """
    Log |d(constrained)/d(unconstrained)| — the change-of-variables correction.
    Must be added to log_joint when operating in unconstrained space.
    """
    if dist_name == "beta":
        # Jacobian of sigmoid: σ(u) * (1 - σ(u))
        s = 1.0 / (1.0 + math.exp(-u))
        return math.log(max(s * (1 - s), 1e-30))
    elif dist_name == "uniform":
        lo, hi = kwargs.get("low", 0.0), kwargs.get("high", 1.0)
        s = 1.0 / (1.0 + math.exp(-u))
        return math.log(max(s * (1 - s) * (hi - lo), 1e-30))
    else:
        return 0.0  # no transformation, no Jacobian


# ─────────────────────────────────────────────
#  Leapfrog integrator
#
#  Simulates Hamiltonian dynamics using the leapfrog (Störmer-Verlet)
#  method. This is a symplectic integrator — it preserves the volume
#  of phase space and is time-reversible, both essential for correctness.
#
#  Leapfrog steps:
#    p(t + ε/2) = p(t) - (ε/2) ∇U(q(t))       # half-step momentum
#    q(t + ε)   = q(t) + ε * p(t + ε/2)        # full-step position
#    p(t + ε)   = p(t + ε/2) - (ε/2) ∇U(q(t+ε)) # half-step momentum
#
#  where U(q) = -log_joint(q) is the potential energy.
# ─────────────────────────────────────────────

def leapfrog(q, p, log_joint_fn, step_size, n_steps):
    """
    Run L leapfrog steps to simulate Hamiltonian dynamics.

    Args:
        q:            dict {var: position (unconstrained)}
        p:            dict {var: momentum}
        log_joint_fn: callable(dict) -> log_probability
        step_size:    ε — leapfrog step size
        n_steps:      L — number of leapfrog steps

    Returns:
        (q_new, p_new) — proposed position and momentum
    """
    q = dict(q)
    p = dict(p)

    # Initial half-step for momentum
    grads = compute_gradient(log_joint_fn, q)
    for var in p:
        # Gradient of log_joint is -gradient of potential energy U
        # p update: p = p + (ε/2) * ∇log_joint  (note: +, not -)
        p[var] = p[var] + (step_size / 2) * grads.get(var, 0.0)

    # Alternate full steps
    for i in range(n_steps - 1):
        # Full step for position
        for var in q:
            q[var] = q[var] + step_size * p[var]

        # Full step for momentum (using gradient at new position)
        grads = compute_gradient(log_joint_fn, q)
        for var in p:
            p[var] = p[var] + step_size * grads.get(var, 0.0)

    # Final full step for position
    for var in q:
        q[var] = q[var] + step_size * p[var]

    # Final half-step for momentum
    grads = compute_gradient(log_joint_fn, q)
    for var in p:
        p[var] = p[var] + (step_size / 2) * grads.get(var, 0.0)

    return q, p


# ─────────────────────────────────────────────
#  HMC sampler
# ─────────────────────────────────────────────

def run_hmc(body, eval_expr_fn, samples=1000, warmup=500,
            step_size=0.1, n_steps=10):
    """
    Run Hamiltonian Monte Carlo on an Aether model.

    HMC is dramatically more efficient than MH for:
    - High-dimensional models (many latent variables)
    - Hierarchical models with correlated parameters
    - Models where MH gets low acceptance rates

    Args:
        body:         model body (AST nodes)
        eval_expr_fn: expression evaluator from interpreter
        samples:      number of post-warmup samples
        warmup:       warmup samples to discard
        step_size:    ε — leapfrog step size (tune for 60-90% acceptance)
        n_steps:      L — number of leapfrog steps per proposal

    Returns:
        dict with keys: samples, accepted, total, latent_vars, backend
    """
    backend = "jax" if JAX_AVAILABLE else "numerical"

    # Initialize environment
    current_env = _init_env(body, eval_expr_fn)

    # Collect latent variables
    latent_info, observed_keys = _collect_latent_vars(body, eval_expr_fn, current_env)

    # Filter out discrete variables — HMC only works for continuous vars.
    # Discrete variables fall back to MH proposals.
    continuous_latent = []
    discrete_latent = []
    dist_map = {}

    for key, dist_name, kwargs_nodes in latent_info:
        dist_map[key] = (dist_name, kwargs_nodes)
        if dist_name in DISCRETE_DISTS:
            discrete_latent.append(key)
        else:
            continuous_latent.append(key)

    if not continuous_latent and not discrete_latent:
        print("  ✗  No latent variables to infer.")
        return {}

    latent_vars = continuous_latent + discrete_latent

    # Build the log_joint function for continuous variables
    def log_joint_fn(q_unconstrained):
        """
        Evaluate log_joint given unconstrained values.
        Transforms back to constrained space, adds Jacobian correction.
        """
        env = dict(current_env)
        lj = 0.0

        for key in continuous_latent:
            dist_name, kwargs_nodes = dist_map[key]
            kwargs = {k: eval_expr_fn(v, env) for k, v in kwargs_nodes.items()}
            u = q_unconstrained.get(key, 0.0)
            constrained = to_constrained(u, dist_name, kwargs)
            env[key] = constrained
            lj += log_jacobian(u, dist_name, kwargs)

        # Keep discrete vars at current values
        for key in discrete_latent:
            env[key] = current_env.get(key, 0)

        _, model_lp = execute_model(body, env, eval_expr_fn)
        return model_lp + lj

    # Initialize unconstrained position
    q_current = {}
    for key in continuous_latent:
        dist_name, kwargs_nodes = dist_map[key]
        kwargs = {k: eval_expr_fn(v, current_env) for k, v in kwargs_nodes.items()}
        val = current_env.get(key, SAMPLERS.get(dist_name, lambda **_: 0.0)(**kwargs))
        q_current[key] = to_unconstrained(val, dist_name, kwargs)

    current_lp = log_joint_fn(q_current)

    all_samples = []
    accepted_count = 0
    total_steps = samples + warmup

    print(f"\n  ⟁  Aether HMC — starting")
    print(f"  Backend    : {'JAX (autodiff)' if JAX_AVAILABLE else 'numerical gradients (install jax for better performance)'}")
    print(f"  Variables  : {len(continuous_latent)} continuous, {len(discrete_latent)} discrete")
    print(f"  Leapfrog   : {n_steps} steps × ε={step_size}")

    for step in range(total_steps):
        # Sample fresh momentum from N(0, I) — kinetic energy
        p_current = {var: random.gauss(0, 1) for var in continuous_latent}

        # Current Hamiltonian: H = -log_joint + 0.5 * ||p||²
        kinetic_current = 0.5 * sum(p ** 2 for p in p_current.values())
        H_current = -current_lp + kinetic_current

        # Leapfrog trajectory
        try:
            q_proposed, p_proposed = leapfrog(
                q_current, p_current, log_joint_fn, step_size, n_steps
            )
            proposed_lp = log_joint_fn(q_proposed)
        except (ValueError, OverflowError, ZeroDivisionError):
            # Numerical instability — reject
            q_proposed = q_current
            proposed_lp = -math.inf

        kinetic_proposed = 0.5 * sum(p ** 2 for p in p_proposed.values()) if q_proposed is not q_current else math.inf
        H_proposed = -proposed_lp + kinetic_proposed

        # Metropolis acceptance criterion on Hamiltonian energy
        # Accept with probability min(1, exp(H_current - H_proposed))
        delta_H = H_current - H_proposed
        if math.log(random.random() + 1e-300) < delta_H:
            q_current = q_proposed
            current_lp = proposed_lp
            if step >= warmup:
                accepted_count += 1

        # Handle discrete variables with MH proposals
        for key in discrete_latent:
            dist_name, kwargs_nodes = dist_map[key]
            kwargs = {k: eval_expr_fn(v, current_env) for k, v in kwargs_nodes.items()}
            fn = SAMPLERS.get(dist_name)
            if fn:
                proposal = fn(**kwargs)
                env_prop = dict(current_env)
                env_prop[key] = proposal
                _, prop_lp = execute_model(body, env_prop, eval_expr_fn)
                _, cur_lp = execute_model(body, current_env, eval_expr_fn)
                if math.log(random.random() + 1e-300) < prop_lp - cur_lp:
                    current_env[key] = proposal

        # Collect post-warmup samples — transform back to constrained space
        if step >= warmup:
            snapshot = {}
            for key in continuous_latent:
                dist_name, kwargs_nodes = dist_map[key]
                kwargs = {k: eval_expr_fn(v, current_env) for k, v in kwargs_nodes.items()}
                snapshot[key] = to_constrained(q_current[key], dist_name, kwargs)
            for key in discrete_latent:
                snapshot[key] = current_env.get(key, 0)
            all_samples.append(snapshot)

    return {
        "samples":     all_samples,
        "accepted":    accepted_count,
        "total":       samples,
        "latent_vars": latent_vars,
        "backend":     backend,
    }


# ─────────────────────────────────────────────
#  Output formatter
# ─────────────────────────────────────────────

def print_hmc_results(model_name, result, warmup):
    samples     = result["samples"]
    accepted    = result["accepted"]
    total       = result["total"]
    latent_vars = result["latent_vars"]
    backend     = result.get("backend", "unknown")

    acceptance_rate = accepted / total * 100 if total > 0 else 0

    print(f"\n  ⟁  Aether HMC — '{model_name}'")
    print(f"  {'─'*50}")
    print(f"  Algorithm  : Hamiltonian Monte Carlo")
    print(f"  Backend    : {backend}")
    print(f"  Samples    : {total:,}  (warmup discarded: {warmup:,})")
    print(f"  Acceptance : {acceptance_rate:.1f}%", end="")

    # HMC targets 60-90% acceptance (higher than MH's 20-70%)
    if acceptance_rate < 40:
        print("  ⚠  low — try smaller step_size or fewer leapfrog steps")
    elif acceptance_rate > 95:
        print("  ⚠  very high — try larger step_size or more leapfrog steps")
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

        is_discrete = all(float(v).is_integer() and abs(v) < 100 for v in vals)
        if not is_discrete:
            print(f"  {var}  — trace plot  (HMC should mix faster than MH)")
            print()
            ascii_trace(vals)
            print()
