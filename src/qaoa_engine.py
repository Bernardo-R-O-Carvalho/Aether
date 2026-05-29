"""
Aether QAOA Engine
==================

Quantum Approximate Optimization Algorithm (QAOA) for graph problems.

What is QAOA?
-------------
QAOA finds approximate solutions to combinatorial optimization problems
on quantum hardware. It was introduced by Farhi, Goldstone, and Gutmann
in 2014 as a hybrid classical-quantum algorithm.

The canonical problem: MaxCut.
Given a graph G = (V, E), partition the nodes into two sets S and S̄
to maximize the number of edges between S and S̄.

The QAOA circuit:
  |ψ(γ,β)⟩ = B(β_p) C(γ_p) ... B(β_1) C(γ_1) |+⟩^n

where:
  |+⟩^n  = Hadamard on all qubits — uniform superposition
  C(γ)   = exp(-i γ H_C) — problem unitary (encodes the graph)
  B(β)   = exp(-i β H_B) — mixing unitary (explores solutions)
  H_C    = Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2  — MaxCut cost Hamiltonian
  H_B    = Σ_i X_i — mixer Hamiltonian

The classical optimizer finds γ and β that maximize ⟨ψ|H_C|ψ⟩.
The measurement outcome gives an approximate MaxCut solution.

Why this matters:
  MaxCut is NP-hard classically. QAOA provides a quantum approach
  with provable approximation guarantees (ratio ≥ 0.6924 for p=1).
  For p → ∞, QAOA converges to the exact solution.

Aether syntax:
  graph Triangle:
      nodes 3
      edge 0 1
      edge 1 2
      edge 0 2

  infer Triangle using qaoa(layers=2, shots=1024)

This compiles automatically to the QAOA circuit — no boilerplate.
"""

import math
import cmath
import random
from collections import defaultdict
from scipy.optimize import minimize
import numpy as np


# ─────────────────────────────────────────────
#  Graph type
# ─────────────────────────────────────────────

class Graph:
    """A simple undirected graph for QAOA."""
    def __init__(self, name: str, n_nodes: int):
        self.name    = name
        self.n_nodes = n_nodes
        self.edges   = []   # list of (i, j) tuples, i < j

    def add_edge(self, i: int, j: int):
        if i > j:
            i, j = j, i
        if (i, j) not in self.edges:
            self.edges.append((i, j))

    def max_cut_classical(self):
        """
        Brute-force exact MaxCut for small graphs (n ≤ 20).
        Returns (best_cut, best_partition).
        """
        best_cut = 0
        best_partition = None
        for mask in range(1 << self.n_nodes):
            cut = sum(
                1 for i, j in self.edges
                if ((mask >> i) & 1) != ((mask >> j) & 1)
            )
            if cut > best_cut:
                best_cut = cut
                best_partition = mask
        return best_cut, best_partition

    def __repr__(self):
        return f"Graph({self.name!r}, {self.n_nodes} nodes, {len(self.edges)} edges)"


# ─────────────────────────────────────────────
#  State vector operations for QAOA
# ─────────────────────────────────────────────

def _apply_rz(state, qubit, theta, n):
    """Apply Rz(theta) = diag(e^{-iθ/2}, e^{iθ/2})."""
    dim = len(state)
    q_bit = n - 1 - qubit
    e_neg = cmath.exp(complex(0, -theta / 2))
    e_pos = cmath.exp(complex(0,  theta / 2))
    new_state = list(state)
    for i in range(dim):
        if not (i >> q_bit) & 1:
            new_state[i] = e_neg * state[i]
        else:
            new_state[i] = e_pos * state[i]
    return new_state


def _apply_rzz(state, q0, q1, theta, n):
    """
    Apply exp(-i θ/2 Z_q0 Z_q1).

    Z_q0 Z_q1 has eigenvalue +1 when q0==q1 (same bit), -1 when different.
    exp(-i θ/2 Z_q0 Z_q1)|x⟩ = exp(-i θ/2 * (-1)^(x_q0 XOR x_q1))|x⟩
    """
    dim = len(state)
    bit0 = n - 1 - q0
    bit1 = n - 1 - q1
    new_state = list(state)
    for i in range(dim):
        b0 = (i >> bit0) & 1
        b1 = (i >> bit1) & 1
        parity = b0 ^ b1   # 0 if same spin, 1 if different
        phase = cmath.exp(complex(0, -theta / 2 * (1 - 2 * parity)))
        new_state[i] = phase * state[i]
    return new_state


def _apply_rx(state, qubit, theta, n):
    """Apply Rx(theta) = cos(θ/2)I - i sin(θ/2)X."""
    dim = len(state)
    q_bit = n - 1 - qubit
    step = 1 << q_bit
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    new_state = list(state)
    for i in range(dim):
        if not (i >> q_bit) & 1:
            partner = i ^ step
            a, b = state[i], state[partner]
            new_state[i]       = complex(c) * a + complex(0, -s) * b
            new_state[partner] = complex(0, -s) * a + complex(c) * b
    return new_state


# ─────────────────────────────────────────────
#  QAOA circuit
# ─────────────────────────────────────────────

def qaoa_state(graph: Graph, gammas: list, betas: list) -> list:
    """
    Prepare the QAOA state |ψ(γ,β)⟩ for MaxCut on a graph.

    Circuit:
      1. Apply H to all qubits → |+⟩^n
      2. For each layer p:
         a. Problem unitary C(γ_p):
            For each edge (i,j): apply exp(-i γ_p/2 (1 - Z_i Z_j))
              = exp(-i γ_p/2) * exp(+i γ_p/2 Z_i Z_j)
            Global phase exp(-i γ_p/2) doesn't affect measurements,
            so we only apply the ZZ rotation.
         b. Mixing unitary B(β_p):
            For each qubit i: apply exp(-i β_p X_i) = Rx(2β_p)

    Returns the final state vector.
    """
    n = graph.n_nodes
    dim = 2 ** n
    p = len(gammas)

    # Step 1: |+⟩^n = H^⊗n |0⟩^n
    # All amplitudes equal to 1/sqrt(2^n)
    amp = complex(1.0 / math.sqrt(dim))
    state = [amp] * dim

    # Step 2: QAOA layers
    for layer in range(p):
        gamma = gammas[layer]
        beta  = betas[layer]

        # Problem unitary C(γ): edge ZZ rotations
        for i, j in graph.edges:
            state = _apply_rzz(state, i, j, gamma, n)

        # Mixing unitary B(β): Rx rotations on all qubits
        for q in range(n):
            state = _apply_rx(state, q, 2 * beta, n)

    return state


def maxcut_expectation(state: list, graph: Graph) -> float:
    """
    Compute ⟨ψ|H_C|ψ⟩ where H_C = Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2.

    For each edge (i,j), the contribution is:
      (1 - ⟨Z_i Z_j⟩) / 2

    ⟨Z_i Z_j⟩ = Σ_x |⟨x|ψ⟩|² * (-1)^(x_i XOR x_j)
    """
    n = graph.n_nodes
    dim = len(state)
    total = 0.0

    for qi, qj in graph.edges:
        bit_i = n - 1 - qi
        bit_j = n - 1 - qj
        zzexp = 0.0
        for x in range(dim):
            bi = (x >> bit_i) & 1
            bj = (x >> bit_j) & 1
            sign = 1 - 2 * (bi ^ bj)   # +1 if same, -1 if different
            zzexp += (abs(state[x]) ** 2) * sign
        total += (1 - zzexp) / 2

    return total


def sample_maxcut(state: list, graph: Graph, shots: int) -> dict:
    """
    Sample bitstrings from |ψ⟩² and compute cut values.
    Returns dict: bitstring -> count.
    """
    n = graph.n_nodes
    dim = len(state)
    probs = [abs(a) ** 2 for a in state]

    counts = defaultdict(int)
    for _ in range(shots):
        r = random.random()
        cumulative = 0.0
        outcome = 0
        for idx, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                outcome = idx
                break
        bitstring = format(outcome, f'0{n}b')
        counts[bitstring] += 1

    return dict(counts)


# ─────────────────────────────────────────────
#  QAOA optimizer
# ─────────────────────────────────────────────

def run_qaoa(graph: Graph, layers: int = 1, shots: int = 1024) -> dict:
    """
    Run QAOA for MaxCut on a graph.

    Args:
        graph:  Aether Graph object
        layers: number of QAOA layers p (more layers = better approximation)
        shots:  number of measurement shots for final sampling

    Returns dict with:
        gammas:        optimal problem parameters
        betas:         optimal mixing parameters
        max_cut_qaoa:  best cut found by QAOA
        max_cut_exact: exact MaxCut (brute force)
        approximation_ratio: max_cut_qaoa / max_cut_exact
        counts:        measurement outcome distribution
    """
    n = graph.n_nodes
    n_edges = len(graph.edges)

    print(f"\n  ⟁  Aether QAOA — '{graph.name}'")
    print(f"  {'─'*50}")
    print(f"  Nodes:   {n}")
    print(f"  Edges:   {n_edges}  {graph.edges}")
    print(f"  Layers:  {layers}  (p={layers})")
    print(f"  Qubits:  {n}")
    print(f"  Shots:   {shots}")
    print()

    # Exact MaxCut for comparison
    exact_cut, exact_partition = graph.max_cut_classical()
    exact_bits = format(exact_partition, f'0{n}b')
    print(f"  Exact MaxCut:  {exact_cut} (partition {exact_bits})")
    print()

    # Optimize QAOA parameters via BFGS
    call_count = [0]

    def neg_expectation(params):
        call_count[0] += 1
        g = list(params[:layers])
        b = list(params[layers:])
        state = qaoa_state(graph, g, b)
        return -maxcut_expectation(state, graph)   # negative for minimization

    print(f"  {'Call':>5}  {'⟨H_C⟩':>12}  {'Approx ratio':>14}")
    print(f"  {'─'*34}")

    best_result = None
    best_val = float('inf')

    # Multi-start with random initializations
    random.seed(42)
    for start in range(5):
        # Initialize γ in [0, π], β in [0, π/2]
        g0 = [random.uniform(0, math.pi) for _ in range(layers)]
        b0 = [random.uniform(0, math.pi / 2) for _ in range(layers)]
        x0 = np.array(g0 + b0)

        result = minimize(
            neg_expectation, x0,
            method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-6}
        )

        if result.fun < best_val:
            best_val = result.fun
            best_result = result

        exp_val = -result.fun
        ratio = exp_val / exact_cut if exact_cut > 0 else 0
        print(f"  {call_count[0]:>5}  {exp_val:>12.6f}  {ratio:>14.4f}")

    print(f"  {'─'*34}")
    print()

    # Extract best parameters
    best_params = best_result.x
    gammas = list(best_params[:layers])
    betas  = list(best_params[layers:])

    # Final state and sampling
    opt_state = qaoa_state(graph, gammas, betas)
    exp_val = maxcut_expectation(opt_state, graph)
    counts = sample_maxcut(opt_state, graph, shots)

    # Find best cut in samples
    best_sample_cut = 0
    best_sample_bits = None
    for bitstring, count in counts.items():
        cut = sum(
            1 for qi, qj in graph.edges
            if bitstring[qi] != bitstring[qj]
        )
        if cut > best_sample_cut:
            best_sample_cut = cut
            best_sample_bits = bitstring

    approx_ratio = exp_val / exact_cut if exact_cut > 0 else 0

    return {
        "gammas":             gammas,
        "betas":              betas,
        "expectation":        exp_val,
        "max_cut_qaoa":       best_sample_cut,
        "max_cut_exact":      exact_cut,
        "exact_partition":    exact_bits,
        "approximation_ratio": approx_ratio,
        "counts":             counts,
        "n_nodes":            n,
        "layers":             layers,
    }


def print_qaoa_results(name: str, result: dict):
    """Print QAOA results in Aether's visual style."""
    n = result["n_nodes"]
    p = result["layers"]
    exact = result["max_cut_exact"]
    qaoa  = result["max_cut_qaoa"]
    ratio = result["approximation_ratio"]

    print(f"  ⟁  QAOA Results — '{name}'")
    print(f"  {'─'*50}")
    print(f"  Layers (p):           {p}")
    print(f"  ⟨H_C⟩ (expectation):  {result['expectation']:.6f}")
    print(f"  Exact MaxCut:         {exact}  (partition {result['exact_partition']})")
    print(f"  QAOA best sample:     {qaoa}")
    print(f"  Approximation ratio:  {ratio:.4f}")
    if ratio >= 0.6924:
        print(f"  ✓  Beats QAOA p=1 guarantee (≥ 0.6924)")
    if qaoa == exact:
        print(f"  ✓  Exact solution found!")
    print()

    # Show top outcomes
    counts = result["counts"]
    shots = sum(counts.values())
    top = sorted(counts.items(), key=lambda x: -x[1])[:6]

    print(f"  Top measurement outcomes:")
    print(f"  {'Bitstring':>12}  {'Cut':>5}  {'Count':>6}  {'Prob':>7}  Bar")
    print(f"  {'─'*55}")
    for bitstring, count in top:
        cut = sum(
            1 for qi, qj in [
                (int(e[0]), int(e[1]))
                for e in result.get("edges", [])
            ]
            if bitstring[qi] != bitstring[qj]
        )
        prob = count / shots
        bar = "█" * round(prob * 30)
        print(f"  {bitstring:>12}  {count:>6}  {prob:>7.3f}  {bar}")
    print()

    # Parameters
    print(f"  Optimal parameters:")
    for k, (g, b) in enumerate(zip(result["gammas"], result["betas"])):
        print(f"    Layer {k+1}:  γ = {g:.4f}  β = {b:.4f}")
    print()
