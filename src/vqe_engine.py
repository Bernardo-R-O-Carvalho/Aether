"""
Aether VQE Engine — Closed classical-quantum optimization loop
==============================================================

What is VQE?
  The Variational Quantum Eigensolver (VQE) finds the ground state energy
  of a Hamiltonian H by minimizing ⟨ψ(θ)|H|ψ(θ)⟩ over circuit parameters θ.

  The variational principle guarantees:
    ⟨ψ(θ)|H|ψ(θ)⟩ ≥ E_ground  for all θ

  So the minimum over θ is the best achievable approximation to E_ground
  given the ansatz circuit family.

The hybrid loop:
  ┌─────────────────────────────────────────────┐
  │  Classical computer                         │
  │    θ → parameter update (gradient descent)  │
  │              ↑                              │
  │         energy E(θ)                         │
  │              ↑                              │
  │  Quantum computer                           │
  │    prepare |ψ(θ)⟩, measure ⟨H⟩             │
  └─────────────────────────────────────────────┘

Optimizer:
  We use gradient descent with numerical gradients (parameter shift rule).
  The parameter shift rule is the standard method for quantum circuits:
    ∂E/∂θ_k ≈ [E(θ + π/2 * eₖ) - E(θ - π/2 * eₖ)] / 2

  This is exact (not approximate) for Pauli rotation gates — an important
  property that makes it compatible with real quantum hardware.

Usage in .aeth:
  infer Ansatz using vqe(
      hamiltonian = H2,
      shots       = 1024,
      iterations  = 80,
      step_size   = 0.3
  )
"""

import math
import random
from hamiltonian import (
    Hamiltonian, prepare_ansatz_state, exact_energy,
    measure_energy_shots, print_hamiltonian
)


# ─────────────────────────────────────────────
#  Parameter shift rule gradient
#
#  For a circuit with Pauli rotation gates R(θ) = exp(-iθP/2),
#  the gradient is exact (not finite difference):
#    ∂⟨H⟩/∂θ_k = [⟨H⟩(θ_k + π/2) - ⟨H⟩(θ_k - π/2)] / 2
#
#  We use the exact state vector for optimization (faster, no shot noise).
#  Shot noise is added when reporting the final measurement.
# ─────────────────────────────────────────────

def parameter_shift_gradient(thetas: list, hamiltonian: Hamiltonian) -> list:
    """
    Compute gradient of ⟨H⟩ with respect to all parameters θ_k
    using the parameter shift rule.

    Returns: list of partial derivatives ∂⟨H⟩/∂θ_k
    """
    n = hamiltonian.n_qubits()
    grads = []
    shift = math.pi / 2

    for k in range(len(thetas)):
        # θ + π/2 in direction k
        thetas_plus = list(thetas)
        thetas_plus[k] += shift
        state_plus = prepare_ansatz_state(thetas_plus, n)
        e_plus = exact_energy(state_plus, hamiltonian)

        # θ - π/2 in direction k
        thetas_minus = list(thetas)
        thetas_minus[k] -= shift
        state_minus = prepare_ansatz_state(thetas_minus, n)
        e_minus = exact_energy(state_minus, hamiltonian)

        grads.append((e_plus - e_minus) / 2.0)

    return grads


# ─────────────────────────────────────────────
#  VQE optimizer
# ─────────────────────────────────────────────

def run_vqe(hamiltonian: Hamiltonian, shots: int = 1024,
            iterations: int = 80, step_size: float = 0.3,
            n_params: int = None) -> dict:
    """
    Run the closed VQE loop: quantum energy measurement + classical gradient descent.

    Args:
        hamiltonian:  Aether Hamiltonian object
        shots:        number of shots for final energy estimate
        iterations:   number of optimization steps
        step_size:    gradient descent learning rate
        n_params:     number of variational parameters (default: 2 * n_qubits)

    Returns dict with:
        thetas_opt:   optimized parameters
        energy_opt:   final ground state energy estimate
        energy_exact: exact energy at optimal parameters (noiseless)
        history:      list of (iteration, energy) tuples
    """
    n = hamiltonian.n_qubits()
    if n == 0:
        raise ValueError("Hamiltonian has no qubits")

    # Default: 2 layers of Ry gates (2 * n_qubits parameters)
    if n_params is None:
        n_params = 2 * n

    print_hamiltonian(hamiltonian)

    print(f"  ⟁  Aether VQE — '{hamiltonian.name}'")
    print(f"  {'─'*50}")
    print(f"  Qubits:     {n}")
    print(f"  Parameters: {n_params}  (Ry ansatz, {n_params // n} layers)")
    print(f"  Iterations: {iterations}")
    print(f"  Step size:  {step_size}")
    print(f"  Shots:      {shots}  (final estimate)")
    print()

    # Initialize parameters randomly near 0
    # Near 0 → near |00...0⟩, a known easy starting state
    random.seed(42)
    thetas = [random.gauss(0, 0.1) for _ in range(n_params)]

    history = []
    best_energy = float("inf")
    best_thetas = list(thetas)

    # Optimization loop with adaptive step size (simple momentum)
    velocity = [0.0] * n_params
    momentum = 0.9

    print(f"  {'Iter':>5}  {'Energy (exact)':>18}  {'ΔE':>12}  {'|∇E|':>10}")
    print(f"  {'─'*52}")

    prev_energy = None
    for it in range(iterations):
        # Compute energy at current parameters
        state = prepare_ansatz_state(thetas, n)
        energy = exact_energy(state, hamiltonian)

        history.append((it, energy))

        if energy < best_energy:
            best_energy = energy
            best_thetas = list(thetas)

        # Gradient via parameter shift rule
        grads = parameter_shift_gradient(thetas, hamiltonian)
        grad_norm = math.sqrt(sum(g**2 for g in grads))

        # Progress display (every 5 iterations + first + last)
        delta_str = ""
        if prev_energy is not None:
            delta = energy - prev_energy
            delta_str = f"{delta:+.6f}"
        if it % 5 == 0 or it == iterations - 1:
            print(f"  {it:>5}  {energy:>+18.8f}  {delta_str:>12}  {grad_norm:>10.6f}")

        # Gradient descent with momentum
        for k in range(n_params):
            velocity[k] = momentum * velocity[k] - step_size * grads[k]
            thetas[k] += velocity[k]

        prev_energy = energy

        # Early stopping if converged
        if grad_norm < 1e-6:
            print(f"  {'─'*52}")
            print(f"  Converged at iteration {it} (|∇E| < 1e-6)")
            break

    print(f"  {'─'*52}")
    print()

    # Final measurement with shots (simulates real quantum hardware)
    opt_state = prepare_ansatz_state(best_thetas, n)
    energy_exact = exact_energy(opt_state, hamiltonian)
    energy_shots = measure_energy_shots(opt_state, hamiltonian, shots=shots)

    return {
        "thetas_opt":   best_thetas,
        "energy_opt":   energy_shots,
        "energy_exact": energy_exact,
        "history":      history,
        "n_qubits":     n,
        "n_params":     n_params,
    }


def print_vqe_results(name: str, result: dict):
    """Print the final VQE results in Aether's visual style."""
    print(f"  ⟁  VQE Results — '{name}'")
    print(f"  {'─'*50}")
    print(f"  Ground state energy (shot-based): {result['energy_opt']:+.8f} Hartree")
    print(f"  Ground state energy (exact sim):  {result['energy_exact']:+.8f} Hartree")
    print()
    print(f"  Optimal parameters (θ):")
    for i, theta in enumerate(result["thetas_opt"]):
        print(f"    θ[{i}] = {theta:+.6f}  ({theta/math.pi:+.4f}π)")
    print()

    # Energy convergence plot (ASCII)
    history = result["history"]
    if len(history) > 1:
        energies = [e for _, e in history]
        e_min = min(energies)
        e_max = max(energies)
        e_range = e_max - e_min if e_max != e_min else 1.0
        width = 50
        height = 8

        print(f"  Energy convergence:")
        # Downsample to width points
        step = max(1, len(energies) // width)
        sampled = energies[::step][:width]

        rows = []
        for row in range(height):
            threshold = e_max - (row / (height - 1)) * e_range
            line = ""
            for e in sampled:
                line += "█" if e >= threshold - e_range/(2*height) else " "
            rows.append(f"  {threshold:+.4f} │{line}")

        for r in rows:
            print(r)
        print(f"  {'─'*(width+12)}")
        print(f"  iter 0{' '*(width-8)}iter {len(history)-1}")
        print()

    # Compare to known result if this looks like H2
    known_h2 = -1.9153   # exact FCI eigenvalue for these STO-3G coefficients
    e = result["energy_exact"]
    if abs(e - known_h2) < 0.2:
        error = abs(e - known_h2) * 1000  # in milliHartree
        print(f"  ── H₂ benchmark ──────────────────────────────────")
        print(f"  Aether VQE:    {e:+.6f} Hartree")
        print(f"  Exact (FCI):   {known_h2:+.6f} Hartree")
        print(f"  Error:          {error:.2f} milliHartree")
        chem_acc = 1.6  # milliHartree (chemical accuracy = 1 kcal/mol)
        if error < chem_acc:
            print(f"  ✓  Chemical accuracy achieved (< {chem_acc} mH)")
        else:
            print(f"  ↑  Try more iterations or a deeper ansatz")
        print()
