"""
Aether VQE Engine
=================

Loop fechado: medição quântica de energia + otimização clássica.

Dois modos:
  UCCSD  — para moléculas conhecidas (LiH, BeH2, H2O)
           usa OpenFermion + scipy BFGS
           exato, sem aproximação de Trotter
  Hardware-efficient — para H2 e Hamiltonianos genéricos
           ansatz Ry+CNOT, parameter shift rule, gradient descent
"""

import math
import random
from hamiltonian import (
    Hamiltonian, prepare_ansatz_state, exact_energy,
    measure_energy_shots, print_hamiltonian
)

# Molecular benchmarks: n_qubits -> (fci_energy, label)
BENCHMARKS = {
    2: (-1.9153,       "H₂ (STO-3G, 2-qubit parity reduction)"),
    6: (-7.86418329,   "LiH (STO-3G, frozen core, 6-qubit)"),
    8: (-15.56674241,  "BeH₂ (STO-3G, frozen core, 8-qubit)"),
}


def run_vqe(hamiltonian, shots: int = 1024,
            iterations: int = 80, step_size: float = 0.3,
            n_params: int = None, ansatz: str = "auto") -> dict:
    """
    Run the closed VQE loop.

    ansatz: "auto"     — UCCSD for LiH/BeH2/H2O, hardware-efficient otherwise
            "uccsd"    — force UCCSD
            "hardware" — force hardware-efficient
    """
    from uccsd import MOLECULE_CONFIGS

    n = hamiltonian.n_qubits()
    if n == 0:
        raise ValueError("Hamiltonian has no qubits")

    # ── Select ansatz ──────────────────────────────────────
    use_uccsd = False
    if ansatz == "uccsd" or (ansatz == "auto" and n in MOLECULE_CONFIGS):
        try:
            from uccsd import make_uccsd_for_hamiltonian
            uccsd_obj = make_uccsd_for_hamiltonian(hamiltonian)
            use_uccsd = True
        except Exception as e:
            print(f"  ⚠  UCCSD unavailable ({e}), using hardware-efficient")
            use_uccsd = False

    print_hamiltonian(hamiltonian)

    if use_uccsd:
        return _run_vqe_uccsd(hamiltonian, uccsd_obj, shots, iterations)
    else:
        return _run_vqe_hardware(hamiltonian, shots, iterations, step_size, n_params)


# ─────────────────────────────────────────────
#  UCCSD path — scipy BFGS + exact matrix exp
# ─────────────────────────────────────────────

def _run_vqe_uccsd(hamiltonian, uccsd_obj, shots, iterations):
    from scipy.optimize import minimize
    import numpy as np

    n = hamiltonian.n_qubits()

    print(f"  ⟁  Aether VQE — '{hamiltonian.name}'")
    print(f"  {'─'*50}")
    print(f"  Qubits:     {n}")
    print(f"  Ansatz:     UCCSD ({uccsd_obj.n_params} params, exact matrix exp)")
    print(f"  Optimizer:  BFGS (scipy)")
    print(f"  Shots:      {shots}  (final estimate)")
    print()

    history = []
    call_count = [0]

    def energy_fn(thetas):
        call_count[0] += 1
        state = uccsd_obj.prepare(list(thetas))
        e = exact_energy(state, hamiltonian)
        history.append((call_count[0], e))
        if call_count[0] % 20 == 1:
            print(f"  {call_count[0]:>5}  {e:>+18.8f}")
        return e

    # Warm start with CCSD amplitudes
    x0 = np.array(uccsd_obj.initial_params())
    print(f"  {'Call':>5}  {'Energy (exact)':>18}")
    print(f"  {'─'*26}")

    result = minimize(
        energy_fn, x0,
        method='BFGS',
        options={'maxiter': iterations, 'gtol': 1e-7}
    )

    print(f"  {'─'*26}")
    print(f"  BFGS converged: {result.success}  ({call_count[0]} evaluations)")
    print()

    best_thetas = list(result.x)
    opt_state = uccsd_obj.prepare(best_thetas)
    energy_exact = exact_energy(opt_state, hamiltonian)
    energy_shots_val = measure_energy_shots(opt_state, hamiltonian, shots=shots)

    return {
        "thetas_opt":   best_thetas,
        "energy_opt":   energy_shots_val,
        "energy_exact": energy_exact,
        "history":      history,
        "n_qubits":     n,
        "n_params":     uccsd_obj.n_params,
        "ansatz":       f"UCCSD ({uccsd_obj.n_params} params)",
    }


# ─────────────────────────────────────────────
#  Hardware-efficient path — gradient descent
# ─────────────────────────────────────────────

def _run_vqe_hardware(hamiltonian, shots, iterations, step_size, n_params):
    n = hamiltonian.n_qubits()
    n_layers = 4 if n > 4 else 2
    effective_n_params = n_params or (n * n_layers)

    print(f"  ⟁  Aether VQE — '{hamiltonian.name}'")
    print(f"  {'─'*50}")
    print(f"  Qubits:     {n}")
    print(f"  Ansatz:     Hardware-efficient (Ry+CNOT, {n_layers} layers)")
    print(f"  Parameters: {effective_n_params}")
    print(f"  Iterations: {iterations}")
    print(f"  Step size:  {step_size}")
    print(f"  Shots:      {shots}  (final estimate)")
    print()

    random.seed(42)
    thetas = [random.gauss(0, 0.1) for _ in range(effective_n_params)]

    history = []
    best_energy = float("inf")
    best_thetas = list(thetas)
    velocity = [0.0] * effective_n_params
    momentum = 0.9

    print(f"  {'Iter':>5}  {'Energy (exact)':>18}  {'ΔE':>12}  {'|∇E|':>10}")
    print(f"  {'─'*52}")

    prev_energy = None
    shift = math.pi / 2

    for it in range(iterations):
        state = prepare_ansatz_state(thetas, n)
        energy = exact_energy(state, hamiltonian)
        history.append((it, energy))

        if energy < best_energy:
            best_energy = energy
            best_thetas = list(thetas)

        grads = []
        for k in range(effective_n_params):
            tp = list(thetas); tp[k] += shift
            tm = list(thetas); tm[k] -= shift
            ep = exact_energy(prepare_ansatz_state(tp, n), hamiltonian)
            em = exact_energy(prepare_ansatz_state(tm, n), hamiltonian)
            grads.append((ep - em) / 2.0)

        grad_norm = math.sqrt(sum(g**2 for g in grads))
        delta_str = f"{energy - prev_energy:+.6f}" if prev_energy is not None else ""

        if it % 5 == 0 or it == iterations - 1:
            print(f"  {it:>5}  {energy:>+18.8f}  {delta_str:>12}  {grad_norm:>10.6f}")

        for k in range(effective_n_params):
            velocity[k] = momentum * velocity[k] - step_size * grads[k]
            thetas[k] += velocity[k]

        prev_energy = energy
        if grad_norm < 1e-6:
            print(f"  Converged at iteration {it}")
            break

    print(f"  {'─'*52}")
    print()

    opt_state = prepare_ansatz_state(best_thetas, n)
    energy_exact = exact_energy(opt_state, hamiltonian)
    energy_shots_val = measure_energy_shots(opt_state, hamiltonian, shots=shots)

    return {
        "thetas_opt":   best_thetas,
        "energy_opt":   energy_shots_val,
        "energy_exact": energy_exact,
        "history":      history,
        "n_qubits":     n,
        "n_params":     effective_n_params,
        "ansatz":       f"Hardware-efficient (Ry+CNOT, {n_layers} layers)",
    }


# ─────────────────────────────────────────────
#  Print results
# ─────────────────────────────────────────────

def print_vqe_results(name: str, result: dict):
    print(f"  ⟁  VQE Results — '{name}'")
    print(f"  {'─'*50}")
    print(f"  Ansatz:                           {result.get('ansatz','unknown')}")
    print(f"  Ground state energy (shot-based): {result['energy_opt']:+.8f} Hartree")
    print(f"  Ground state energy (exact sim):  {result['energy_exact']:+.8f} Hartree")
    print()

    # Convergence plot
    history = result["history"]
    if len(history) > 1:
        energies = [e for _, e in history]
        e_min = min(energies)
        e_max = max(energies)
        e_range = e_max - e_min if e_max != e_min else 1.0
        width, height = 50, 8
        step = max(1, len(energies) // width)
        sampled = energies[::step][:width]
        print(f"  Energy convergence:")
        for row in range(height):
            threshold = e_max - (row / (height - 1)) * e_range
            line = "".join(
                "█" if e >= threshold - e_range/(2*height) else " "
                for e in sampled
            )
            print(f"  {threshold:+.4f} │{line}")
        print(f"  {'─'*(width+12)}")
        print(f"  call 0{' '*(width-8)}call {len(history)-1}")
        print()

    # Molecular benchmark
    n = result["n_qubits"]
    e = result["energy_exact"]
    if n in BENCHMARKS:
        known_e, mol_name = BENCHMARKS[n]
        error = abs(e - known_e) * 1000
        chem_acc = 1.6
        print(f"  ── Molecular benchmark ────────────────────────────")
        print(f"  Molecule:      {mol_name}")
        print(f"  Aether VQE:    {e:+.8f} Hartree")
        print(f"  Exact (FCI):   {known_e:+.8f} Hartree")
        print(f"  Error:          {error:.4f} milliHartree")
        if error < chem_acc:
            print(f"  ✓  Chemical accuracy achieved (< {chem_acc} mH)")
        else:
            print(f"  ↑  {error/chem_acc:.1f}× above chemical accuracy")
        print()
