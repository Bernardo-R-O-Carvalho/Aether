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
# For Hubbard: keyed by (n_qubits, "hubbard") — handled separately
BENCHMARKS = {
    2: (-1.9153,       "H₂ (STO-3G, 2-qubit parity reduction)"),
    6: (-7.86418329,   "LiH (STO-3G, frozen core, 6-qubit)"),
    8: (-15.56674241,  "BeH₂ (STO-3G, frozen core, 8-qubit)"),
}

# Hubbard benchmarks: identified by Hamiltonian name prefix
HUBBARD_BENCHMARKS = {
    "Hubbard2x1": (-1.00000000, "Hubbard 2×1 chain (t=1, U=4, 2e, 4 qubits)"),
    "Hubbard2x2": (-3.41855072, "Hubbard 2×2 lattice (t=1, U=4, 4e, 8 qubits)"),
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
        "thetas_opt":      best_thetas,
        "energy_opt":      energy_shots_val,
        "energy_exact":    energy_exact,
        "history":         history,
        "n_qubits":        n,
        "n_params":        uccsd_obj.n_params,
        "ansatz":          f"UCCSD ({uccsd_obj.n_params} params)",
        "hamiltonian_name": hamiltonian.name,
    }


# ─────────────────────────────────────────────
#  Hardware-efficient path — gradient descent
# ─────────────────────────────────────────────

def _run_vqe_hardware(hamiltonian, shots, iterations, step_size, n_params):
    from scipy.optimize import minimize
    import numpy as np

    n = hamiltonian.n_qubits()
    n_layers = 6 if n > 4 else 4
    effective_n_params = n_params or (n * n_layers)

    print(f"  ⟁  Aether VQE — '{hamiltonian.name}'")
    print(f"  {'─'*50}")
    print(f"  Qubits:     {n}")
    print(f"  Ansatz:     Hardware-efficient (Ry+CNOT, {n_layers} layers, {effective_n_params} params)")
    print(f"  Optimizer:  BFGS (scipy)")
    print(f"  Shots:      {shots}  (final estimate)")
    print()

    history = []
    call_count = [0]

    def energy_fn(thetas):
        call_count[0] += 1
        state = prepare_ansatz_state(list(thetas), n)
        e = exact_energy(state, hamiltonian)
        history.append((call_count[0], e))
        if call_count[0] % 20 == 1:
            print(f"  {call_count[0]:>5}  {e:>+18.8f}")
        return e

    # Multi-start: try several random initializations, keep best
    random.seed(42)
    best_result = None
    best_e = float('inf')
    n_starts = 3

    print(f"  {'Call':>5}  {'Energy (exact)':>18}")
    print(f"  {'─'*26}")

    for start in range(n_starts):
        x0 = np.array([random.gauss(0, 0.5) for _ in range(effective_n_params)])
        result = minimize(
            energy_fn, x0,
            method='BFGS',
            options={'maxiter': iterations // n_starts, 'gtol': 1e-7}
        )
        if result.fun < best_e:
            best_e = result.fun
            best_result = result

    print(f"  {'─'*26}")
    print(f"  Best energy: {best_e:+.8f}  ({call_count[0]} total evaluations)")
    print()

    best_thetas = list(best_result.x)
    opt_state = prepare_ansatz_state(best_thetas, n)
    energy_exact = exact_energy(opt_state, hamiltonian)
    energy_shots_val = measure_energy_shots(opt_state, hamiltonian, shots=shots)

    return {
        "thetas_opt":      best_thetas,
        "energy_opt":      energy_shots_val,
        "energy_exact":    energy_exact,
        "history":         history,
        "n_qubits":        n,
        "n_params":        effective_n_params,
        "ansatz":          f"Hardware-efficient (Ry+CNOT, {n_layers} layers, BFGS)",
        "hamiltonian_name": hamiltonian.name,
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
    ham_name = result.get("hamiltonian_name", "")

    # Check Hubbard benchmarks first (by name)
    matched = False
    for key, (known_e, label) in HUBBARD_BENCHMARKS.items():
        if ham_name.startswith(key):
            error = abs(e - known_e) * 1000
            print(f"  ── Hubbard benchmark ──────────────────────────────")
            print(f"  System:        {label}")
            print(f"  Aether VQE:    {e:+.8f}")
            print(f"  Exact (ED):    {known_e:+.8f}")
            print(f"  Error:          {error:.4f} milliHartree")
            if error < 1.6:
                print(f"  ✓  Chemical accuracy achieved (< 1.6 mH)")
            else:
                print(f"  ↑  {error/1.6:.1f}× above chemical accuracy")
            print()
            matched = True
            break

    if not matched and n in BENCHMARKS:
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
