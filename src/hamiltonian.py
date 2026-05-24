"""
Aether Hamiltonian — Native Hamiltonian type and energy measurement
====================================================================

What is a Hamiltonian?
  In quantum mechanics, the Hamiltonian H is the operator that governs the
  energy of a quantum system. Finding the minimum eigenvalue (ground state
  energy) of H is the central problem of quantum chemistry.

  For molecules, H can be expressed as a sum of Pauli operator products:
    H = Σ c_k * P_k1 ⊗ P_k2 ⊗ ... ⊗ P_kn

  where c_k are real coefficients and P_ki ∈ {I, X, Y, Z} are single-qubit
  Pauli operators. This is the Pauli decomposition — every Hermitian operator
  on n qubits can be written this way.

Why Pauli strings?
  Quantum computers can only measure in the Z (computational) basis.
  To measure ⟨ψ|H|ψ⟩, we decompose H into Pauli strings and measure each
  term separately by rotating to the appropriate basis before measuring.

  For a term c * X₀Z₁:
    - Apply H to qubit 0 (rotates X basis to Z basis)
    - Leave qubit 1 alone (already in Z basis)
    - Measure both qubits
    - Result contribution: c * (-1)^(q0 ⊕ q1) (parity of outcomes)

H₂ Hamiltonian (STO-3G basis, equilibrium geometry):
  The Hamiltonian of molecular hydrogen, after Jordan-Wigner transformation
  to map fermionic operators to qubits, is:
    H = -1.0523 I
      +  0.3979 Z₀
      -  0.3979 Z₁
      -  0.0112 Z₀Z₁
      +  0.1809 X₀X₁
      +  0.1809 Y₀Y₁

  Ground state energy ≈ -1.137 Hartree (exact: -1.1372 Hartree)
  This is the result that would appear in a chemistry paper.

Usage in .aeth:
  hamiltonian H:
      term -1.0523  identity
      term  0.3979  pauli_z(q0)
      term -0.3979  pauli_z(q1)
      term -0.0112  pauli_z(q0) pauli_z(q1)
      term  0.1809  pauli_x(q0) pauli_x(q1)
      term  0.1809  pauli_y(q0) pauli_y(q1)

  measure_energy H   →  returns ⟨ψ|H|ψ⟩

  infer Ansatz using vqe(hamiltonian=H, shots=1024, iterations=50)
"""

import math
import cmath
import random
from collections import defaultdict


# ─────────────────────────────────────────────
#  Pauli matrices (2×2 complex)
# ─────────────────────────────────────────────

PAULI_I = [[complex(1), complex(0)],
           [complex(0), complex(1)]]

PAULI_X = [[complex(0), complex(1)],
           [complex(1), complex(0)]]

PAULI_Y = [[complex(0),    complex(0, -1)],
           [complex(0, 1), complex(0)]]

PAULI_Z = [[complex(1),  complex(0)],
           [complex(0), complex(-1)]]

PAULIS = {
    "identity": PAULI_I,
    "pauli_x":  PAULI_X,
    "pauli_y":  PAULI_Y,
    "pauli_z":  PAULI_Z,
}


# ─────────────────────────────────────────────
#  Hamiltonian — native type
#
#  A Hamiltonian is a list of (coefficient, Pauli_string) terms.
#  A Pauli string is a dict mapping qubit_index -> Pauli_name.
#  Example: 0.1809 * X₀X₁  →  (0.1809, {0: "pauli_x", 1: "pauli_x"})
# ─────────────────────────────────────────────

class Hamiltonian:
    """
    Represents a qubit Hamiltonian as a sum of weighted Pauli strings.

    H = Σ_k coeff_k * ⊗_i P_ki

    where P_ki is a Pauli operator (I, X, Y, Z) on qubit i for term k.
    """
    def __init__(self, name: str):
        self.name = name
        self.terms = []   # list of (coeff: float, ops: dict[int, str])

    def add_term(self, coeff: float, ops: dict):
        """
        Add one term to the Hamiltonian.

        Args:
            coeff: real coefficient for this term
            ops: dict mapping qubit_index (int) -> pauli_name (str)
                 e.g. {0: "pauli_x", 1: "pauli_x"} for X₀X₁
                 Empty dict = identity term (contributes coeff * I)
        """
        self.terms.append((coeff, ops))

    def n_qubits(self) -> int:
        """Number of qubits needed for this Hamiltonian."""
        if not self.terms:
            return 0
        max_q = max(
            (max(ops.keys()) if ops else -1)
            for _, ops in self.terms
        )
        return max_q + 1

    def __repr__(self):
        lines = [f"Hamiltonian({self.name!r}):"]
        for coeff, ops in self.terms:
            if not ops:
                lines.append(f"  {coeff:+.6f} * I")
            else:
                pauli_str = " ⊗ ".join(
                    f"{p[6:].upper() if p != 'identity' else 'I'}[q{q}]"
                    for q, p in sorted(ops.items())
                )
                lines.append(f"  {coeff:+.6f} * {pauli_str}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Expectation value ⟨ψ|H|ψ⟩
#
#  Given a state vector ψ and a Hamiltonian H, compute the energy.
#  This is the inner product ⟨ψ|H|ψ⟩.
#
#  Method: for each term c * P₁ ⊗ P₂ ⊗ ... ⊗ Pₙ:
#    1. Compute |φ⟩ = (P₁ ⊗ P₂ ⊗ ... ⊗ Pₙ)|ψ⟩
#       using tensor product of Pauli matrices
#    2. Add c * ⟨ψ|φ⟩ to the total energy
#
#  This is the exact mathematical expectation — no sampling noise.
#  Used for noiseless simulation to verify correctness.
# ─────────────────────────────────────────────

def _apply_pauli_string(state: list, ops: dict, n_qubits: int) -> list:
    """
    Apply a Pauli string operator to a state vector.

    ops: dict[qubit_index -> pauli_name]
    Returns the new state vector ⟨pauli_string⟩ applied to |state⟩.
    """
    new_state = list(state)
    dim = len(state)

    for qubit, pauli_name in ops.items():
        if pauli_name == "identity":
            continue  # identity does nothing

        mat = PAULIS[pauli_name]
        q_bit = n_qubits - 1 - qubit   # bit position in integer index
        step = 1 << q_bit
        result = [complex(0)] * dim

        for i in range(dim):
            bit = (i >> q_bit) & 1
            partner = i ^ step
            if bit == 0:
                result[i]       += mat[0][0] * new_state[i] + mat[0][1] * new_state[partner]
                result[partner] += mat[1][0] * new_state[i] + mat[1][1] * new_state[partner]

        new_state = result

    return new_state


def exact_energy(state: list, hamiltonian: Hamiltonian) -> float:
    """
    Compute ⟨ψ|H|ψ⟩ exactly from the state vector.

    This is the mathematically exact expectation value — no sampling.
    Used internally for VQE optimization (gradient computation).
    """
    n = hamiltonian.n_qubits()
    total = 0.0

    for coeff, ops in hamiltonian.terms:
        # Compute H|ψ⟩ for this term
        applied = _apply_pauli_string(state, ops, n)
        # Compute ⟨ψ|applied⟩ = Σ conj(ψ_i) * applied_i
        overlap = sum(
            state[i].conjugate() * applied[i]
            for i in range(len(state))
        )
        total += coeff * overlap.real   # Hermitian → imaginary part is 0

    return total


# ─────────────────────────────────────────────
#  Shot-based energy estimation
#
#  In real quantum hardware, you cannot access the state vector.
#  You can only measure qubits in the computational (Z) basis.
#  To measure ⟨ψ|P|ψ⟩ for a Pauli string P, you:
#    1. Rotate to the appropriate basis (H gate for X, S†H for Y, nothing for Z)
#    2. Measure all qubits
#    3. Compute the parity: (-1)^(sum of measured bits for non-identity qubits)
#    4. Average over many shots to estimate ⟨P⟩
#
#  This is how real quantum computers measure Hamiltonians.
# ─────────────────────────────────────────────

def _rotation_to_z_basis(pauli_name: str):
    """
    Return the rotation gate matrix that transforms a Pauli eigenstate
    into the Z basis for measurement.

    X basis: apply H (Hadamard) to rotate to Z
    Y basis: apply S†H to rotate to Z  (S† = [[1,0],[0,-i]])
    Z basis: no rotation needed
    Identity: no rotation needed
    """
    INV_SQRT2 = 1 / math.sqrt(2)
    if pauli_name == "pauli_x":
        # H gate: maps |+⟩→|0⟩, |−⟩→|1⟩
        return [
            [complex(INV_SQRT2),  complex(INV_SQRT2)],
            [complex(INV_SQRT2), complex(-INV_SQRT2)],
        ]
    elif pauli_name == "pauli_y":
        # S†H gate: maps |+i⟩→|0⟩, |−i⟩→|1⟩
        return [
            [complex(INV_SQRT2),        complex(INV_SQRT2)],
            [complex(0, INV_SQRT2), complex(0, -INV_SQRT2)],
        ]
    else:
        return None   # no rotation needed for Z or I


def measure_energy_shots(state: list, hamiltonian: Hamiltonian, shots: int = 1024) -> float:
    """
    Estimate ⟨ψ|H|ψ⟩ using shot-based measurements.

    This simulates how a real quantum computer would measure the energy:
    - For each Pauli string term, rotate to Z basis and measure
    - Use parity of outcomes to estimate the term's contribution
    - Average over shots

    Returns a noisy estimate of the true energy.
    The estimate converges to exact_energy() as shots → ∞.
    """
    n = hamiltonian.n_qubits()
    dim = len(state)
    total_energy = 0.0

    for coeff, ops in hamiltonian.terms:
        if not ops:
            # Pure identity term: contributes coeff directly
            total_energy += coeff
            continue

        # For each shot, measure parity of relevant qubits after basis rotation
        term_sum = 0.0
        for _ in range(shots):
            # Work on a copy of the state for each shot
            shot_state = list(state)

            # Apply basis rotation gates for each non-identity Pauli
            for qubit, pauli_name in ops.items():
                rot = _rotation_to_z_basis(pauli_name)
                if rot is not None:
                    # Apply single-qubit rotation
                    q_bit = n - 1 - qubit
                    step = 1 << q_bit
                    new_state = [complex(0)] * dim
                    for i in range(dim):
                        bit = (i >> q_bit) & 1
                        partner = i ^ step
                        if bit == 0:
                            new_state[i]       += rot[0][0]*shot_state[i] + rot[0][1]*shot_state[partner]
                            new_state[partner] += rot[1][0]*shot_state[i] + rot[1][1]*shot_state[partner]
                    shot_state = new_state

            # Now measure qubits in Z basis (computational basis)
            # Compute measurement probabilities
            probs = [abs(a)**2 for a in shot_state]

            # Sample a basis state according to probabilities
            r = random.random()
            cumulative = 0.0
            outcome_idx = 0
            for idx, p in enumerate(probs):
                cumulative += p
                if r < cumulative:
                    outcome_idx = idx
                    break

            # Compute parity of measured bits for qubits in this Pauli string
            parity = 0
            for qubit, pauli_name in ops.items():
                if pauli_name != "identity":
                    q_bit = n - 1 - qubit
                    bit_val = (outcome_idx >> q_bit) & 1
                    parity ^= bit_val   # XOR accumulates parity

            # Contribution: +1 if even parity, -1 if odd
            term_sum += (-1) ** parity

        total_energy += coeff * (term_sum / shots)

    return total_energy


# ─────────────────────────────────────────────
#  Ansatz circuit parametrization
#
#  For VQE, we need a parametrized quantum circuit (ansatz).
#  We support a hardware-efficient ansatz that works for 2-qubit systems:
#
#  Hardware-efficient ansatz for n=2:
#    Ry(θ₀)|0⟩  ─── ●  ───  Ry(θ₂)|⟩
#    Ry(θ₁)|0⟩  ─── X  ───   Ry(θ₃)|⟩
#
#  This creates sufficient entanglement to reach the H₂ ground state.
# ─────────────────────────────────────────────

def ry_gate(theta: float):
    """Rotation around Y axis by angle theta."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return [
        [complex(c), complex(-s)],
        [complex(s), complex(c)],
    ]


def prepare_ansatz_state(thetas: list, n_qubits: int) -> list:
    """
    Prepare a hardware-efficient ansatz state vector given parameters theta.

    For 2 qubits: Ry(θ₀) ⊗ Ry(θ₁), then CNOT, then Ry(θ₂) ⊗ Ry(θ₃).
    For 1 qubit:  Ry(θ₀).
    For n qubits: generalizes the above pattern.

    This is a pure state preparation (no measurement) — returns the full
    state vector so we can compute ⟨ψ|H|ψ⟩ exactly.
    """
    dim = 2 ** n_qubits
    state = [complex(0)] * dim
    state[0] = complex(1)   # |00...0⟩

    # Layer 1: Ry rotations on each qubit
    for q in range(min(n_qubits, len(thetas))):
        theta = thetas[q]
        mat = ry_gate(theta)
        q_bit = n_qubits - 1 - q
        step = 1 << q_bit
        new_state = [complex(0)] * dim
        for i in range(dim):
            bit = (i >> q_bit) & 1
            partner = i ^ step
            if bit == 0:
                new_state[i]       += mat[0][0]*state[i] + mat[0][1]*state[partner]
                new_state[partner] += mat[1][0]*state[i] + mat[1][1]*state[partner]
        state = new_state

    # Entangling layer: CNOT between adjacent qubits
    if n_qubits >= 2:
        for ctrl in range(n_qubits - 1):
            tgt = ctrl + 1
            ctrl_bit = n_qubits - 1 - ctrl
            tgt_bit  = n_qubits - 1 - tgt
            new_state = list(state)
            for i in range(dim):
                if (i >> ctrl_bit) & 1:   # control is |1⟩
                    flipped = i ^ (1 << tgt_bit)
                    new_state[i], new_state[flipped] = state[flipped], state[i]
            state = new_state

    # Layer 2: second round of Ry rotations (if enough parameters provided)
    for q in range(min(n_qubits, len(thetas) - n_qubits)):
        theta = thetas[n_qubits + q]
        mat = ry_gate(theta)
        q_bit = n_qubits - 1 - q
        step = 1 << q_bit
        new_state = [complex(0)] * dim
        for i in range(dim):
            bit = (i >> q_bit) & 1
            partner = i ^ step
            if bit == 0:
                new_state[i]       += mat[0][0]*state[i] + mat[0][1]*state[partner]
                new_state[partner] += mat[1][0]*state[i] + mat[1][1]*state[partner]
        state = new_state

    return state


# ─────────────────────────────────────────────
#  Print Hamiltonian (used by measure_energy)
# ─────────────────────────────────────────────

def print_hamiltonian(h: Hamiltonian):
    """Pretty-print a Hamiltonian in Aether's visual style."""
    n = h.n_qubits()
    print(f"\n  ⟁  Aether — Hamiltonian '{h.name}' ({len(h.terms)} terms, {n} qubits)")
    print(f"  {'─'*50}")
    for coeff, ops in h.terms:
        if not ops:
            label = "I"
        else:
            parts = []
            for q in sorted(ops):
                p = ops[q]
                short = {"identity": "I", "pauli_x": "X", "pauli_y": "Y", "pauli_z": "Z"}
                parts.append(f"{short.get(p,'?')}({q})")
            label = " ⊗ ".join(parts)
        print(f"  {coeff:+.6f}  {label}")
    print()
