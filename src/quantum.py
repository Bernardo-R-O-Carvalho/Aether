"""
Aether Quantum — Classical simulation of quantum circuits
==========================================================

What this module does:
  Simulates quantum circuits on a classical computer using state vectors.
  The simulation is mathematically exact — it computes the same probability
  distributions that a real quantum computer would produce.

Why state vectors?
  A quantum system of n qubits is fully described by a vector of 2^n complex
  numbers called amplitudes. Each amplitude α_i corresponds to a basis state
  |i⟩ (a specific pattern of 0s and 1s across all qubits). The probability
  of measuring outcome i is |α_i|² (Born rule).

  We store this entire vector and apply gates as matrix multiplications.
  This is the standard approach for classical simulation of small circuits.

Limitation:
  Memory grows as 2^n. At 20 qubits we need 2^20 = ~1 million complex numbers
  (~16 MB). At 30 qubits that's ~16 GB. This is the fundamental barrier of
  classical quantum simulation — real quantum computers don't have this problem
  because the quantum state is physical, not stored in memory.

Why the math matches real quantum hardware:
  Every gate we implement is a unitary matrix — a matrix that preserves the
  total probability (sum of |α_i|² = 1). This is required by quantum mechanics.
  Measurement collapses the state vector according to the Born rule, exactly
  as it happens on real hardware. The only difference is we simulate shots
  classically with random numbers; real hardware does it physically.
"""

import math
import random
import cmath
from collections import defaultdict


# ─────────────────────────────────────────────
#  State vector simulator
#
#  Represents n qubits as a 2^n-dimensional vector of complex amplitudes.
#  Index i in the vector corresponds to the computational basis state |i⟩,
#  where i is interpreted as a binary string over the n qubits.
#
#  Convention: qubit 0 is the most significant bit.
#  So for 2 qubits: index 0 = |00⟩, 1 = |01⟩, 2 = |10⟩, 3 = |11⟩.
# ─────────────────────────────────────────────

class QuantumRegister:
    """
    Simulates n qubits as a 2^n complex state vector.
    All operations preserve the normalization invariant: Σ|α_i|² = 1.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        # Initialize to |00...0⟩: amplitude 1 for state 0, 0 for all others
        self.state = [complex(0)] * self.dim
        self.state[0] = complex(1)

    def _qubit_index(self, qubit: int) -> int:
        """
        Convert a qubit label (0-indexed from most significant) to
        the bit position in the integer index of the state vector.
        """
        return self.n - 1 - qubit

    def apply_single(self, qubit: int, matrix):
        """
        Apply a 2×2 unitary gate to a single qubit.

        For each pair of amplitudes that differ only in the target qubit's
        bit (|...0...⟩ and |...1...⟩), we apply the 2×2 matrix to the pair.
        This implements the tensor product structure of quantum gates.

        The loop visits each pair exactly once (only processes when bit == 0)
        to avoid double-counting.
        """
        new_state = [complex(0)] * self.dim
        q = self._qubit_index(qubit)
        step = 1 << q   # distance between paired indices

        for i in range(self.dim):
            bit = (i >> q) & 1      # this qubit's value in basis state i
            partner = i ^ step       # the paired state (qubit flipped)
            if bit == 0:
                # Apply the 2×2 matrix to the (|0⟩, |1⟩) pair
                new_state[i]       += matrix[0][0] * self.state[i] + matrix[0][1] * self.state[partner]
                new_state[partner] += matrix[1][0] * self.state[i] + matrix[1][1] * self.state[partner]

        self.state = new_state

    def apply_cnot(self, control: int, target: int):
        """
        Apply a CNOT (controlled-NOT) gate.

        CNOT flips the target qubit if and only if the control qubit is |1⟩.
        In the state vector, this means swapping amplitudes between basis states
        that differ only in the target bit, for all states where the control bit is 1.

        CNOT is the primary entangling gate — it creates correlations between
        qubits that have no classical analog.
        """
        new_state = list(self.state)
        cq = self._qubit_index(control)
        tq = self._qubit_index(target)

        for i in range(self.dim):
            if (i >> cq) & 1:   # control qubit is |1⟩
                flipped = i ^ (1 << tq)   # flip the target bit
                new_state[i], new_state[flipped] = self.state[flipped], self.state[i]

        self.state = new_state

    def measure_qubit(self, qubit: int) -> int:
        """
        Measure a single qubit and collapse the state vector.

        Measurement is irreversible and probabilistic:
        1. Compute the probability of measuring |1⟩: sum of |α_i|² for all
           states where this qubit is 1 (Born rule).
        2. Sample the outcome: 0 or 1.
        3. Collapse: zero out all amplitudes inconsistent with the outcome,
           then renormalize so the remaining amplitudes still sum to 1.

        After measurement, the qubit is in a definite state (no superposition)
        and entanglement with other qubits may be partially destroyed.
        """
        q = self._qubit_index(qubit)

        # Probability of measuring |1⟩ = sum of |α_i|² where qubit q is 1
        prob1 = sum(
            abs(self.state[i]) ** 2
            for i in range(self.dim)
            if (i >> q) & 1
        )

        result = 1 if random.random() < prob1 else 0

        # Collapse: zero out inconsistent amplitudes, renormalize the rest
        norm = 0.0
        for i in range(self.dim):
            if ((i >> q) & 1) != result:
                self.state[i] = complex(0)
            else:
                norm += abs(self.state[i]) ** 2
        norm = math.sqrt(norm)
        if norm > 1e-12:
            self.state = [s / norm for s in self.state]

        return result

    def probabilities(self) -> list:
        """Return the probability of each basis state: |α_i|² for all i."""
        return [abs(a) ** 2 for a in self.state]


# ─────────────────────────────────────────────
#  Quantum Gates
#
#  Each gate is a 2×2 unitary matrix. Unitary means U†U = I (the conjugate
#  transpose times itself equals identity), which preserves normalization.
#
#  These are the standard gates of the quantum circuit model:
#
#  Hadamard (H): puts a qubit in equal superposition.
#    |0⟩ → (|0⟩ + |1⟩)/√2    |1⟩ → (|0⟩ - |1⟩)/√2
#    Applied twice, it cancels itself (H² = I). This is quantum interference.
#
#  Pauli-X: the quantum NOT gate. Flips |0⟩ ↔ |1⟩.
#
#  Pauli-Y: rotation by π around the Y axis of the Bloch sphere.
#
#  Pauli-Z: phase flip. |0⟩ → |0⟩, |1⟩ → -|1⟩.
#    Doesn't change measurement probabilities, but affects interference.
#
#  Identity: does nothing. Useful for padding circuits to the same depth.
#
#  Phase gate: parameterized rotation. R(θ): |1⟩ → e^(iθ)|1⟩.
#    The building block for quantum Fourier transform and Grover's algorithm.
#
#  CNOT (see apply_cnot above): the two-qubit entangling gate.
#    Not representable as a 2×2 matrix — it's a 4×4 on the two-qubit space.
# ─────────────────────────────────────────────

INV_SQRT2 = 1 / math.sqrt(2)

GATES = {
    "hadamard": [
        [complex(INV_SQRT2),  complex(INV_SQRT2)],
        [complex(INV_SQRT2),  complex(-INV_SQRT2)],
    ],
    "pauli_x": [
        [complex(0), complex(1)],
        [complex(1), complex(0)],
    ],
    "pauli_y": [
        [complex(0),    complex(0, -1)],
        [complex(0, 1), complex(0)],
    ],
    "pauli_z": [
        [complex(1),  complex(0)],
        [complex(0), complex(-1)],
    ],
    "identity": [
        [complex(1), complex(0)],
        [complex(0), complex(1)],
    ],
}

def phase_gate(theta: float):
    """
    Parameterized phase gate R(θ).
    Leaves |0⟩ unchanged and multiplies |1⟩ by e^(iθ).
    Used in quantum Fourier transform and phase estimation algorithms.
    """
    return [
        [complex(1), complex(0)],
        [complex(0), cmath.exp(complex(0, theta))],
    ]


# ─────────────────────────────────────────────
#  Quantum Circuit
#
#  Collects qubit declarations and gate operations, then runs them
#  by delegating to QuantumRegister.
#
#  Design decision: two-pass construction.
#  Pass 1: register all qubits to know n (and therefore dim = 2^n).
#  Pass 2: build the operation list.
#  This mirrors how real quantum hardware works — you declare the register
#  size before applying gates.
# ─────────────────────────────────────────────

class QuantumCircuit:
    def __init__(self, name: str):
        self.name = name
        self.qubits = {}        # qubit name -> integer index in the register
        self.operations = []    # ordered list of (op_type, args...)
        self.n_qubits = 0

    def add_qubit(self, name: str):
        """Register a new qubit. Qubits are numbered in declaration order."""
        if name not in self.qubits:
            self.qubits[name] = self.n_qubits
            self.n_qubits += 1

    def add_gate(self, gate: str, target: str, control: str = None, theta: float = None):
        self.operations.append(("gate", gate, target, control, theta))

    def add_measure(self, name: str):
        self.operations.append(("measure", name))

    def run(self) -> dict:
        """
        Execute the circuit once and return measurement outcomes.
        Initializes a fresh QuantumRegister in |00...0⟩ each time.
        """
        if self.n_qubits == 0:
            return {}
        reg = QuantumRegister(self.n_qubits)
        results = {}

        for op in self.operations:
            if op[0] == "gate":
                _, gate, target, control, theta = op
                idx = self.qubits[target]
                if gate == "cnot" and control:
                    ctrl_idx = self.qubits[control]
                    reg.apply_cnot(ctrl_idx, idx)
                elif gate == "phase" and theta is not None:
                    reg.apply_single(idx, phase_gate(theta))
                elif gate in GATES:
                    reg.apply_single(idx, GATES[gate])
            elif op[0] == "measure":
                name = op[1]
                idx = self.qubits[name]
                results[name] = reg.measure_qubit(idx)

        return results

    def run_shots(self, shots: int) -> dict:
        """
        Run the circuit `shots` times and aggregate outcome counts.

        Each shot is an independent execution from |00...0⟩.
        The distribution of outcomes converges to the true quantum probability
        distribution as shots → ∞. This mirrors how real quantum computers
        work: you run the circuit many times and estimate probabilities from
        the frequency of outcomes.

        Returns: dict mapping outcome tuples to counts.
        """
        counts = defaultdict(int)
        for _ in range(shots):
            result = self.run()
            # Convert result dict to a sorted tuple for use as a dict key
            key = tuple(sorted(result.items()))
            counts[key] += 1
        return dict(counts)


# ─────────────────────────────────────────────
#  Entry point
#
#  Called by the interpreter's run_quantum() method.
#  Takes the AST body of a `quantum circuit` block and runs it.
# ─────────────────────────────────────────────

def parse_quantum_model(name: str, body: list, shots: int = 1024) -> None:
    """
    Build and run a quantum circuit from an Aether AST body.

    Two-pass approach:
      Pass 1: register all `qubit` declarations.
      Pass 2: add gates and measures in order.

    Then run for `shots` iterations and print the outcome distribution.
    """
    circuit = QuantumCircuit(name)

    # Pass 1: register all qubits (must know n before allocating state vector)
    for stmt in body:
        if stmt[0] == "qubit":
            circuit.add_qubit(stmt[1])

    # Pass 2: build the operation list in declaration order
    for stmt in body:
        kind = stmt[0]
        if kind == "gate":
            _, gate_name, target, kwargs = stmt
            control = kwargs.get("control")
            theta = kwargs.get("theta")
            circuit.add_gate(gate_name, target, control=control, theta=theta)
        elif kind == "measure_q":
            circuit.add_measure(stmt[1])

    if circuit.n_qubits == 0:
        print(f"\n  ✗  No qubits defined in '{name}'")
        return

    print(f"\n  ⟁  Aether Quantum — '{name}' ({shots} shots, {circuit.n_qubits} qubits)")
    print(f"  {'─'*46}")

    counts = circuit.run_shots(shots)

    # Aggregate per-qubit |0⟩/|1⟩ probabilities across all outcomes
    qubit_ones = defaultdict(int)
    for outcome, count in counts.items():
        for qubit_name, val in outcome:
            qubit_ones[qubit_name] += val * count

    print(f"  Qubits measured: {list(circuit.qubits.keys())}")
    print()

    # Print per-qubit marginal distribution as a visual bar
    for qubit_name in circuit.qubits:
        if qubit_name in dict(list(counts.keys())[0]) if counts else {}:
            ones = qubit_ones[qubit_name]
            prob1 = ones / shots
            prob0 = 1 - prob1
            bar_len = 20
            # Visual bar: ░ for |0⟩ probability, █ for |1⟩ probability
            bar1 = "█" * round(prob1 * bar_len)
            bar0 = "░" * (bar_len - round(prob1 * bar_len))
            print(f"  {qubit_name}")
            print(f"    |0⟩  {prob0:.3f}  {bar0}{bar1}  {prob1:.3f}  |1⟩")
            print()

    # Show the most frequent joint outcomes
    # For entangled systems (Bell pair, GHZ), these reveal the correlations:
    # e.g. a Bell pair always shows q0=0,q1=0 or q0=1,q1=1 — never mixed.
    top = sorted(counts.items(), key=lambda x: -x[1])[:6]
    print(f"  Top outcomes:")
    for outcome, count in top:
        label = " ".join(f"{n}={v}" for n, v in outcome)
        pct = count / shots * 100
        print(f"    {label:30s}  {count:4d}×  ({pct:.1f}%)")
    print()
