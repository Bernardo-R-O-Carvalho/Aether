"""
Aether Qiskit Backend
=====================

Compiles Aether quantum circuit AST to Qiskit QuantumCircuit objects
and executes them on IBM simulators or real quantum hardware.

Usage in .aeth:
    infer MyCircuit using qiskit(shots=1024)
    infer MyCircuit using qiskit(shots=1024, backend="ibm_brisbane")

Architecture:
    Aether AST  →  compile_to_qiskit()  →  QuantumCircuit
                →  run_local()          →  AerSimulator (free, fast)
                →  run_ibm()            →  IBM Quantum hardware (real)

Gate mapping (Aether → Qiskit):
    hadamard  →  qc.h()
    pauli_x   →  qc.x()
    pauli_y   →  qc.y()
    pauli_z   →  qc.z()
    cnot      →  qc.cx()
    phase     →  qc.p()
    identity  →  qc.id()

The compilation is nearly 1:1 for basic gates.
The complexity is in IBM authentication and job management.
"""

import math
from collections import defaultdict

# ─────────────────────────────────────────────
#  Qiskit availability check
# ─────────────────────────────────────────────

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import HGate, CXGate, XGate, YGate, ZGate, PhaseGate, IGate
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    try:
        from qiskit.providers.basic_provider import BasicSimulator
        AER_AVAILABLE = False
    except ImportError:
        AER_AVAILABLE = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False


# ─────────────────────────────────────────────
#  Compiler: Aether AST → Qiskit QuantumCircuit
#
#  Walks the circuit body (list of AST nodes from the parser)
#  and emits the equivalent Qiskit operations.
#
#  Design decision: two-pass compilation.
#  Pass 1: count qubits and measurements to size the registers.
#  Pass 2: apply gates and measurements in order.
# ─────────────────────────────────────────────

def compile_to_qiskit(circuit_name, body):
    """
    Compile an Aether quantum circuit AST body to a Qiskit QuantumCircuit.

    Args:
        circuit_name: name of the circuit (for labeling)
        body:         list of AST nodes from the Aether parser

    Returns:
        (qc, qubit_names) where qc is a QuantumCircuit and
        qubit_names maps Aether qubit names to Qiskit register indices.
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "\n  ✗  Qiskit not installed.\n"
            "     Run: pip install qiskit qiskit-ibm-runtime"
        )

    # Pass 1: register qubits and count measurements
    qubit_names = {}   # Aether name -> integer index
    n_qubits = 0
    n_measurements = 0

    for stmt in body:
        if stmt[0] == "qubit":
            qubit_names[stmt[1]] = n_qubits
            n_qubits += 1
        elif stmt[0] == "measure_q":
            n_measurements += 1

    if n_qubits == 0:
        raise RuntimeError(f"\n  ✗  Circuit '{circuit_name}' has no qubits")

    # Build Qiskit circuit with quantum and classical registers
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_measurements, 'c')
    qc = QuantumCircuit(qr, cr, name=circuit_name)

    # Pass 2: apply gates and measurements
    meas_idx = 0

    for stmt in body:
        kind = stmt[0]

        if kind == "qubit":
            # Already handled in pass 1
            continue

        elif kind == "gate":
            _, gate_name, target, kwargs = stmt
            t_idx = qubit_names.get(target)
            if t_idx is None:
                raise RuntimeError(f"\n  ✗  Unknown qubit '{target}'")

            if gate_name == "hadamard":
                qc.h(qr[t_idx])

            elif gate_name == "pauli_x":
                qc.x(qr[t_idx])

            elif gate_name == "pauli_y":
                qc.y(qr[t_idx])

            elif gate_name == "pauli_z":
                qc.z(qr[t_idx])

            elif gate_name == "identity":
                qc.id(qr[t_idx])

            elif gate_name == "cnot":
                control = kwargs.get("control")
                c_idx = qubit_names.get(control)
                if c_idx is None:
                    raise RuntimeError(f"\n  ✗  Unknown control qubit '{control}'")
                qc.cx(qr[c_idx], qr[t_idx])

            elif gate_name == "phase":
                # Phase gate: R(theta) — rotation around Z axis
                theta = kwargs.get("theta", 0.0)
                if isinstance(theta, str):
                    # theta was passed as a variable name — use pi as default
                    theta = math.pi
                qc.p(float(theta), qr[t_idx])

            else:
                raise RuntimeError(
                    f"\n  ✗  Gate '{gate_name}' not supported in Qiskit backend\n"
                    f"     Supported: hadamard, pauli_x, pauli_y, pauli_z, cnot, phase, identity"
                )

        elif kind == "measure_q":
            q_name = stmt[1]
            q_idx = qubit_names.get(q_name)
            if q_idx is None:
                raise RuntimeError(f"\n  ✗  Unknown qubit '{q_name}'")
            qc.measure(qr[q_idx], cr[meas_idx])
            meas_idx += 1

    return qc, qubit_names


# ─────────────────────────────────────────────
#  Local simulation via Aer
#
#  AerSimulator is IBM's high-performance local simulator.
#  Much faster than Aether's pure-Python state vector simulator
#  for large circuits, and supports noise models from real hardware.
#  Free, no authentication required.
# ─────────────────────────────────────────────

def run_local(qc, shots=1024):
    """
    Run a Qiskit circuit on the local Aer simulator.

    Args:
        qc:    compiled QuantumCircuit
        shots: number of measurement repetitions

    Returns:
        dict mapping outcome bitstrings to counts
    """
    if AER_AVAILABLE:
        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
    else:
        # Fallback: Qiskit's basic simulator (slower but always available)
        from qiskit.providers.basic_provider import BasicProvider
        provider = BasicProvider()
        simulator = provider.get_backend('basic_simulator')
        from qiskit import transpile
        compiled = transpile(qc, simulator)
        job = simulator.run(compiled, shots=shots)
        counts = job.result().get_counts()

    return counts


# ─────────────────────────────────────────────
#  IBM Quantum hardware execution
#
#  Submits the circuit to real IBM quantum hardware via the
#  IBM Quantum Runtime Service. Requires:
#    1. IBM Quantum account (free at quantum.ibm.com)
#    2. API token saved locally via QiskitRuntimeService.save_account()
#
#  The circuit is transpiled to the target backend's native gate set
#  before submission — this is handled automatically by Qiskit.
# ─────────────────────────────────────────────

def run_ibm(qc, backend_name="ibm_kingston", shots=1024):
    """
    Run a Qiskit circuit on real IBM quantum hardware.

    Requires IBM Quantum account. First-time setup:
        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(
            token="YOUR_TOKEN",
            channel="ibm_quantum_platform"
        )

    Args:
        qc:           compiled QuantumCircuit
        backend_name: IBM backend name (e.g. "ibm_kingston", "ibm_fez")
        shots:        number of shots (max 4096 on free tier)

    Returns:
        dict mapping outcome bitstrings to counts
    """
    if not IBM_AVAILABLE:
        raise RuntimeError(
            "\n  ✗  IBM Runtime not available.\n"
            "     Run: pip install qiskit-ibm-runtime"
        )

    print(f"  Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")

    print(f"  Loading backend '{backend_name}'...")
    backend = service.backend(backend_name)

    # Transpile to backend's native gate set and topology
    print(f"  Transpiling circuit for {backend_name}...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)

    print(f"  Submitting job ({shots} shots)...")
    print(f"  Circuit depth after transpilation: {isa_circuit.depth()}")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)

    print(f"  Job ID: {job.job_id()}")
    print(f"  Waiting for results (may take minutes in the queue)...")

    result = job.result()
    counts_raw = result[0].data.c.get_counts()

    return counts_raw


# ─────────────────────────────────────────────
#  Entry point
#
#  Called by the interpreter's run_qiskit() method.
#  Routes to local simulator or IBM hardware based on params.
# ─────────────────────────────────────────────

def run_qiskit_circuit(circuit_name, body, shots=1024, backend=None):
    """
    Main entry point for Qiskit backend execution.

    Args:
        circuit_name: Aether circuit name
        body:         circuit AST body
        shots:        number of measurement shots
        backend:      None = local Aer simulator
                      "ibm_*" = IBM Quantum hardware

    Returns:
        None (prints results directly)
    """
    # Compile Aether AST to Qiskit circuit
    qc, qubit_names = compile_to_qiskit(circuit_name, body)

    print(f"\n  ⟁  Aether → Qiskit — '{circuit_name}'")
    print(f"  {'─'*50}")
    print(f"  Qubits    : {qc.num_qubits}")
    print(f"  Gates     : {qc.size()}")
    print(f"  Depth     : {qc.depth()}")
    print(f"  Backend   : {'IBM ' + backend if backend else 'Aer local simulator'}")
    print(f"  Shots     : {shots:,}")
    print()

    # Execute
    if backend and backend.startswith("ibm"):
        counts = run_ibm(qc, backend_name=backend, shots=shots)
    else:
        counts = run_local(qc, shots=shots)

    # Format and print results
    _print_results(counts, qubit_names, shots)

    # Print the Qiskit circuit diagram
    print(f"  Qiskit circuit:")
    try:
        diagram = qc.draw(output='text', fold=60)
        for line in str(diagram).split('\n'):
            print(f"    {line}")
    except Exception:
        pass
    print()


def _print_results(counts, qubit_names, shots):
    """Format and display measurement results."""
    total = sum(counts.values())
    qubit_list = list(qubit_names.keys())

    print(f"  Results ({total:,} shots):")
    print()

    # Per-qubit marginal probabilities
    qubit_ones = defaultdict(int)
    for bitstring, count in counts.items():
        # Qiskit bitstrings are reversed (rightmost = qubit 0)
        bits = bitstring.replace(' ', '')[::-1]
        for i, bit in enumerate(bits):
            if i < len(qubit_list) and bit == '1':
                qubit_ones[qubit_list[i]] += count

    for q_name in qubit_list:
        ones = qubit_ones[q_name]
        p1 = ones / total
        p0 = 1 - p1
        bar_len = 20
        bar1 = "█" * round(p1 * bar_len)
        bar0 = "░" * (bar_len - round(p1 * bar_len))
        print(f"  {q_name}")
        print(f"    |0⟩  {p0:.3f}  {bar0}{bar1}  {p1:.3f}  |1⟩")
        print()

    # Top outcomes
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    print(f"  Top outcomes:")
    for bitstring, count in sorted_counts[:6]:
        bits = bitstring.replace(' ', '')[::-1]
        label_parts = []
        for i, bit in enumerate(bits):
            if i < len(qubit_list):
                label_parts.append(f"{qubit_list[i]}={bit}")
        label = "  ".join(label_parts)
        pct = count / total * 100
        print(f"    {label:30s}  {count:4d}×  ({pct:.1f}%)")
    print()
