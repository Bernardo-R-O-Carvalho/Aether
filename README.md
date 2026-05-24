# ⟁ Aether

## Variables don't have values. They have distributions.

```python
x = 5        # every language you have ever written
```
```
x ~ normal(mean=5, std=1)   # Aether
```

One of these is lying to you.

The real world doesn't produce exact values. Sensors have noise. Diagnoses have uncertainty. Forecasts have error bars. Quantum particles don't have positions — they have probability amplitudes. And yet every programming language ever built pretends that `x = 5` is a complete statement about reality.

Aether doesn't make that assumption.

---

## What problem does Aether solve?

Probabilistic reasoning is one of the most important tools in modern science, medicine, and AI. Bayesian inference, uncertainty quantification, quantum simulation — these are not niche topics. They are how you reason correctly when you don't have complete information.

The tools that exist today make this unreasonably hard:

- **PyMC, Stan, Pyro** — powerful, but they are libraries inside other languages. You write Python that calls PyMC. The probabilistic model is buried inside a general-purpose language that was never designed for it.
- **Qiskit, Cirq** — quantum frameworks that require deep familiarity with linear algebra and circuit notation just to ask a simple question.
- **No existing tool** bridges classical probabilistic inference and quantum simulation in the same syntax.

Aether is a language. Not a library. Not a framework. A language — with its own syntax, its own keywords, its own file extension. When you write `~`, you are not calling an overloaded operator. You are declaring that a variable *is distributed as* a distribution. When you write `observe`, you are not calling a function. You are conditioning the model on evidence. When you write `infer`, the language reasons about the rest.

---

## Why Aether is different

| | PyMC / Stan | Qiskit / Cirq | Aether |
|---|---|---|---|
| Own syntax | ✗ | ✗ | ✓ |
| Zero dependencies | ✗ | ✗ | ✓ |
| Bayesian inference | ✓ | ✗ | ✓ |
| Quantum simulation | ✗ | ✓ | ✓ |
| Both in one file | ✗ | ✗ | ✓ |
| Hierarchical models | ✓ | ✗ | ✓ |
| HMC inference | ✓ | ✗ | ✓ |
| IBM hardware execution | ✗ | ✓ | ✓ |
| Readable without docs | ✗ | ✗ | ✓ |

---

## Runs on real quantum hardware

Aether compiles `.aeth` circuits to Qiskit and executes on IBM Quantum processors.

```
infer BellPair using qiskit(shots=1024, backend="ibm_kingston")
```

First execution on IBM Kingston (156 qubits, us-east):
- **Job ID:** `d88rmaqs46sc73f9rhn0` — verifiable on IBM Quantum
- **Result:** 93.2% entangled correctly — 6.8% quantum noise from real hardware decoherence
- **Circuit:** transpiled from depth 3 → depth 8 for native gate set

```
Top outcomes:
  q0=0  q1=0    511×  (49.9%)   ← entangled ✓
  q0=1  q1=1    443×  (43.3%)   ← entangled ✓
  q0=1  q1=0     64×  ( 6.2%)   ← hardware noise
  q0=0  q1=1      6×  ( 0.6%)   ← hardware noise
```

The deviation from perfect 50/50 is not a bug. It is real quantum decoherence — the physical limitation of today's hardware. A perfect simulator gives exactly 50% each. Real hardware gives 93.2%. That 6.8% is physics.

---

## The language

### Classical: Bayesian inference

```
model CoinBias:
    bias ~ beta(a=1, b=1)
    flip ~ bernoulli(prob=bias)

    observe flip = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0]

infer CoinBias using mcmc(samples=5000, warmup=1000)
```

```
⟁  Aether MCMC — 'CoinBias'
──────────────────────────────────────────────────────
Algorithm  : Metropolis-Hastings
Samples    : 5,000  (warmup discarded: 1,000)
Acceptance : 61.3%  ✓

bias  — posterior distribution

   0.433  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.5%
   0.511  ██░░░░░░░░░░░░░░░░░░░░░░░░░░   4.1%
   0.589  █████░░░░░░░░░░░░░░░░░░░░░░░   9.8%
   0.622  ████████░░░░░░░░░░░░░░░░░░░░  16.2%
   0.667  ████████████████████████████  32.4%
   0.733  █████████████░░░░░░░░░░░░░░░  24.1%
   0.800  ██████░░░░░░░░░░░░░░░░░░░░░░  11.2%
   0.867  █░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.7%

   mean   = 0.6742   std = 0.1204   median = 0.6801
   90% CI = [0.4812, 0.8534]   R-hat = 1.001 ✓
```

### Hierarchical models

```
model Schools:
    global_mean ~ normal(mean=70, std=10)
    global_std  ~ normal(mean=5, std=2)

    for school in ["A", "B", "C", "D", "E"]:
        mean[school] ~ normal(mean=global_mean, std=global_std)

infer Schools using hmc(samples=2000, warmup=500, step_size=0.15, steps=15)
```

Each school has its own mean, drawn from a global distribution. Schools inform each other through the shared prior. This is partial pooling — the core of hierarchical Bayesian modeling.

### Quantum: state vector simulation

```
quantum circuit BellPair:
    qubit q0
    qubit q1

    gate hadamard(target=q0)
    gate cnot(control=q0, target=q1)

    measure q0
    measure q1

infer BellPair using quantum(shots=2048)
```

```
⟁  Aether Quantum — 'BellPair' (2048 shots, 2 qubits)
──────────────────────────────────────────────────────
q0   |0⟩  0.501  ░░░░░░░░░░██████████  0.499  |1⟩
q1   |0⟩  0.499  ░░░░░░░░░░██████████  0.501  |1⟩

Top outcomes:
  q0=0 q1=0        1024×  (50.0%)
  q0=1 q1=1        1024×  (50.0%)
  q0=0 q1=1           0×  (0.0%)
  q0=1 q1=0           0×  (0.0%)
```

### Quantum algorithms

Aether expresses canonical quantum algorithms in a fraction of the lines required by Qiskit:

```
# Deutsch-Jozsa — proves quantum advantage in one query
quantum circuit DeutschJozsa_Balanced:
    qubit q0
    qubit q1
    gate pauli_x(target=q1)
    gate hadamard(target=q0)
    gate hadamard(target=q1)
    gate cnot(control=q0, target=q1)
    gate hadamard(target=q0)
    measure q0

infer DeutschJozsa_Balanced using quantum(shots=1024)
# Result: q0 = 1 (100%) — balanced function detected in one query
```

```
# Grover's search — finds marked item in sqrt(N) steps
quantum circuit Grover2:
    qubit q0
    qubit q1
    gate hadamard(target=q0)
    gate hadamard(target=q1)
    # oracle + diffusion
    ...
    measure q0
    measure q1

infer Grover2 using quantum(shots=1024)
# Result: q0=1, q1=1 (100%) — marked item found
```

### Classical and quantum. Same file.

```
# The same question. Two paradigms. One file.

model CoinBias:
    bias ~ beta(a=1, b=1)
    flip ~ bernoulli(prob=bias)
    observe flip = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0]

infer CoinBias using mcmc(samples=5000, warmup=1000)

quantum circuit QuantumCoin:
    qubit q
    gate hadamard(target=q)
    measure q

infer QuantumCoin using quantum(shots=2048)

# The classical model learns from evidence.
# The quantum model samples from pure uncertainty.
# Same syntax. Same file. Different physics.
```

No other language does this.

---

## Architecture

Aether is not a wrapper. It is built from scratch in pure Python with zero dependencies.

**Tokenizer** — hand-written lexer. Understands `~` as a first-class token meaning *"is distributed as"*, not an operator borrowed from another language.

**Parser** — recursive descent parser producing an Abstract Syntax Tree. Hand-written over a parser generator to give full control over error messages. When something goes wrong, Aether tells you what and where — not Python.

**Type system** — lightweight runtime type checking. `beta(...)` produces a value in `[0,1]`. `bernoulli(...)` produces `{0, 1}`. Violations are caught with useful messages, not silent wrong results.

**Classical runtime** — evaluates probabilistic programs. Supports six built-in distributions: `normal`, `beta`, `bernoulli`, `uniform`, `poisson`, `categorical`. Three inference engines: rejection sampling, Metropolis-Hastings MCMC, and Hamiltonian Monte Carlo.

**MCMC engine** — Metropolis-Hastings with Gaussian random walk proposals. Supports multiple observations (`observe x = [1, 0, 1, 1]`), R-hat convergence diagnostics, ASCII posterior histograms, and trace plots for chain health visualization.

**HMC engine** — Hamiltonian Monte Carlo with leapfrog integration. Uses gradient information to explore the posterior much more efficiently than MH — especially for hierarchical models with many correlated variables. Supports automatic differentiation via JAX when available, with a pure-Python numerical gradient fallback that requires zero dependencies.

**Hierarchical models** — indexed variables (`mean[group] ~ normal(...)`) and `for` loops inside model bodies. Enables partial pooling across groups — the most powerful pattern in Bayesian statistics.

**Quantum simulator** — full complex state vector simulation. Represents n qubits as a 2ⁿ-dimensional vector of complex amplitudes. Implements Hadamard, CNOT, Pauli X/Y/Z, and phase gates using correct unitary matrix mathematics. Collapses the wave function on measurement via the Born rule.

**Qiskit backend** — compiles Aether circuit AST to Qiskit `QuantumCircuit` objects. Routes to Aer local simulator or IBM Quantum hardware based on the `backend` parameter. Handles transpilation to native gate sets automatically.

**Variable scoping** — models run in isolated scopes with a parent chain. Multiple models in one file don't share variables.

**For loops** — iterate over lists within models. Foundation for hierarchical models.

**Imports** — `import "other_model.aeth"` loads and executes another file in the global scope. Circular imports are prevented.

**Error messages** — all errors are caught and presented as Aether errors, not Python tracebacks. Line numbers, context, and hints are included.

**VS Code extension** — syntax highlighting for `.aeth` files. Keywords, distributions, quantum gates, and operators each get distinct colors.

**Web playground** — the entire interpreter ported to JavaScript. Runs locally in the browser. No server. No install. Write `.aeth`, press run, see results.

---

## Built-in distributions

| Distribution | Parameters | Use for |
|---|---|---|
| `normal(mean, std)` | μ, σ | Continuous measurements, sensor noise |
| `beta(a, b)` | α, β | Probabilities, rates, beliefs |
| `bernoulli(prob)` | p | Binary outcomes, coin flips |
| `uniform(low, high)` | a, b | Bounded uncertainty, ignorance priors |
| `poisson(lam)` | λ | Event counts, arrivals |
| `categorical(probs)` | [p₁…pₙ] | Discrete choices, classifications |

## Quantum gates

| Gate | Keyword | Effect |
|---|---|---|
| Hadamard | `hadamard` | Equal superposition: \|0⟩ → (\|0⟩+\|1⟩)/√2 |
| CNOT | `cnot` | Entangles two qubits |
| Pauli X | `pauli_x` | Quantum NOT — flips \|0⟩ ↔ \|1⟩ |
| Pauli Y | `pauli_y` | Rotation around Y axis |
| Pauli Z | `pauli_z` | Phase flip |
| Phase | `phase` | Rotation by angle θ |

---

## Getting started

```bash
git clone https://github.com/Bernardo-R-O-Carvalho/Aether
cd Aether
python src/interpreter.py examples/bell_pair.aeth
python src/interpreter.py examples/hierarchical_schools.aeth
python src/interpreter.py examples/deutsch_jozsa.aeth
python src/interpreter.py examples/vqe.aeth
```

No pip install. No virtual environment. No dependencies. Pure Python 3.8+.

## Try it online

No install needed — runs entirely in the browser:

**[⟁ Open Aether Playground](https://bernardo-r-o-carvalho.github.io/Aether/playground.html)**

Write `.aeth` directly in the browser. Classical inference, HMC, hierarchical models, and quantum circuits — all running locally with zero dependencies.

---

## Inference methods

```
# Rejection sampling — simple models, few variables
infer MyModel using montecarlo(samples=5000)

# Metropolis-Hastings MCMC — complex models, multiple observations
infer MyModel using mcmc(samples=5000, warmup=1000, step_size=0.3)

# Hamiltonian Monte Carlo — hierarchical models, high-dimensional posteriors
infer MyModel using hmc(samples=2000, warmup=500, step_size=0.1, steps=10)

# Aether state vector simulator — fast, exact, no dependencies
infer MyCircuit using quantum(shots=2048)

# Qiskit Aer local simulator — high performance, noise models
infer MyCircuit using qiskit(shots=1024)

# IBM Quantum real hardware
infer MyCircuit using qiskit(shots=1024, backend="ibm_kingston")
```

**When to use each:**
- `montecarlo` — fewer than 3 variables, loose observations.
- `mcmc` — any model with tight observations or many variables.
- `hmc` — hierarchical models, correlated parameters. Install JAX for maximum performance.
- `quantum` — fast local simulation, no setup required.
- `qiskit` — high-performance local simulation or real IBM hardware.

### HMC and JAX

HMC uses gradients of the log-probability to navigate the posterior intelligently. Aether supports two backends:

- **JAX** (recommended): exact automatic differentiation. Install with `pip install jax jaxlib`.
- **Numerical** (default): pure Python finite differences. Zero dependencies.

### IBM Quantum hardware

To run on real IBM quantum computers:

```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer
```

Save your IBM Quantum account (one time):
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    token="YOUR_TOKEN",
    channel="ibm_quantum_platform"
)
```

Get your token at [quantum.ibm.com](https://quantum.ibm.com). Free tier includes 10 minutes of runtime per month.

---

## Reading MCMC/HMC output

```
Acceptance : 61.3%  ✓       # MH healthy range: 20-70%
                             # HMC healthy range: 60-90%

R-hat = 1.001 ✓             # < 1.1: chain converged
                             # > 1.1: run longer or debug model

90% CI = [0.48, 0.85]       # true value lies here with 90% probability
                             # (Bayesian credible interval, not frequentist)
```

---

## Honest limitations

The quantum simulator uses full state vectors, which require 2ⁿ memory. Accurate up to ~20 qubits on a standard machine. It is a classical simulation, not quantum hardware — but the mathematics is identical to what runs on real devices.

HMC without JAX uses numerical gradients (finite differences), which require 2N model evaluations per gradient computation. For models with many variables, `pip install jax jaxlib` is strongly recommended.

IBM hardware results include real quantum noise — decoherence, gate errors, measurement errors. This is not a limitation of Aether. It is the current state of quantum hardware.

---

## Roadmap

- [x] Core distributions — normal, beta, bernoulli, uniform, poisson, categorical
- [x] `observe` for Bayesian conditioning
- [x] Multiple observations — `observe x = [1, 0, 1, 1]`
- [x] Rejection sampling inference
- [x] Metropolis-Hastings MCMC
- [x] R-hat convergence diagnostic
- [x] ASCII posterior histograms and trace plots
- [x] Complex state vector quantum simulator
- [x] Hadamard, CNOT, Pauli X/Y/Z, phase gates
- [x] For loops and hierarchical models with indexed variables
- [x] Hamiltonian Monte Carlo (HMC) with JAX autodiff backend
- [x] Variable scoping and imports
- [x] Type system with useful error messages
- [x] VS Code syntax highlighting
- [x] Web playground — runs in the browser
- [x] Qiskit backend — local Aer simulator
- [x] IBM Quantum hardware execution — verified on ibm_kingston
- [x] Quantum algorithms — Deutsch-Jozsa, Grover, VQE
- [ ] Hamiltonians as first-class types
- [ ] `measure_energy` primitive for VQE
- [ ] Full VQE loop — closed classical-quantum optimization
- [ ] Schrödinger equation for H₂ on real hardware
- [ ] Variational inference
- [ ] Plot output — matplotlib histograms and trace plots
- [ ] Paper — arXiv

---

## Philosophy

Most programming languages were designed in a world that believed in certainty. You assign a value. You check a condition. You return a result. The assumption underneath all of it is that the universe, if you look carefully enough, resolves to exact answers.

It doesn't.

Physics has known this since 1927. Statistics has known it since Bayes. And yet our programming languages still pretend that `x = 5` is a complete statement about the world.

Aether is a small argument that it isn't.

---

## Contributing

Aether is early. The grammar is young, the runtime is honest about its limitations, and there is enormous room to grow. If you write a `.aeth` program, open an issue, suggest syntax, or implement a new inference algorithm — you are shaping the language.

Every contribution is also a statement: that uncertainty deserves to be in the language, not the library.

---

*Built with curiosity. Powered by probability. Ready for the quantum age.*

⟁
