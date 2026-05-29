"""
Aether (.aeth) — Probabilistic + Quantum Programming Language
Interpreter v0.5

Architecture overview:
  Source code (.aeth)
    → Tokenizer   : raw text into a flat list of typed tokens
    → Parser      : tokens into an Abstract Syntax Tree (AST)
    → Runtime     : walks the AST and executes each node

  Four execution modes coexist in the same .aeth file:
    - Classical probabilistic  (model / infer using montecarlo or mcmc)
    - Quantum circuit          (quantum circuit / infer using quantum)
    - Hybrid VQE               (hamiltonian / infer using vqe)
    - Graph optimization       (graph / infer using qaoa)

New in v0.5:
  1. Graph type                graph G: nodes N  edge i j ...
  2. QAOA inference engine     infer G using qaoa(layers=2, shots=1024)
  3. Native MaxCut solver      automatic exact comparison + approximation ratio
"""

import re, math, random, sys, os
from collections import defaultdict


# ─────────────────────────────────────────────
#  Type system
#
#  Every distribution produces values of a known type.
#  We track this at sampling time so we can catch mistakes early —
#  e.g. passing a Poisson count (which can be > 1) as a Bernoulli
#  probability (which must be in [0, 1]) is a silent bug without types.
#
#  This is intentionally lightweight: not a full static type system,
#  just runtime assertions that fire with useful messages.
# ─────────────────────────────────────────────

class AetherType:
    CONTINUOUS  = "continuous"
    PROBABILITY = "probability"   # constrained to [0, 1]
    BINARY      = "binary"        # constrained to {0, 1}
    COUNT       = "count"         # non-negative integer
    CATEGORICAL = "categorical"   # non-negative integer index
    ANY         = "any"           # unconstrained — for derived variables

# Maps each distribution name to the type of values it produces.
# Used by check_type() after each ~ sampling operation.
DIST_TYPES = {
    "normal":      AetherType.CONTINUOUS,
    "beta":        AetherType.PROBABILITY,
    "bernoulli":   AetherType.BINARY,
    "uniform":     AetherType.CONTINUOUS,
    "poisson":     AetherType.COUNT,
    "categorical": AetherType.CATEGORICAL,
}

def check_type(var_name, value, atype, line=None):
    """
    Assert that a sampled value respects its distribution's constraints.
    Raises AetherError with a helpful message if the constraint is violated.
    In practice this fires when distribution parameters are themselves
    random variables that drift out of valid range during MCMC.
    """
    loc = f" (line {line})" if line else ""
    if atype == AetherType.PROBABILITY:
        if not (0.0 <= value <= 1.0):
            raise AetherError(f"Type error{loc}: '{var_name}' must be in [0, 1], got {value:.4f}")
    elif atype == AetherType.BINARY:
        if value not in (0, 1):
            raise AetherError(f"Type error{loc}: '{var_name}' must be 0 or 1, got {value}")
    elif atype == AetherType.COUNT:
        if value < 0:
            raise AetherError(f"Type error{loc}: '{var_name}' must be >= 0, got {value}")


# ─────────────────────────────────────────────
#  Error handling
#
#  All errors raised by Aether are AetherError instances.
#  The run_file() entry point catches them and prints a clean,
#  user-facing message — no Python tracebacks visible to the user.
# ─────────────────────────────────────────────

class AetherError(Exception):
    pass

def aether_error(msg, line=None, src_lines=None):
    """Format a user-facing error with source context."""
    parts = [f"\n  ✗  Aether error: {msg}"]
    if line and src_lines and 0 < line <= len(src_lines):
        parts.append(f"     line {line}: {src_lines[line-1].strip()}")
    raise AetherError("\n".join(parts))


# ─────────────────────────────────────────────
#  Distributions
#
#  Each distribution is a plain function that takes keyword arguments
#  and returns a single sampled value. We use Python's standard library
#  (random module) throughout — no external dependencies.
#
#  Design decision: distributions validate their parameters eagerly
#  and raise AetherError (not Python's ValueError) so error messages
#  stay within Aether's error format and are user-readable.
# ─────────────────────────────────────────────

def _normal(mean=0.0, std=1.0, **_):
    # Box-Muller transform via Python's random.gauss
    return random.gauss(mean, std)

def _bernoulli(prob=0.5, **_):
    # Binary outcome: 1 with probability prob, 0 otherwise.
    # prob must be a valid probability — catch bad values from MCMC proposals.
    if not 0 <= prob <= 1:
        raise AetherError(f"Type error: bernoulli(prob=...) requires prob in [0, 1], got {prob:.4f}\n     Hint: prob is a probability — it must be between 0 and 1")
    return 1 if random.random() < prob else 0

def _beta(a=1.0, b=1.0, **_):
    # Beta distribution: models probabilities and rates, always in (0, 1).
    # With a=b=1 it's uniform over (0,1) — a non-informative prior.
    # Both shape parameters must be strictly positive.
    if a <= 0 or b <= 0:
        raise AetherError(f"Type error: beta(a=..., b=...) requires a > 0 and b > 0, got a={a}, b={b}\n     Hint: both shape parameters must be positive")
    return random.betavariate(a, b)

def _uniform(low=0.0, high=1.0, **_):
    # Uniform distribution: equal probability for all values in [low, high].
    # Useful as an uninformative prior when you only know the bounds.
    if low >= high:
        raise AetherError(f"Type error: uniform(low=..., high=...) requires low < high, got [{low}, {high}]\n     Hint: the interval must have positive width")
    return random.uniform(low, high)

def _poisson(lam=1.0, **_):
    # Poisson distribution: models event counts (non-negative integers).
    # lam (lambda) is the expected number of events.
    # Uses Knuth's algorithm: exact but slow for large lam.
    if lam <= 0:
        raise AetherError(f"Type error: poisson(lam=...) requires lam > 0, got {lam}\n     Hint: lam is the expected event count — it must be positive")
    L, k, p = math.exp(-lam), 0, 1.0
    while p > L: k += 1; p *= random.random()
    return k - 1

def _categorical(probs=None, **_):
    # Categorical distribution: picks one of N discrete outcomes.
    # probs is a list of probabilities that must sum to 1.0.
    # Returns the index (0-based) of the chosen outcome.
    if probs is None: probs = [0.5, 0.5]
    if abs(sum(probs) - 1.0) > 0.01:
        raise AetherError(f"Type error: categorical(probs=...) requires probs to sum to 1.0, got {sum(probs):.4f}\n     Hint: normalize your probabilities so they add up to exactly 1")
    r, cum = random.random(), 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum: return i
    return len(probs) - 1

DISTRIBUTIONS = {
    "normal": _normal, "bernoulli": _bernoulli, "beta": _beta,
    "uniform": _uniform, "poisson": _poisson, "categorical": _categorical,
}


# ─────────────────────────────────────────────
#  Tokenizer
#
#  Converts raw .aeth source text into a flat list of typed tokens.
#  Each token is a 3-tuple: (kind, value, line_number).
#
#  Token kinds:
#    ID     — identifiers and keywords  (model, infer, normal, myVar)
#    NUM    — numeric literals          (3, 0.5, -1 handled at parse time)
#    STRING — string literals           ("path/to/file.aeth")
#    OP     — operators and punctuation (~, =, (, ), [, ], ,, :, .)
#    EOF    — sentinel at end of stream
#
#  Design decision: newlines are consumed silently (they increment the
#  line counter but produce no token). Aether uses indentation for
#  readability only — the grammar is not whitespace-sensitive.
#  Block boundaries are detected by keyword lookahead in the parser.
# ─────────────────────────────────────────────

TOKEN_RE = re.compile(
    r'[ \t]*(?:(#[^\n]*\n?)|(\"(?:[^\"\\]|\\.)*\")|(\b\d+\.\d+\b)|(\b\d+\b)'
    r'|([A-Za-z_][A-Za-z0-9_]*)'
    r'|(~|==|!=|<=|>=|<|>|\*\*|[=\+\-\*/\(\)\[\],:\.]|\n)'
    r')[ \t]*'
)

def tokenize(src):
    toks = []
    line = 1
    for m in TOKEN_RE.finditer(src):
        comment, string, flt, integer, ident, op = m.groups()
        if comment:
            # Comments span to end of line — count the newline they consume
            line += comment.count('\n')
            continue
        elif op == '\n':
            line += 1
            continue
        elif string:  toks.append(("STRING", string[1:-1], line))
        elif flt:     toks.append(("NUM", float(flt), line))
        elif integer: toks.append(("NUM", int(integer), line))
        elif ident:   toks.append(("ID", ident, line))
        elif op:      toks.append(("OP", op, line))
    toks.append(("EOF", None, line))
    return toks


# ─────────────────────────────────────────────
#  Parser
#
#  Hand-written recursive descent parser. Converts the token stream
#  into an Abstract Syntax Tree (AST) made of nested Python tuples.
#
#  Design decision: hand-written over a parser generator (e.g. ANTLR,
#  Lark) because it gives full control over error messages. When the
#  grammar is violated, we can say exactly what was expected and why,
#  rather than surfacing a generic parse error.
#
#  Block detection: Aether has no explicit block delimiters (no `end`,
#  no `}`). Blocks (model bodies, for bodies) end when the parser sees
#  a top-level keyword (model, infer, quantum, for, import).
#  This is stored in BLOCK_KEYWORDS.
#
#  AST node format: each node is a tuple whose first element is a
#  string tag, e.g.:
#    ("model", name, body, line)
#    ("sample", name, dist_node, line)
#    ("observe", name, expr_node, line)
#    ("infer", name, method, n, warmup, step_size)
# ─────────────────────────────────────────────

class ParseError(AetherError): pass

# Keywords that signal the start of a new TOP-LEVEL block.
# Used to detect the end of a model body.
# Note: 'for' is intentionally excluded — for loops can appear INSIDE models
# (hierarchical models). 'for' at top level is also valid but rare.
BLOCK_KEYWORDS = ("model", "infer", "quantum", "import", "hamiltonian", "measure_energy", "graph")

# Keywords that end a for loop body (subset — for can't be nested at top level
# but CAN appear inside a model body)
TOP_LEVEL_KEYWORDS = ("model", "infer", "quantum", "import", "hamiltonian", "measure_energy", "graph")

class AetherParser:
    def __init__(self, tokens, src_lines=None):
        self.tokens = tokens
        self.pos = 0
        self.src_lines = src_lines or []

    def peek(self): return self.tokens[self.pos]
    def line(self): return self.tokens[self.pos][2]

    def consume(self, kind=None, value=None):
        """Advance and return the current token, asserting its kind/value."""
        tok = self.tokens[self.pos]
        if kind and tok[0] != kind:
            raise ParseError(f"\n  ✗  Parse error line {tok[2]}: expected {kind}, got '{tok[1]}'")
        if value and tok[1] != value:
            raise ParseError(f"\n  ✗  Parse error line {tok[2]}: expected '{value}', got '{tok[1]}'")
        self.pos += 1
        return tok

    def match(self, kind, value=None):
        """Consume and return the token if it matches, otherwise return None."""
        tok = self.peek()
        if tok[0] == kind and (value is None or tok[1] == value):
            return self.consume()
        return None

    def parse(self):
        """Entry point: parse all top-level statements."""
        stmts = []
        while self.peek()[0] != "EOF":
            stmts.append(self.parse_statement())
        return stmts

    def parse_statement(self):
        """Dispatch to the appropriate statement parser based on the current token."""
        tok = self.peek()
        if tok[0] == "ID":
            if tok[1] == "model":         return self.parse_model()
            if tok[1] == "quantum":       return self.parse_quantum()
            if tok[1] == "infer":         return self.parse_infer()
            if tok[1] == "observe":       return self.parse_observe()
            if tok[1] == "print":         return self.parse_print()
            if tok[1] == "for":           return self.parse_for()
            if tok[1] == "import":        return self.parse_import()
            if tok[1] == "hamiltonian":   return self.parse_hamiltonian()
            if tok[1] == "measure_energy": return self.parse_measure_energy()
            if tok[1] == "graph":         return self.parse_graph()
            return self.parse_assignment_or_sample()
        raise ParseError(f"\n  ✗  Parse error line {tok[2]}: unexpected token '{tok[1]}'")

    def parse_model(self):
        """
        Parse: model <Name>: <body>
        Body ends when a BLOCK_KEYWORD is encountered at the top level.
        """
        line = self.line()
        self.consume("ID", "model")
        name = self.consume("ID")[1]
        if not self.match("OP", ":"):
            raise ParseError(f"\n  ✗  Parse error line {self.line()}: expected ':' after model name '{name}'\n     Did you forget the colon?  →  model {name}:")
        body = self._parse_block()
        return ("model", name, body, line)

    def parse_quantum(self):
        """
        Parse: quantum circuit <Name>: <body>
        Body contains qubit declarations, gate applications, and measure statements.
        """
        self.consume("ID", "quantum"); self.consume("ID", "circuit")
        name = self.consume("ID")[1]; self.consume("OP", ":")
        body = []
        while self.peek()[0] != "EOF":
            if self.peek()[0] == "ID" and self.peek()[1] in BLOCK_KEYWORDS: break
            s = self.parse_quantum_stmt()
            if s: body.append(s)
        return ("quantum", name, body)

    def _parse_block(self):
        """Parse statements until a top-level keyword is encountered."""
        body = []
        while self.peek()[0] != "EOF":
            if self.peek()[0] == "ID" and self.peek()[1] in BLOCK_KEYWORDS: break
            body.append(self.parse_statement())
        return body

    def parse_quantum_stmt(self):
        """
        Parse one statement inside a quantum circuit block:
          qubit <name>
          gate <name>(target=<q>, control=<q>)
          measure <name>
        Unknown tokens are silently skipped to allow inline comments.
        """
        tok = self.peek()
        if tok[0] == "ID" and tok[1] == "qubit":
            self.consume("ID", "qubit"); name = self.consume("ID")[1]
            return ("qubit", name)
        if tok[0] == "ID" and tok[1] == "gate":
            self.consume("ID", "gate"); gate_name = self.consume("ID")[1]
            self.consume("OP", "(")
            kwargs = {}
            while self.peek()[1] != ")":
                key = self.consume("ID")[1]; self.consume("OP", "="); val = self.consume("ID")[1]
                kwargs[key] = val
                if self.peek()[1] == ",": self.consume("OP", ",")
            self.consume("OP", ")")
            return ("gate", gate_name, kwargs.get("target"), kwargs)
        if tok[0] == "ID" and tok[1] == "measure":
            self.consume("ID", "measure"); name = self.consume("ID")[1]
            return ("measure_q", name)
        self.consume(); return None

    def parse_infer(self):
        """
        Parse: infer <ModelName> using <method>(<params>)

        Supported methods:
          montecarlo(samples=N)                           — rejection sampling
          mcmc(samples=N, warmup=W, step_size=S)         — Metropolis-Hastings
          hmc(samples=N, warmup=W, step_size=S, steps=L) — Hamiltonian MC
          quantum(shots=N)                               — Aether state vector
          qiskit(shots=N)                                — Qiskit Aer simulator
          qiskit(shots=N, backend="ibm_brisbane")        — IBM real hardware
          vqe(hamiltonian=H, shots=N, iterations=I, step_size=S) — VQE closed loop
          qaoa(layers=P, shots=N)                        — QAOA for graph problems
        """
        self.consume("ID", "infer"); name = self.consume("ID")[1]
        method, n, warmup, step_size, n_steps = "montecarlo", 1000, 500, 0.3, 10
        backend = None
        hamiltonian_name = None
        iterations = 80
        layers = 1
        if self.match("ID", "using"):
            method = self.consume("ID")[1]
            if self.match("OP", "("):
                while self.peek()[1] != ")":
                    key = self.consume("ID")[1]; self.consume("OP", "=")
                    if key == "backend":
                        backend = self.consume("STRING")[1]
                    elif key == "hamiltonian":
                        hamiltonian_name = self.consume("ID")[1]
                    else:
                        val = self.consume("NUM")[1]
                        if key in ("samples", "shots"): n = int(val)
                        elif key == "warmup":      warmup = int(val)
                        elif key == "step_size":   step_size = float(val)
                        elif key == "steps":       n_steps = int(val)
                        elif key == "iterations":  iterations = int(val)
                        elif key == "layers":      layers = int(val)
                    if self.peek()[1] == ",": self.consume("OP", ",")
                self.consume("OP", ")")
        return ("infer", name, method, n, warmup, step_size, n_steps, backend,
                hamiltonian_name, iterations, layers)

    def parse_observe(self):
        """
        Parse: observe <name> = <expr>
               observe <name>[<idx>] = <expr>
        expr can be a scalar or a list: observe x = [1, 0, 1, 1]
        Lists represent multiple independent observations of the same variable.
        Indexed form: observe bias["F1"] = 0.7
        """
        line = self.line()
        self.consume("ID", "observe")
        name = self.consume("ID")[1]
        # Check for indexed observe: observe bias["F1"] = 0.7
        if self.peek()[1] == "[":
            self.consume("OP", "[")
            idx = self.parse_expr()
            self.consume("OP", "]")
            self.consume("OP", "=")
            expr = self.parse_expr()
            return ("observe_indexed", name, idx, expr, line)
        self.consume("OP", "=")
        expr = self.parse_expr()
        return ("observe", name, expr, line)

    def parse_print(self):
        self.consume("ID", "print"); return ("print", self.parse_expr())

    def parse_for(self):
        """
        Parse: for <var> in <list>: <body>
        Enables iteration over discrete sets — foundation for hierarchical models.
        Each iteration runs in a child scope; non-loop variables propagate up.
        """
        line = self.line()
        self.consume("ID", "for")
        var = self.consume("ID")[1]
        self.consume("ID", "in")
        iterable = self.parse_expr()
        self.consume("OP", ":")
        body = self._parse_block()
        return ("for", var, iterable, body, line)

    def parse_import(self):
        """
        Parse: import "path/to/file.aeth"
        Loads and executes another .aeth file in the global scope.
        Circular imports are prevented by tracking imported paths.
        """
        line = self.line()
        self.consume("ID", "import")
        path = self.consume("STRING")[1]
        return ("import", path, line)

    def parse_hamiltonian(self):
        """
        Parse: hamiltonian <Name>:
                   term <coeff> identity
                   term <coeff> pauli_z(q0)
                   term <coeff> pauli_x(q0) pauli_x(q1)
                   ...

        Each term is a weighted Pauli string.
        The coefficient is a numeric literal (can be negative).
        The Pauli operators: identity, pauli_x(qN), pauli_y(qN), pauli_z(qN).
        Multiple Pauli operators on one term = tensor product.
        """
        line = self.line()
        self.consume("ID", "hamiltonian")
        name = self.consume("ID")[1]
        self.consume("OP", ":")
        terms = []

        PAULI_OPS = {"identity", "pauli_x", "pauli_y", "pauli_z"}

        while self.peek()[0] != "EOF":
            tok = self.peek()
            # End of hamiltonian block when we hit a top-level keyword
            if tok[0] == "ID" and tok[1] in BLOCK_KEYWORDS and tok[1] != "term":
                break
            if tok[0] == "ID" and tok[1] == "term":
                self.consume("ID", "term")
                # Parse coefficient — may have explicit + or - sign
                coeff_tok = self.peek()
                negative = False
                if coeff_tok[0] == "OP" and coeff_tok[1] == "-":
                    self.consume("OP", "-")
                    negative = True
                elif coeff_tok[0] == "OP" and coeff_tok[1] == "+":
                    self.consume("OP", "+")
                coeff = self.consume("NUM")[1]
                if negative:
                    coeff = -coeff

                # Parse one or more Pauli operators for this term
                ops = {}   # qubit_index -> pauli_name
                while self.peek()[0] == "ID" and self.peek()[1] in PAULI_OPS:
                    pauli_name = self.consume("ID")[1]
                    if pauli_name == "identity":
                        # identity has no qubit argument
                        pass
                    else:
                        # pauli_x(q0), pauli_z(q1), etc.
                        self.consume("OP", "(")
                        qubit_tok = self.consume("ID")[1]  # e.g. "q0", "q1"
                        # Extract qubit index: "q0" -> 0, "q1" -> 1
                        if qubit_tok.startswith("q") and qubit_tok[1:].isdigit():
                            qubit_idx = int(qubit_tok[1:])
                        else:
                            raise ParseError(
                                f"\n  ✗  Parse error line {self.line()}: "
                                f"expected qubit like 'q0', 'q1', got '{qubit_tok}'"
                            )
                        self.consume("OP", ")")
                        ops[qubit_idx] = pauli_name

                terms.append((coeff, ops))
            else:
                # Skip unexpected tokens inside hamiltonian block
                self.consume()

        return ("hamiltonian", name, terms, line)

    def parse_measure_energy(self):
        """
        Parse: measure_energy <HamiltonianName>

        Computes and prints ⟨ψ|H|ψ⟩ for the current quantum state.
        Must appear after an infer ... using quantum(...) statement.
        Requires that a quantum circuit has been run and its state stored.
        """
        line = self.line()
        self.consume("ID", "measure_energy")
        name = self.consume("ID")[1]
        return ("measure_energy", name, line)

    def parse_graph(self):
        """
        Parse: graph <Name>:
                   nodes <N>
                   edge <i> <j>
                   edge <i> <j>
                   ...

        Defines an undirected graph for QAOA optimization.
        nodes declares the number of vertices.
        edge declares an undirected edge between two node indices.

        Example:
          graph Triangle:
              nodes 3
              edge 0 1
              edge 1 2
              edge 0 2
        """
        line = self.line()
        self.consume("ID", "graph")
        name = self.consume("ID")[1]
        self.consume("OP", ":")

        n_nodes = 0
        edges = []

        while self.peek()[0] != "EOF":
            tok = self.peek()
            if tok[0] == "ID" and tok[1] in BLOCK_KEYWORDS:
                break
            if tok[0] == "ID" and tok[1] == "nodes":
                self.consume("ID", "nodes")
                n_nodes = int(self.consume("NUM")[1])
            elif tok[0] == "ID" and tok[1] == "edge":
                self.consume("ID", "edge")
                i = int(self.consume("NUM")[1])
                j = int(self.consume("NUM")[1])
                edges.append((i, j))
            else:
                self.consume()

        return ("graph", name, n_nodes, edges, line)

    def parse_assignment_or_sample(self):
        """
        Parse either:
          <name> ~ <distribution>(...)       — probabilistic sampling
          <name> = <expr>                    — deterministic assignment
          <name>[<idx>] ~ <distribution>(...)— indexed sampling (hierarchical)
          <name>[<idx>] = <expr>             — indexed assignment
        The ~ operator is the core of Aether's probabilistic semantics.
        It means 'is distributed as', not 'equals'.
        """
        line = self.line()
        name = self.consume("ID")[1]

        # Check for indexed variable: media[escola] ~ ...
        if self.peek()[1] == "[":
            self.consume("OP", "[")
            idx = self.parse_expr()
            self.consume("OP", "]")
            if self.peek()[1] == "~":
                self.consume("OP", "~")
                return ("sample_indexed", name, idx, self.parse_distribution(), line)
            elif self.peek()[1] == "=":
                self.consume("OP", "=")
                return ("assign_indexed", name, idx, self.parse_expr(), line)
            raise ParseError(f"\n  ✗  Parse error line {line}: expected '~' or '=' after '{name}[...]'")

        if self.peek()[1] == "~":
            self.consume("OP", "~"); return ("sample", name, self.parse_distribution(), line)
        elif self.peek()[1] == "=":
            self.consume("OP", "="); return ("assign", name, self.parse_expr(), line)
        raise ParseError(f"\n  ✗  Parse error line {line}: expected '~' or '=' after '{name}'")

    def parse_distribution(self):
        """
        Parse: <dist_name>(<key>=<expr>, ...)
        Validates that the distribution name is known at parse time.
        """
        line = self.line()
        name = self.consume("ID")[1]
        if name not in DISTRIBUTIONS:
            raise ParseError(f"\n  ✗  Parse error line {line}: unknown distribution '{name}'\n     Available: {', '.join(DISTRIBUTIONS)}")
        self.consume("OP", "(")
        kwargs = {}
        while self.peek()[1] != ")":
            key = self.consume("ID")[1]; self.consume("OP", "="); val = self.parse_expr()
            kwargs[key] = val
            if self.peek()[1] == ",": self.consume("OP", ",")
        self.consume("OP", ")")
        return ("dist", name, kwargs, line)

    def parse_expr(self):
        """
        Parse an arithmetic expression with left-to-right associativity.
        Supported operators: + - * / **
        Unary minus is handled in parse_primary.
        """
        left = self.parse_primary()
        while self.peek()[1] in ("+", "-", "*", "/", "**"):
            op = self.consume("OP")[1]; right = self.parse_primary()
            left = ("binop", op, left, right)
        return left

    def parse_primary(self):
        """
        Parse a primary expression: literal, variable reference, indexed variable,
        function call, or sub-expression.
        Unary minus is handled here by wrapping as (* -1 <expr>).
        """
        tok = self.peek()
        # Unary minus: -1, -mean, etc.
        if tok[0] == "OP" and tok[1] == "-":
            self.consume()
            inner = self.parse_primary()
            return ("binop", "*", ("num", -1), inner)
        if tok[0] == "NUM":    self.consume(); return ("num", tok[1])
        if tok[0] == "STRING": self.consume(); return ("str", tok[1])
        if tok[0] == "ID":
            name = self.consume()
            # Check for function call: range(...), len(...)
            if self.peek()[1] == "(":
                self.consume("OP", "(")
                args = []
                while self.peek()[1] != ")":
                    args.append(self.parse_expr())
                    if self.peek()[1] == ",": self.consume("OP", ",")
                self.consume("OP", ")")
                return ("call", name[1], args, name[2])
            # Check for indexed variable: media[escola]
            if self.peek()[1] == "[":
                self.consume("OP", "[")
                idx = self.parse_expr()
                self.consume("OP", "]")
                return ("index", name[1], idx, name[2])
            return ("var", name[1], name[2])
        if tok[1] == "(":
            self.consume("OP", "("); e = self.parse_expr(); self.consume("OP", ")"); return e
        if tok[1] == "[":
            self.consume("OP", "["); items = []
            while self.peek()[1] != "]":
                items.append(self.parse_expr())
                if self.peek()[1] == ",": self.consume("OP", ",")
            self.consume("OP", "]"); return ("list", items)
        raise ParseError(f"\n  ✗  Parse error line {tok[2]}: unexpected '{tok[1]}'")


# ─────────────────────────────────────────────
#  Variable scope
#
#  Each model body runs in its own Scope, with a reference to the
#  global scope as its parent. Variable lookup walks up the chain.
#
#  Design decision: isolated scopes prevent models from accidentally
#  sharing variables, which would produce silent bugs in programs
#  that define multiple models in one file.
# ─────────────────────────────────────────────

class Scope:
    """Lexical scope with parent chain for variable lookup."""
    def __init__(self, parent=None):
        self.vars = {}
        self.types = {}
        self.parent = parent

    def get(self, name):
        """Look up a variable, walking up the scope chain."""
        if name in self.vars: return self.vars[name]
        if self.parent: return self.parent.get(name)
        raise AetherError(f"\n  ✗  Name error: '{name}' is not defined")

    def set(self, name, value, atype=None):
        self.vars[name] = value
        if atype: self.types[name] = atype

    def get_type(self, name):
        if name in self.types: return self.types[name]
        if self.parent: return self.parent.get_type(name)
        return AetherType.ANY

    def has(self, name):
        if name in self.vars: return True
        if self.parent: return self.parent.has(name)
        return False


# ─────────────────────────────────────────────
#  Runtime
#
#  Walks the AST and executes each node. Maintains a registry of
#  defined models and quantum circuits, and a global scope for
#  top-level variable assignments.
#
#  Execution is single-pass: models are registered when their
#  definition is encountered, and run when an `infer` statement
#  is executed. This means you can define a model and infer it
#  in the same file, in order.
# ─────────────────────────────────────────────

class AetherRuntime:
    def __init__(self, base_path=None):
        self.models       = {}
        self.circuits     = {}
        self.hamiltonians = {}
        self.graphs       = {}          # name -> Graph object
        self.last_circuit_state = None
        self.global_scope = Scope()
        self.base_path = base_path or os.getcwd()
        self._imported = set()

    def eval_expr(self, node, scope):
        """
        Evaluate an expression node and return its value.
        Dispatches on node type tag (first element of the tuple).
        """
        k = node[0]
        if k == "num":    return node[1]
        if k == "str":    return node[1]
        if k == "var":
            name, line = node[1], node[2]
            try:
                return scope.get(name)
            except AetherError:
                raise AetherError(f"\n  ✗  Name error line {line}: '{name}' is not defined")
        if k == "index":
            # Indexed variable: media[escola] -> looks up "media[A]" in scope
            # Enables hierarchical models where each group has its own variable.
            name, idx_node, line = node[1], node[2], node[3]
            idx = self.eval_expr(idx_node, scope)
            key = f"{name}[{idx}]"
            try:
                return scope.get(key)
            except AetherError:
                raise AetherError(f"\n  ✗  Name error line {line}: '{key}' is not defined")
        if k == "call":
            # Built-in function calls: range(n), len(list)
            fname, args, line = node[1], node[2], node[3]
            evaluated = [self.eval_expr(a, scope) for a in args]
            if fname == "range":
                if len(evaluated) == 1:
                    return list(range(int(evaluated[0])))
                elif len(evaluated) == 2:
                    return list(range(int(evaluated[0]), int(evaluated[1])))
                raise AetherError(f"\n  ✗  range() takes 1 or 2 arguments, got {len(evaluated)}")
            if fname == "len":
                if len(evaluated) != 1:
                    raise AetherError(f"\n  ✗  len() takes 1 argument")
                return len(evaluated[0])
            raise AetherError(f"\n  ✗  Unknown function '{fname}' line {line}")
        if k == "list":   return [self.eval_expr(i, scope) for i in node[1]]
        if k == "binop":
            _, op, l, r = node
            lv, rv = self.eval_expr(l, scope), self.eval_expr(r, scope)
            if op == "+" : return lv + rv
            if op == "-" : return lv - rv
            if op == "*" : return lv * rv
            if op == "/" :
                if rv == 0: raise AetherError("Division by zero")
                return lv / rv
            if op == "**": return lv ** rv
        if k == "dist":
            # Evaluate distribution parameters, then call the sampler.
            # This is where ~ expressions are actually executed.
            _, dist_name, kwargs_nodes, line = node
            resolved = {k2: self.eval_expr(v, scope) for k2, v in kwargs_nodes.items()}
            fn = DISTRIBUTIONS.get(dist_name)
            if not fn: raise AetherError(f"\n  ✗  Unknown distribution '{dist_name}' line {line}")
            return fn(**resolved)
        raise AetherError(f"Unknown expr node: {k}")

    def exec_stmt(self, stmt, scope):
        """Execute a single AST statement node in the given scope."""
        k = stmt[0]

        if k == "sample":
            # x ~ dist(...) — sample from distribution and store in scope.
            # Type-check the result against the distribution's known type.
            _, name, dist_node, line = stmt
            val = self.eval_expr(dist_node, scope)
            atype = DIST_TYPES.get(dist_node[1], AetherType.ANY)
            check_type(name, val, atype, line)
            scope.set(name, val, atype)

        elif k == "sample_indexed":
            # media[escola] ~ dist(...) — indexed variable sampling.
            # Stores as "media[A]", "media[B]", etc. in scope.
            # This is the foundation of hierarchical models.
            _, name, idx_node, dist_node, line = stmt
            idx = self.eval_expr(idx_node, scope)
            key = f"{name}[{idx}]"
            val = self.eval_expr(dist_node, scope)
            atype = DIST_TYPES.get(dist_node[1], AetherType.ANY)
            check_type(key, val, atype, line)
            scope.set(key, val, atype)

        elif k == "assign":
            # x = expr — deterministic assignment, no distribution.
            _, name, expr, line = stmt
            val = self.eval_expr(expr, scope)
            scope.set(name, val)

        elif k == "assign_indexed":
            # media[escola] = expr — indexed deterministic assignment.
            _, name, idx_node, expr, line = stmt
            idx = self.eval_expr(idx_node, scope)
            key = f"{name}[{idx}]"
            val = self.eval_expr(expr, scope)
            scope.set(key, val)

        elif k == "observe":
            # observe x = val or observe x = [v1, v2, ...]
            # In rejection sampling: checks sampled value against observed.
            # In MCMC: scored via log-likelihood in mcmc.py.
            # Lists enable multiple observations of the same variable.
            _, name, expr, line = stmt
            val = self.eval_expr(expr, scope)
            if isinstance(val, list):
                scope.set(f"__obs_{name}", val)
            else:
                scope.set(name, val)

        elif k == "observe_indexed":
            # observe bias["F1"] = 0.7 — condition on a specific group's value
            _, name, idx_node, expr, line = stmt
            idx = self.eval_expr(idx_node, scope)
            key = f"{name}[{idx}]"
            val = self.eval_expr(expr, scope)
            scope.set(key, val)

        elif k == "print":
            val = self.eval_expr(stmt[1], scope)
            print(f"  → {val}")

        elif k == "model":
            # Register the model body for later inference.
            # Models are not executed at definition time.
            _, name, body, line = stmt
            self.models[name] = body

        elif k == "quantum":
            # Register the quantum circuit body for later simulation.
            self.circuits[stmt[1]] = stmt[2]

        elif k == "hamiltonian":
            # Build and register a Hamiltonian from the parsed terms.
            _, name, terms, line = stmt
            from hamiltonian import Hamiltonian
            h = Hamiltonian(name)
            for coeff, ops in terms:
                h.add_term(coeff, ops)
            self.hamiltonians[name] = h

        elif k == "measure_energy":
            # Print ⟨ψ|H|ψ⟩ for the last-run quantum circuit state.
            _, name, line = stmt
            from hamiltonian import exact_energy, print_hamiltonian
            if name not in self.hamiltonians:
                raise AetherError(f"\n  ✗  measure_energy: Hamiltonian '{name}' not defined")
            h = self.hamiltonians[name]
            if self.last_circuit_state is None:
                raise AetherError(
                    f"\n  ✗  measure_energy: no quantum circuit has been run yet\n"
                    f"     Run a circuit with 'infer <Circuit> using quantum(...)' first"
                )
            energy = exact_energy(self.last_circuit_state, h)
            print(f"\n  ⟁  measure_energy '{name}'")
            print(f"  {'─'*40}")
            print(f"  ⟨ψ|H|ψ⟩ = {energy:+.8f} Hartree")
            print(f"  (exact state vector, no shot noise)\n")

        elif k == "for":
            # Iterate over a list, running the body in a child scope per item.
            # Child scope inherits all parent variables (including global_mean,
            # population_mean, etc.) so hierarchical models work correctly.
            # After each iteration, indexed variables propagate back up.
            _, var, iterable_node, body, line = stmt
            iterable = self.eval_expr(iterable_node, scope)
            if not isinstance(iterable, list):
                raise AetherError(f"\n  ✗  for loop line {line}: expected a list, got {type(iterable).__name__}")
            for item in iterable:
                loop_scope = Scope(parent=scope)
                loop_scope.set(var, item)
                for s in body:
                    self.exec_stmt(s, loop_scope)
                # Propagate ALL new variables (including indexed ones) back up
                for vname, vval in loop_scope.vars.items():
                    if vname != var:
                        scope.set(vname, vval)

        elif k == "import":
            # Load and execute another .aeth file in the global scope.
            # Paths are resolved relative to the importing file's directory.
            _, path, line = stmt
            full_path = os.path.join(self.base_path, path)
            if not os.path.exists(full_path):
                raise AetherError(f"\n  ✗  Import error line {line}: file not found '{full_path}'")
            if full_path in self._imported:
                return  # already imported — skip to prevent double-execution
            self._imported.add(full_path)
            with open(full_path) as f: src = f.read()
            toks = tokenize(src)
            ast  = AetherParser(toks, src.splitlines()).parse()
            for s in ast:
                self.exec_stmt(s, self.global_scope)

        elif k == "graph":
            # Build and register a Graph for QAOA.
            _, name, n_nodes, edges, line = stmt
            from qaoa_engine import Graph
            g = Graph(name, n_nodes)
            for i, j in edges:
                g.add_edge(i, j)
            self.graphs[name] = g

        elif k == "infer":
            # Route to the appropriate inference engine based on method.
            _, name, method, n, warmup, step_size, n_steps, backend, hamiltonian_name, iterations, layers = stmt
            if method == "quantum":   self.run_quantum(name, n)
            elif method == "qiskit":  self.run_qiskit(name, n, backend)
            elif method == "hmc":     self.run_hmc(name, n, warmup, step_size, n_steps)
            elif method == "mcmc":    self.run_mcmc(name, n, warmup, step_size)
            elif method == "vqe":     self.run_vqe(name, hamiltonian_name, n, iterations, step_size)
            elif method == "qaoa":    self.run_qaoa(name, n, layers)
            else:                     self.run_classical(name, n)

    # ── Classical rejection sampling ─────────────────────────────────────
    #
    #  The simplest possible inference algorithm:
    #  1. Sample all latent variables from their priors.
    #  2. Check sampled values against observations.
    #  3. If they match (within tolerance), keep the sample.
    #  4. Repeat.
    #
    #  Pros: trivial to implement, always correct.
    #  Cons: acceptance rate drops exponentially with model complexity.
    #        Use mcmc() for models with > 2-3 variables or tight observations.
    # ─────────────────────────────────────────────────────────────────────

    def run_classical(self, name, samples):
        if name not in self.models:
            raise AetherError(f"\n  ✗  Model '{name}' is not defined")
        body = self.models[name]
        results = defaultdict(list); accepted = 0

        print(f"\n  ⟁  Aether — inferring '{name}' ({samples:,} samples)")
        print(f"  {'─'*48}")

        for _ in range(samples):
            scope = Scope(parent=self.global_scope)
            valid = True
            for stmt in body:
                k = stmt[0]
                try:
                    if k == "sample":
                        val = self.eval_expr(stmt[2], scope)
                        scope.set(stmt[1], val)
                    elif k == "assign":
                        scope.set(stmt[1], self.eval_expr(stmt[2], scope))
                    elif k == "observe":
                        obs_val = self.eval_expr(stmt[2], scope)
                        sam_val = scope.vars.get(stmt[1])
                        # Tolerance-based comparison: exact match is impossible
                        # for continuous variables. 15% relative tolerance is a
                        # practical choice — tight enough to be meaningful,
                        # loose enough to get non-zero acceptance.
                        if isinstance(obs_val, list):
                            for obs in obs_val:
                                if sam_val is None:
                                    scope.set(stmt[1], obs)
                                else:
                                    tol = max(abs(obs) * 0.15, 0.15)
                                    if abs(sam_val - obs) > tol:
                                        valid = False; break
                        else:
                            if sam_val is None:
                                scope.set(stmt[1], obs_val)
                            else:
                                tol = max(abs(obs_val) * 0.15, 0.15)
                                if abs(sam_val - obs_val) > tol:
                                    valid = False; break
                except AetherError:
                    valid = False; break
                if not valid: break

            if valid:
                accepted += 1
                for kv, v in scope.vars.items():
                    if not kv.startswith("_"): results[kv].append(v)

        if accepted == 0:
            print("  ✗  No samples accepted. Check your observations or use mcmc.")
            return

        print(f"  Accepted: {accepted:,}/{samples:,} ({accepted/samples*100:.1f}%)\n")
        for var, vals in results.items():
            mean = sum(vals) / len(vals)
            std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            lo   = sorted(vals)[int(len(vals) * 0.05)]
            hi   = sorted(vals)[int(len(vals) * 0.95)]
            print(f"  {var}\n    mean = {mean:.4f}   std = {std:.4f}")
            print(f"    90% CI = [{lo:.4f}, {hi:.4f}]\n")

    # ── MCMC ──────────────────────────────────────────────────────────────
    #
    #  Delegates to mcmc.py which implements Metropolis-Hastings.
    #  We pass an eval_fn closure so mcmc.py can evaluate expression nodes
    #  without importing the full interpreter — keeping the modules decoupled.
    # ─────────────────────────────────────────────────────────────────────

    def run_mcmc(self, name, samples, warmup, step_size):
        if name not in self.models:
            raise AetherError(f"\n  ✗  Model '{name}' is not defined")
        from mcmc import run_mcmc, print_mcmc_results
        body = self.models[name]
        def eval_fn(node, env):
            s = Scope(parent=self.global_scope)
            s.vars.update(env)
            return self.eval_expr(node, s)
        result = run_mcmc(body, eval_fn, samples=samples, warmup=warmup, step_size=step_size)
        print_mcmc_results(name, result, warmup)

    def run_hmc(self, name, samples, warmup, step_size, n_steps):
        if name not in self.models:
            raise AetherError(f"\n  ✗  Model '{name}' is not defined")
        from hmc import run_hmc, print_hmc_results
        body = self.models[name]
        def eval_fn(node, env):
            s = Scope(parent=self.global_scope)
            s.vars.update(env)
            return self.eval_expr(node, s)
        result = run_hmc(body, eval_fn, samples=samples, warmup=warmup,
                        step_size=step_size, n_steps=n_steps)
        print_hmc_results(name, result, warmup)

    # ── Quantum ───────────────────────────────────────────────────────────
    #
    #  Delegates to quantum.py which implements the state vector simulator.
    #  The same design: circuit body is passed, quantum.py handles execution.
    # ─────────────────────────────────────────────────────────────────────

    def run_qiskit(self, name, shots, backend=None):
        """
        Compile and run an Aether circuit via Qiskit.
        Routes to Aer local simulator or IBM hardware based on backend param.
        """
        from qiskit_backend import run_qiskit_circuit
        if name not in self.circuits:
            raise AetherError(f"\n  ✗  Quantum circuit '{name}' is not defined")
        run_qiskit_circuit(name, self.circuits[name], shots=shots, backend=backend)

    def run_quantum(self, name, shots):
        from quantum import parse_quantum_model, build_circuit
        if name not in self.circuits:
            raise AetherError(f"\n  ✗  Quantum circuit '{name}' is not defined")
        state = parse_quantum_model(name, self.circuits[name], shots)
        # Store the final state vector so measure_energy can use it
        if state is not None:
            self.last_circuit_state = state

    def run_vqe(self, circuit_name, hamiltonian_name, shots, iterations, step_size):
        """
        Run the closed VQE loop:
          - Uses the Hamiltonian defined by hamiltonian_name
          - Optimizes a hardware-efficient ansatz (Ry + CNOT + Ry)
          - Does NOT require a separately defined quantum circuit —
            the ansatz is built automatically from the Hamiltonian's qubit count
          - If circuit_name exists as a registered circuit, uses its qubit count

        This is the unified hybrid loop:
          classical gradient descent ←→ quantum energy measurement
        """
        from vqe_engine import run_vqe as _run_vqe, print_vqe_results
        if hamiltonian_name is None:
            raise AetherError(
                "\n  ✗  VQE requires hamiltonian=<Name>\n"
                "     Example: infer Ansatz using vqe(hamiltonian=H2, shots=1024)"
            )
        if hamiltonian_name not in self.hamiltonians:
            raise AetherError(
                f"\n  ✗  Hamiltonian '{hamiltonian_name}' not defined\n"
                f"     Define it with:  hamiltonian {hamiltonian_name}: ..."
            )
        h = self.hamiltonians[hamiltonian_name]
        result = _run_vqe(
            hamiltonian=h,
            shots=shots,
            iterations=iterations,
            step_size=step_size,
        )
        print_vqe_results(circuit_name, result)

    def run_qaoa(self, graph_name: str, shots: int, layers: int):
        """
        Run QAOA for MaxCut on a registered graph.

        The graph must be defined with a graph block:
          graph MyGraph:
              nodes N
              edge i j
              ...

        infer MyGraph using qaoa(layers=P, shots=N)
        """
        from qaoa_engine import run_qaoa as _run_qaoa, print_qaoa_results
        if graph_name not in self.graphs:
            raise AetherError(
                f"\n  ✗  Graph '{graph_name}' not defined\n"
                f"     Define it with:\n"
                f"       graph {graph_name}:\n"
                f"           nodes N\n"
                f"           edge i j"
            )
        g = self.graphs[graph_name]
        result = _run_qaoa(g, layers=layers, shots=shots)
        # Pass edges for cut calculation in print
        result["edges"] = g.edges
        print_qaoa_results(graph_name, result)

    def run(self, stmts):
        """Execute a list of top-level AST statements."""
        for s in stmts:
            self.exec_stmt(s, self.global_scope)


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def run_file(path):
    """Load and execute a .aeth file. Prints clean errors without Python tracebacks."""
    with open(path) as f: src = f.read()
    src_lines = src.splitlines()
    try:
        toks = tokenize(src)
        ast  = AetherParser(toks, src_lines).parse()
        rt   = AetherRuntime(base_path=os.path.dirname(os.path.abspath(path)))
        rt.run(ast)
    except (AetherError, RuntimeError) as e:
        msg = str(e)
        if "✗" in msg:
            lines = [l for l in msg.splitlines() if "✗" in l or "Hint" in l or "line" in l.lower()]
            print("\n" + "\n".join(lines))
        else:
            print(f"\n  ✗  Error: {msg}")
        sys.exit(1)

def run_source(src, base_path=None):
    """Execute Aether source code from a string. Used by the web playground."""
    src_lines = src.splitlines()
    toks = tokenize(src)
    ast  = AetherParser(toks, src_lines).parse()
    rt   = AetherRuntime(base_path=base_path or os.getcwd())
    rt.run(ast)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python interpreter.py <file.aeth>")
        sys.exit(1)
    run_file(sys.argv[1])
