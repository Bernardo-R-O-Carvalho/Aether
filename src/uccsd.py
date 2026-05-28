"""
Aether UCCSD Ansatz
===================

Unitary Coupled Cluster Singles and Doubles (UCCSD) via exact matrix exponentiation.

|ψ(θ)⟩ = exp(T(θ) - T†(θ)) |HF⟩

onde T = Σ θ_k * (a†_a a_i - a†_i a_a)  [singles]
       + Σ θ_k * (a†_a a†_b a_j a_i - h.c.)  [doubles]

Implementação via scipy.linalg.expm — exata, sem aproximação de Trotter.
Usa os amplitudes CCSD como ponto de partida (via OpenFermion).

Uso:
  ansatz = UCCSDansatz(mol)
  state = ansatz.prepare(thetas)
  n_params = ansatz.n_params
"""

import numpy as np
from scipy.linalg import expm

try:
    from openfermion.circuits import (
        uccsd_singlet_generator,
        uccsd_singlet_get_packed_amplitudes,
    )
    from openfermion.linalg import get_sparse_operator
    _OF_AVAILABLE = True
except ImportError:
    _OF_AVAILABLE = False

# Known molecule configs: n_qubits -> (n_electrons, n_params)
MOLECULE_CONFIGS = {
    6:  (2, "LiH (frozen core)"),
    8:  (4, "BeH₂ (frozen core)"),
    10: (8, "H₂O (frozen core)"),
}


class UCCSDansatz:
    """
    UCCSD ansatz using exact matrix exponentiation via OpenFermion.

    Parameters
    ----------
    n_qubits   : total qubits
    n_electrons: electrons in active space
    ccsd_amps  : optional (singles, doubles) CCSD amplitudes for warm start
    """
    def __init__(self, n_qubits: int, n_electrons: int, ccsd_amps=None):
        if not _OF_AVAILABLE:
            raise ImportError(
                "UCCSD requires OpenFermion: pip install openfermion openfermionpyscf"
            )
        self.n_qubits    = n_qubits
        self.n_electrons = n_electrons
        self._ccsd_amps  = ccsd_amps

        # Determine correct n_params by probing OpenFermion
        # uccsd_singlet_generator uses a specific packed format
        # that doesn't match the naive combinatorial count
        n_params_found = None
        for n_try in range(1, 200):
            try:
                from openfermion.circuits import uccsd_singlet_generator
                uccsd_singlet_generator([0.0]*n_try, n_qubits, n_electrons)
                n_params_found = n_try
                break
            except (IndexError, Exception):
                continue
        self.n_params = n_params_found or 5

        # HF state index
        hf_idx = 0
        for i in range(n_electrons):
            hf_idx |= (1 << (n_qubits - 1 - i))
        self._hf_idx = hf_idx

    def initial_params(self) -> list:
        """Return CCSD amplitudes as warm-start parameters, or zeros."""
        if self._ccsd_amps is not None:
            s_amps, d_amps = self._ccsd_amps
            from openfermion.circuits import uccsd_singlet_get_packed_amplitudes
            packed = uccsd_singlet_get_packed_amplitudes(
                s_amps, d_amps, self.n_qubits, self.n_electrons
            )
            return [float(x) for x in packed]
        return [0.0] * self.n_params

    def prepare(self, thetas: list) -> list:
        """
        Prepare |ψ(θ)⟩ = exp(T(θ) - T†(θ))|HF⟩

        Uses exact matrix exponentiation — no Trotter error.
        Returns the state vector as a Python list of complex numbers.
        """
        dim = 2 ** self.n_qubits

        # Build the anti-Hermitian generator T - T†
        gen = uccsd_singlet_generator(
            list(thetas), self.n_qubits, self.n_electrons
        )
        gen_mat = get_sparse_operator(gen, n_qubits=self.n_qubits).toarray()
        anti_herm = gen_mat - gen_mat.conj().T  # ensures unitarity

        # Exact unitary: U = exp(T - T†)
        U = expm(anti_herm)

        # Apply to HF state
        hf = np.zeros(dim, dtype=complex)
        hf[self._hf_idx] = 1.0
        psi = U @ hf

        return list(psi)

    def __repr__(self):
        return (
            f"UCCSDansatz(n_qubits={self.n_qubits}, "
            f"n_electrons={self.n_electrons}, "
            f"n_params={self.n_params})"
        )


def make_uccsd_for_hamiltonian(hamiltonian, ccsd_amps=None) -> UCCSDansatz:
    """
    Auto-detect molecule from Hamiltonian qubit count and return UCCSD ansatz.
    """
    n = hamiltonian.n_qubits()
    if n not in MOLECULE_CONFIGS:
        raise ValueError(
            f"No UCCSD config for {n}-qubit Hamiltonian. "
            f"Known: {list(MOLECULE_CONFIGS.keys())}"
        )
    n_elec, _, name = MOLECULE_CONFIGS[n]
    return UCCSDansatz(n_qubits=n, n_electrons=n_elec, ccsd_amps=ccsd_amps)
