"""
Defines the specific Clifford gates used in the project, including their unitary and 
symplectic matrix representations.
"""

import numpy as np
from math import gcd
from sympy import Matrix, isprime

from qudit_clifford_synthesis.Essentials.Gates import ToKronIndex, Gate, GateSet

# Useful Functions ############################################################################

def EmbedSymGate(circ_num_qudits: int, gate_sym: np.ndarray, which_qudits: list[int]
                 ) -> np.ndarray:
    """
    Embeds a smaller symplectic matrix into the full circuit's "symplectic space".

    Args:
        circ_num_qudits (int): The total number of qudits in the circuit.

        gate_sym (np.ndarray): The symplectic matrix of the gate to embed.

        which_qudits (list[int]): A list of indices specifying which qudits the
            gate is applied to.

    Returns:
        circuit_sym (np.ndarray): The resulting (2N, 2N) symplectic matrix.
    """
    # Number of qudits the gate acts on
    n = round(len(gate_sym)/2)
    
    # Number of qudits in the circuit
    N = circ_num_qudits

    circuit_sym = np.eye(2*N, dtype = np.uint32)
    
    # Embedding.
    # Recall the ordering convention is [x_0, x_1, ... x_n, z_0, z_1, ... z_n]
    for qudit_number in range(n):
        x_index = qudit_number
        z_index = qudit_number + n
        X_index = which_qudits[qudit_number]
        Z_index = which_qudits[qudit_number] + N

        for qudit_number_prime in range(n):
            x_prime_index = qudit_number_prime
            z_prime_index = qudit_number_prime + n
            X_prime_index = which_qudits[qudit_number_prime]
            Z_prime_index = which_qudits[qudit_number_prime] + N
            
            circuit_sym[X_index, X_prime_index] = gate_sym[x_index, x_prime_index]
            circuit_sym[Z_index, Z_prime_index] = gate_sym[z_index, z_prime_index]
            circuit_sym[X_index, Z_prime_index] = gate_sym[x_index, z_prime_index]
            circuit_sym[Z_index, X_prime_index] = gate_sym[z_index, x_prime_index]
    
    return circuit_sym

def MatModMulInv(matrix: np.ndarray, mod: int) -> np.ndarray:
    """
    Calculates the modular multiplicative inverse of a matrix using SymPy.

    Args:
        matrix (np.ndarray): The input matrix.

        mod (int): The modulus.

    Returns:
        inv_matrix (np.ndarray): The modular inverse of the matrix.
    """
    M = Matrix(matrix)
    M_inv = M.inv_mod(mod) # type: ignore
    return np.array(M_inv.tolist(), dtype = np.uint32)

def SymForm(num_qudits: int, num_lvs: int) -> np.ndarray:
    """
    Generates the standard symplectic form matrix.
    """
    I = np.eye(num_qudits, dtype = int)
    O = np.zeros((num_qudits, num_qudits), dtype=int)
    return np.block([[O, I],[-I, O]]) % num_lvs

# Structures ##################################################################################

class CliffGate(Gate):
    """
    A special case (child) of the "Gate" class. Includes a symplectic matrix representation.
    """
    def __init__(self, num_qudits: int, num_lvs: int, gate_id: str, unitary_rep: np.ndarray, 
                 sym_rep: np.ndarray):
        """
        Initializes a Clifford gate.

        Args:
            sym_rep (np.ndarray): The (2*num_qudits, 2*num_qudits) symplectic matrix 
            representation of the gate.
            
            Other args are inherited from the base Gate class. Ref annotations on 
            Essentials.Gates.Gate.
        """
        self.num_qudits = num_qudits
        self.num_lvs = num_lvs
        self.gate_id = gate_id
        self.unitary_rep = unitary_rep
        self.sym_rep = sym_rep

        # For convenience we store the inverse representation as an attribute, but
        # this may have to change in the future for large matrices.
        self.inv_sym_rep = self.InvSymRep()

        super().__init__(
            num_qudits = self.num_qudits, num_lvs = self.num_lvs, gate_id = self.gate_id, 
            unitary_rep = self.unitary_rep
        )
        # Verifies sym_rep's shape is correct.
        if sym_rep.shape != (2*num_qudits, 2*num_qudits):
            raise ValueError(f"sym_rep.shape = {sym_rep.shape}\n\
                            != (2*num_qudits, 2*num_qudits) = {(2*num_qudits, 2*num_qudits)}")
        
        # Verifies sym_rep is symplectic; S^{T}ΩS = Ω, for all S ∈ Sym(2n).
        Ω = SymForm(num_qudits = num_qudits, num_lvs = num_lvs)
        if not np.array_equal((sym_rep.T @ Ω @ sym_rep) % num_lvs, Ω):
            raise ValueError(f"Ω ≠ (sym_rep.T @ Ω @ sym_rep) % num_lvs =\n{(sym_rep.T @ Ω @ sym_rep) % num_lvs}")

    def InvSymRep(self) -> np.ndarray:
        """
        Calculates the inverse of the gate's symplectic matrix (modular inverse).

        Returns:
            (np.ndarray): The inverse of the symplectic matrix.
        """
        return MatModMulInv(matrix = self.sym_rep, mod = self.num_lvs)

class CliffGateSet(GateSet):
    """
    A specific GateSet that can generate all Clifford gates corresponding to 
    n = num_qudits, d = num_lvs. 

    Ref https://arxiv.org/abs/quant-ph/0408190
    """
    def __init__(self, num_lvs: int):
        self.num_lvs = num_lvs
        self.gate_set: set[Gate] = set([
            FGate(num_lvs = num_lvs), 
            PGate(num_lvs = num_lvs),
            SUMGate(num_lvs = num_lvs)
        ])
        self.is_prime = isprime(num_lvs)
        if not self.is_prime:
            self.multiplicative_inverse = [lv for lv in range(num_lvs) if gcd(lv % num_lvs, num_lvs) == 1]
            for a in self.multiplicative_inverse:
                self.gate_set.add(MGate(num_lvs = num_lvs, a = a))
        
        super().__init__(gate_set = self.gate_set)

# Clifford Gates ##############################################################################

class FGate(CliffGate):
    """
    Fourier Transform gate (Generalized Hadamard)
    """
    def __init__(self, num_lvs: int):
        self.num_qudits = 1
        self.num_lvs = num_lvs
        self.gate_id = "F"
        
        omega = np.exp(2j * np.pi / num_lvs)
        self.unitary_rep = np.array([[omega**(j * k) for k in range(num_lvs)] for j in range(num_lvs)], 
                                    dtype=complex) / np.sqrt(num_lvs)

        self.sym_rep = np.array([[0, 1], [(-1) % num_lvs, 0]], dtype = np.uint32)

        super().__init__(
            num_qudits = self.num_qudits, num_lvs = self.num_lvs, gate_id = self.gate_id, 
            unitary_rep = self.unitary_rep, sym_rep = self.sym_rep
        )

class PGate(CliffGate):
    """
    Phase gate (Generalized S)
    """
    def __init__(self, num_lvs: int):
        self.num_qudits = 1
        self.num_lvs = num_lvs
        self.gate_id = "P"
        
        omega = np.exp(2j * np.pi / num_lvs)
        self.unitary_rep = np.diag([omega**((j * (j - 1)) // 2 % num_lvs) for j in range(num_lvs)])

        self.sym_rep = np.array([[1, 0], [1, 1]], dtype = np.uint32)

        super().__init__(
            num_qudits = self.num_qudits, num_lvs = self.num_lvs, gate_id = self.gate_id, 
            unitary_rep = self.unitary_rep, sym_rep = self.sym_rep
        )

class MGate(CliffGate):
    """
    Modular Multiplicative Inverse gate (Not necessary for prime d = num_lvs)
    """
    def __init__(self, num_lvs: int, a: int):
        self.num_qudits = 1
        self.num_lvs = num_lvs
        self.gate_id = "M"+str(a)

        if gcd(a, num_lvs) != 1:
            raise ValueError(f"{a} has no multiplicative inverse mod {num_lvs}")
        
        M = np.zeros((num_lvs, num_lvs), dtype=complex)
        for j in range(num_lvs):
            M[(a * j) % num_lvs, j] = 1
        self.unitary_rep = M

        a_inv = pow(a, -1, num_lvs)
        self.sym_rep = np.array([[a % num_lvs, 0], [0, a_inv % num_lvs]], dtype = np.uint32)

        super().__init__(
            num_qudits = self.num_qudits, num_lvs = self.num_lvs, gate_id = self.gate_id, 
            unitary_rep = self.unitary_rep, sym_rep = self.sym_rep
        )

class SUMGate(CliffGate):
    """
    Controlled Sum gate (Generalized CNOT)
    """
    def __init__(self, num_lvs: int):
        self.num_qudits = 2
        self.num_lvs = num_lvs
        self.gate_id = "SUM"

        SUM = np.zeros((num_lvs**2, num_lvs**2), dtype=complex)
        for ctrl_index in range(num_lvs):
            for target_index in range(num_lvs):
                input_index = ToKronIndex([ctrl_index, target_index], num_lvs)
                output_index = ToKronIndex([ctrl_index, (ctrl_index + target_index) % num_lvs], num_lvs)
                SUM[output_index, input_index] = 1
        self.unitary_rep = SUM
        
        #[x_0, x_1 | z_0, z_1]
        self.sym_rep = np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, (-1) % num_lvs],
            [0, 0, 0, 1]
        ], dtype = np.uint32)

        super().__init__(
            num_qudits = self.num_qudits, num_lvs = self.num_lvs, gate_id = self.gate_id, 
            unitary_rep = self.unitary_rep, sym_rep = self.sym_rep
        )

# End of File #################################################################################