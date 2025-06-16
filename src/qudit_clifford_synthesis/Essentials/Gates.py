"""
Provides a structures for representing quantum gates and gate sets.
"""

import numpy as np
from math import log
from itertools import product

# Useful Functions ############################################################################

def MatrixPower(matrix: np.ndarray, power: int) -> np.ndarray:
    """
    Computes the matrix power of a given square matrix.

    Args:
        matrix (np.ndarray): The square matrix to be raised to a power.
        
        power (int): The integer power to raise the matrix to.

    Returns:
        (np.ndarray): The resulting matrix.
    """
    output = matrix
    for _ in range(power-1):
        output = output @ matrix
    return output

def ToKronIndex(index_list: list[int] | np.ndarray, d: int) -> int:
    """
    Converts a list of indices for a tensor product space into a single flattened index.

    Args:
        index_list (list[int] | np.ndarray): A list of indices, where each index corresponds 
        to a subsystem.
        
        d (int): The dimension of each subsystem (number of states each qudit can occupy).

    Returns:
        (int): The single flattened index corresponding to the tensor product basis.
    """
    index_list = index_list.copy()
    if len(index_list) == 1:
        return index_list[0]
    else:
        first_elem, remaining_index_list = index_list[0], index_list[1:]
        return first_elem * d**len(remaining_index_list) + ToKronIndex(remaining_index_list, d)

def FromKronIndex(index: int, n: int, d: int) -> tuple[int, ...]:
    """
    Converts a single flattened index for a tensor product space back into a tuple of indices.

    Args:
        index (int): The single flattened integer index.
        
        n (int): The number of subsystems (number of qudits).
        
        d (int): The dimension of each subsystem (number of states each qudit can occupy).

    Returns:
        (tuple[int, ...]): A tuple of indices corresponding to each subsystem.
    """
    if index >= d**n:
        raise ValueError("Index is out of range")
    if n == 1:
        return (index,)
    else:
        return FromKronIndex(index // d, n-1, d) + (index % d,) 
    
def EmbedUnitaryGate(circ_num_qudits: int, gate_matrix: np.ndarray, 
                     which_qudits: list[int] | np.ndarray) -> np.ndarray:
    """
    Embeds a smaller unitary gate into a larger quantum circuit's unitary matrix.
    Ref "Embedding Unitary Gates" on Noteful.
    TODO: Try to make this more efficient.

    Args:
        circ_num_qudits (int): The total number of qudits in the circuit.
        
        gate_matrix (np.ndarray): The unitary matrix of the gate to be embedded.
        
        which_qudits (list[int] | np.ndarray): A list of indices specifying which qudits 
            the gate acts on.

    Returns:
        (np.ndarray): The full unitary matrix for the circuit with the embedded gate.
    """
    d = round(len(gate_matrix)**(1/len(which_qudits)))
    n = round(log(len(gate_matrix), d))
    N = circ_num_qudits

    # For single-qudit gates, we have an efficient implementation
    if n == 1:
        target = which_qudits[0]
        IM = np.kron(np.eye(d ** target), gate_matrix)
        IMI = np.kron(IM, np.eye(d ** (N - target - 1)))
        return IMI

    # Embedding for general (non single-qudit) gates ----------------------------------------------
    circuit_matrix = np.eye(d ** circ_num_qudits, dtype = np.complex128)
    core_indices = [list(elem) for elem in product(range(d), repeat = n)]
    all_indices = [list(elem) for elem in product(range(d), repeat = N)]
    
    for core_index_vector1 in core_indices:
        for core_index_vector2 in core_indices:
            for full_index_vector in all_indices:

                full_index_vector1 = np.array(full_index_vector)[which_qudits] = core_index_vector1
                full_index_vector2 = np.array(full_index_vector)[which_qudits] = core_index_vector2

                circuit_matrix[ToKronIndex(full_index_vector1, d), 
                               ToKronIndex(full_index_vector2, d)] \
                = gate_matrix[ToKronIndex(core_index_vector1, d), 
                              ToKronIndex(core_index_vector2, d)]
    #-----------------------------------------------------------------------------------------------
    return circuit_matrix

# Structures ##################################################################################

class Gate():
    """
    Represents a quantum gate with its unitary matrix.
    """
    def __init__(self, num_qudits: int, num_lvs: int, gate_id: str, unitary_rep: np.ndarray):
        """
        Args:
            num_qudits (int): The number of qudits the gate acts on.
            
            num_lvs (int): The number of states for each qudit.
            
            gate_id (str): A unique string ID for the gate.
            
            unitary_rep (np.ndarray): The unitary matrix representation of the gate.
        """
        self.num_qudits = num_qudits
        self.num_lvs = num_lvs
        self.gate_id = gate_id
        self.unitary_rep = unitary_rep

        # For convenience we store the inverse representation as an attribute, but
        # this may have to change in the future for large matrices.
        self.inv_unitary_rep = self.InvUnitaryRep()

        # Basic validations ------------------------------------------------------------------------
        if self.num_qudits < 1:
            raise ValueError(f"self.num_qudits = {self.num_qudits} < 1")
        
        if self.num_lvs < 2:
            raise ValueError(f"self.num_lvs = {self.num_lvs} < 2")
        
        if unitary_rep.shape != (num_lvs ** num_qudits, num_lvs ** num_qudits):
            raise ValueError(f"unitary_rep.shape = {unitary_rep.shape}\n\
                            != (num_lvs ** num_qudits, num_lvs ** num_qudits) = \
                             {(num_lvs ** num_qudits, num_lvs ** num_qudits)}")
        #------------------------------------------------------------------------------------------
        
        # Verifies unitary_rep is unitary; U U^{\dagger} = U^{\dagger} U = I, for all U(d^n).
        I = np.eye(num_lvs ** num_qudits)
        UU_inv = np.round(unitary_rep @ self.inv_unitary_rep, decimals = 10)
        U_invU = np.round(self.inv_unitary_rep @ unitary_rep, decimals = 10)
        if not (np.array_equal(UU_inv, I) and np.array_equal(U_invU, I)):
            raise ValueError(f"I ≠ UU_inv =\n{UU_inv}")

    def InvUnitaryRep(self) -> np.ndarray:
        """
        Calculates the inverse of the gate's unitary matrix (its conjugate transpose).

        Returns:
            (np.ndarray): The inverse of the unitary matrix.
        """
        return np.conj(self.unitary_rep.T)
    
class GateSet():
    """
    Represents a collection of gates.
    """
    def __init__(self, gate_set: set[Gate]):
        """
        Args:
            gate_set (set[Gate]): A set containing Gate objects.
        """
        self.gate_set = gate_set
    
        #self.is_universal = self.UnivCheck()

    def UnivCheck(self) -> bool:
        """
        Checks if the gate set is universal. (Not yet implemented).
        TODO: Implement!

        Returns:
            (bool): True if the set is universal, False otherwise.
        """
        print("Not Implemented")
        return False
    
    def NumQuditsDict(self) -> dict[int, list[Gate]]:
        """
        Organizes the gates in the set into a dictionary based on the number of qudits they act on.

        Returns:
            num_qudits_dict (dict[int, list[Gate]]): A dictionary where keys are the number of 
                qudits (int) and values are lists of Gates that act on that many qudits.
        """
        num_qudits_dict = {}
        for gate in self.gate_set:
            num_qudits = gate.num_qudits
            if num_qudits not in num_qudits_dict.keys():
                num_qudits_dict[num_qudits] = [gate]
            else:
                num_qudits_dict[num_qudits].append(gate)
        return num_qudits_dict
    
    def GateIDDict(self) -> dict[str, Gate]:
        """
        Creates a dictionary mapping gate IDs to their corresponding Gate objects.

        Returns:
            gate_id_dict (dict[str, Gate]): A dictionary where keys are the gate_id (str)
                               and values are the corresponding Gate objects.
        """
        gate_id_dict = {}
        for gate in self.gate_set:
            gate_id = gate.gate_id
            gate_id_dict[gate_id] = gate
        return gate_id_dict

# Gates #######################################################################################





# End of File #################################################################################