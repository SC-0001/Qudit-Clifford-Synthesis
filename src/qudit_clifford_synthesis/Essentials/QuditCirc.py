"""
Defines the "QuditCircuit" class, which simulates a quantum circuit composed of 
homogeneous qudits (all qudits have the same energy levels). Restrictions in couplings
between states (within a single qudit) is currently not supported. (e.g. 0 <-> 1 and 1 <-> 2
is allowed but not 0 <-> 2)
"""

from itertools import combinations
from typing import Union
import numpy as np

from qudit_clifford_synthesis.Essentials.CliffGates import EmbedSymGate, CliffGateSet
from qudit_clifford_synthesis.Essentials.Gates import EmbedUnitaryGate, GateSet

# Main ########################################################################################

class QuditCircuit():
    """
        Simulates a circuit that consists of homogeneous qudits.
        Tracks the history of applied gates.
        """
    def __init__(
            self, num_qudits: int, 
            num_lvs: int, 
            coupling_map: list[list[int]] | None = None, 
            gate_set: GateSet | None = None,
            rep_type: str = "Symplectic"):
        """
        Initializes the QuditCircuit.

        Args:
            num_qudits (int): The number of qudits.

            num_lvs (int): The number of energy levels in ALL qudits.

            coupling_map (list[list[int]] | None): A list of allowed two-qudit interactions.
                If None, a fully connected topology is assumed.

            gate_set (GateSet | None): The set of allowed gates. If None, a default
                Clifford gate set is used.

            rep_type (str): The internal representation to use. Can be "Symplectic"
                for efficient Clifford simulation or "Unitary" for general circuits.
        """
        self.num_qudits = num_qudits
        self.num_lvs = num_lvs
        self.rep_type = rep_type
        
        # Assumes a fully connected topology if a coupling map isn't provided.
        if coupling_map is None:
            self.coupling_map = [list(elem) for elem in combinations(range(self.num_qudits), 2)]
        else:
            self.coupling_map = coupling_map

        # Uses the Clifford gate set if a gate set isn't provided.
        if gate_set is None:
            self.gate_set = CliffGateSet(num_lvs)
        else:
            self.gate_set = gate_set
        
        # Makes searching for the matrix representations of gates more efficient.
        self._id_rep_dict = self.gate_set.GateIDDict()

        # Configures functions based on representation type. We want to minimize 
        # the need to check self.rep_type elsewhere so it's easy to expand
        # the representation types supported.
        if self.rep_type == "Symplectic":
            self._RepFunc = lambda gate: gate.sym_rep
            self._InvRepFunc = lambda gate: gate.inv_sym_rep
            self._EmbedFunc = EmbedSymGate
            self._init_rep = np.eye(2 * self.num_qudits, dtype = np.uint32)
        elif self.rep_type == "Unitary":
            self._RepFunc = lambda gate: gate.unitary_rep
            self._InvRepFunc = lambda gate: gate.inv_unitary_rep
            self._EmbedFunc = EmbedUnitaryGate
            self._init_rep = np.eye(self.num_lvs ** self.num_qudits, dtype = np.complex128)
        else:
            raise NotImplementedError

        self.Reset()

    def Reset(self):
        """
        Resets the circuit to an empty state (identity).
        """
        self.gate_count = 0
        self.gate_placements = np.zeros((self.num_qudits,), dtype = np.int32)
        self.gate_placement_history = []
        self.rep = self._init_rep

    def ApplyGate(
            self, gate_id: str, 
            target: int, 
            control: int | None = None, 
            reverse_order: bool = False, 
            inverse_gate: bool = False):
        """
        Applies a gate to the circuit, updating its current representaiton and history.

        Args:
            gate_id (str): The ID of the gate to apply.

            target (int): The index of the target qudit.

            control (int | None): The index of the control qudit (for two-qudit gates).

            reverse_order (bool): If True, multiplies the new gate on the right (rep @ gate)
                instead of the left (gate @ rep). This is useful when we train our
                synthesis model.

            inverse_gate (bool): If True, applies the inverse of the specified gate.
        """
        self.gate_count += 1

        # Updates circuit depth
        if control is None:
            self.gate_placements[target] += 1
        else:
            control_depth = self.gate_placements[control]
            target_depth = self.gate_placements[target]
            max_depth = max(control_depth, target_depth)
            
            self.gate_placements[control] = max_depth + 1
            self.gate_placements[target] = max_depth + 1
        
        # Log the gate
        gate_dict = {"gate_id": gate_id, "control": control, "target" : target, 
                     "reverse_order": reverse_order, "inverse_gate": inverse_gate}
        self.gate_placement_history.append(gate_dict)
        
        # Makes sure the gate doesn't violate hardware constraints
        if control is None:
            assert target in range(self.num_qudits)
        else:
            assert ([control, target] in self.coupling_map) or \
            ([target, control] in self.coupling_map)

        # Update the circuit's current representation -----------------------------------------
        gate_rep = self._IdToRep(gate_id = gate_id, target = target, control = control, 
                                 inverse_gate = inverse_gate) 

        if reverse_order:
            self.rep = self.rep @ gate_rep
        else:
            self.rep = gate_rep @ self.rep

        if self.rep_type == "Symplectic":  
            self.rep = self.rep % self.num_lvs # type: ignore
        #--------------------------------------------------------------------------------------

    def Depth(self) -> int:
        """
        Calculates the depth of the circuit.

        Returns:
            depth (int): Depth of the circuit.
        """
        if len(self.gate_placements) == 0:
            depth = 0
        else:
            depth = int(np.max(self.gate_placements))
        return depth

    def _IdToRep(self, gate_id: str, target: int, control: Union[int, None] = None, 
                 inverse_gate: bool = False) -> np.ndarray:
        """
        Converts a gate instruction into its embedded matrix representation.

        Args:
            gate_id (str): The ID of the gate to apply.

            target (int): The index of the target qudit.

            control (int | None): The index of the control qudit (for two-qudit gates).

            inverse_gate (bool): If True, applies the inverse of the specified gate.

        Returns:
            gate_matrix (np.ndarray): The embedded matrix representation of the gate.
        """
        gate = self._id_rep_dict[gate_id]

        if inverse_gate:
            gate_rep = self._InvRepFunc(gate)
        else:
            gate_rep = self._RepFunc(gate)

        if gate.num_qudits == 1:
            assert control is None
            return self._EmbedFunc(self.num_qudits, gate_rep, [target])
        elif gate.num_qudits == 2:
            assert control is not None
            return self._EmbedFunc(self.num_qudits, gate_rep, [target, control])
        else:
            raise NotImplementedError("Gates involving more than 2 qudits aren't supported")
        
# End of File #################################################################################