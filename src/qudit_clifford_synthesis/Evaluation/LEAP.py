"""
Provides a wrapper function to evaluate the LEAP synthesis algorithm from BQSKit.
Ref: https://dl.acm.org/doi/10.1145/3548693
"""

import math
import time
from typing import Any, Callable

from bqskit.passes.synthesis import LEAPSynthesisPass
from bqskit.passes import SetModelPass
from bqskit.passes.search.generators import WideLayerGenerator, SingleQuditLayerGenerator
from bqskit.compiler import Compiler, MachineModel
from bqskit.ir import Circuit
from bqskit.ir.gates.constant import CSUMGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate

from qudit_clifford_synthesis.Essentials.CliffGates import CliffGateSet
from qudit_clifford_synthesis.Essentials.QuditCirc import QuditCircuit

# Useful Functions ############################################################################

def HistoryToUnitary(
        num_qudits: int, 
        num_lvs: int, 
        gate_placement_history: list[dict[str, Any]],
        coupling_map: list[list[int]] | None = None,    
        ):
    """
    Used for debugging purposes.
    Converts the "gate_placement_history" attribute of a QuditCircuit instance to its
    corresponding unitary representation.
    
    num_qudits (int): Number of energy levels of the QuditCircuit instance
    
    num_lvs (int): Number of qudits in the QuditCircuit instance
    
    gate_placement_history (list[dict[str, Any]]): "gate_placement_history" attribute of 
        the QuditCircuit instance 
    
    coupling_map (list[list[int]] | None): Coupling graph of the QuditCircuit instance
    """
    dummy_circ = QuditCircuit(
        num_qudits = num_qudits, 
        num_lvs = num_lvs, 
        coupling_map = coupling_map, 
        gate_set = CliffGateSet(num_lvs),
        rep_type = "Unitary"
    )
    for gate_dict in gate_placement_history:
        dummy_circ.ApplyGate(**gate_dict)
    return dummy_circ.rep

# Main ########################################################################################

def EvaluateLEAP(
        num_lvs: int, 
        bi_coupling_map: list[list[int]], 
        target_info: dict[str, Any],
        SimilarityMetric: Callable,
        max_layer: int
    ) -> dict[str, Any]:
    """
    Synthesizes a target unitary using LEAP and evaluates its performance.
    Ref: https://dl.acm.org/doi/10.1145/3548693

    Args:
        num_lvs (int): Number of qudit energy levels.

        bi_coupling_map (list[list[int]]): The hardware coupling graph.

        target_info (dict[str, Any]): A dictionary containing the target unitary,
            its depth, and original gate counts.

        SimilarityMetric (Callable): A function to compute the similarity between the
            synthesized and target unitaries.

        max_layer (int): The maximum number of layers for the LEAP algorithm to search.

    Returns:
        out_dict (dict[str, Any]): A dictionary of performance metrics for the LEAP synthesis.
    """
    target_unitary = target_info["unitary"]
    num_qudits = int(math.log(target_unitary.shape[0], num_lvs))
    radixes = [num_lvs] * num_qudits
    
    # Defining a machine model to enforce the provided coupling map
    coupling_graph = [tuple(elem) for elem in bi_coupling_map]
    model = MachineModel(
        num_qudits = num_qudits, 
        radixes = radixes, 
        coupling_graph = coupling_graph, #type: ignore
    )

    init_circuit = Circuit(num_qudits = num_qudits, radixes = radixes)
    input_circuit = init_circuit.from_unitary(utry = target_unitary)

    # Due to LEAP (and BQSKit's default compiler) having difficulty with a gateset 
    # consisting only of constant (non-parametrized) gates, we decompose the 
    # target in terms of CSUM and general single-qudit gates.
    general_layer_generator = WideLayerGenerator(
        multi_qudit_gates = set([CSUMGate(radix = num_lvs)]), 
        single_qudit_gate = VariableUnitaryGate(num_qudits = 1, radixes = [num_lvs])
    )
    general_pass = [
        SetModelPass(model), 
        LEAPSynthesisPass(layer_generator = general_layer_generator, max_layer = max_layer)
    ]  
    # Compile and time
    with Compiler() as compiler:
        start_time = time.time()
        circuit = compiler.compile(input_circuit, general_pass)
        end_time = time.time()

    # Calculating and returning evaluation metrics
    similarity_metric = SimilarityMetric(
        appx_unitary = circuit.get_unitary().numpy,
        target_unitary = target_info["unitary"]
    )
    out_dict = {
    "leap_runtime": end_time - start_time,
    "leap_depth_saved": target_info["depth"] - circuit.depth, 
    "leap_1q_gates_saved": target_info["1q_gate_count"] - circuit.count(VariableUnitaryGate(
        num_qudits = 1, radixes = [num_lvs])),
    "leap_csum_gates_saved": target_info['csum_gate_count'] - circuit.count(CSUMGate(
        radix = num_lvs)),
    "leap_similarity_metric": similarity_metric
    }
    return out_dict

# End of File #################################################################################