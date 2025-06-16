"""
Provides a wrapper function to evaluate our trained Stable-Baselines3 agent.
"""

from typing import Callable, Any
from stable_baselines3 import PPO
import torch
import time
import copy

from qudit_clifford_synthesis.CliffSyn.Environment import CliffordSynthesisEnv
from qudit_clifford_synthesis.Essentials.QuditCirc import QuditCircuit
from qudit_clifford_synthesis.Essentials.CliffGates import CliffGateSet

# Useful Functions ############################################################################

def CircToUnitary(circuit: QuditCircuit):
    """
    Converts a QuditCircuit object to its unitary matrix representation.

    Args:
        circuit (QuditCircuit): The input circuit.

    Returns:
        unitary_rep (np.ndarray): The unitary matrix of the circuit.
    """
    num_qudits = circuit.num_qudits
    num_lvs = circuit.num_lvs
    coupling_map = circuit.coupling_map
    gate_placement_history = circuit.gate_placement_history
    
    # Create a temporary circuit with unitary representation
    dummy_circ = QuditCircuit(
        num_qudits = num_qudits, 
        num_lvs = num_lvs, 
        coupling_map = coupling_map, 
        gate_set = CliffGateSet(num_lvs),
        rep_type = "Unitary"
    )
    # Replay the gate history to build-up the unitary
    for gate_dict in gate_placement_history:
        dummy_circ.ApplyGate(**gate_dict)
    
    return dummy_circ.rep

def CircCountGates(circuit: QuditCircuit) -> dict[str, int]:
    """
    Counts the number of one-qudit and two-qudit (CSUM) gates in a circuit.

    Args:
        circuit (QuditCircuit): The input circuit.

    Returns:
        count_dict (dict[str, int]): Contains '1q_gate_count' and 'csum_gate_count'.
    """
    count_dict = {'1q_gate_count': 0, 'csum_gate_count': 0}
    for gate_dict in circuit.gate_placement_history:
        if gate_dict["gate_id"] == "SUM":
            count_dict['csum_gate_count'] += 1
        elif \
            (gate_dict["gate_id"] == "F") \
            or (gate_dict["gate_id"] == "P") \
            or (gate_dict["gate_id"][0] == "M"): #type: ignore
            count_dict['1q_gate_count'] += 1
        else:
            raise ValueError("Weird Gate")
    return count_dict

# Main ########################################################################################

def EvaluateSB3(
        model_directory: str, 
        env: CliffordSynthesisEnv, 
        init_obs: Any, 
        target_info: dict[str, Any],
        SimilarityMetric: Callable
    ) -> dict[str, Any]:
    """
    Synthesizes a target circuit using a trained SB3 agent and evaluates its performance.

    Args:
        model_directory (str): Path to the saved SB3 PPO model.

        env (CliffordSynthesisEnv): The environment instance, already reset to the
            desired target circuit.

        init_obs (Any): The initial observation from the environment reset.

        target_info (dict[str, Any]): A dictionary containing details about the target circuit.

        SimilarityMetric (Callable): A function to compute the similarity between the
            synthesized and target unitaries.

    Returns:
        out_dict (dict[str, Any]): A dictionary of performance metrics for the SB3 synthesis.
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(model_directory, env = env, device = device)
    
    # Run inference loop
    done = False
    obs = copy.deepcopy(init_obs)
    start_time = time.time()
    while not done:
        action, _ = model.predict(obs, deterministic = True)
        obs, _, done, _, info = env.step(action)
    end_time = time.time()

    # Calculate and return metrics ------------------------------------------------------------
    current_circuit = env.current_circuit
    current_circuit_gate_counts = CircCountGates(circuit = current_circuit)

    similarity_metric = SimilarityMetric(
        appx_unitary = CircToUnitary(circuit = current_circuit),
        target_unitary = target_info["unitary"]
    )
    
    out_dict = {
        "sb3_runtime": end_time - start_time,
        "sb3_depth_saved": target_info["depth"] - int(current_circuit.Depth()),
        #"sb3_success": info["is_success"], # type: ignore
        "sb3_1q_gates_saved": target_info["1q_gate_count"] - current_circuit_gate_counts["1q_gate_count"],
        "sb3_csum_gates_saved": target_info["csum_gate_count"] - current_circuit_gate_counts["csum_gate_count"],
        "sb3_similarity_metric": similarity_metric
    }
    # -----------------------------------------------------------------------------------------
    return out_dict

# End of File #################################################################################