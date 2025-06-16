"""
Main script for comparing the trained RL agent against other synthesis methods, 
such as LEAP from BQSKit. None of the comparisons will be 100% fair because I'm not aware
of any model / alg that is built with the objective of qudit clifford circuit synthesis.

For instance, LEAP doesn't seem to work well with a gate set without parametrized gates.
(TODO: apply the improved version of the Solovay-Kitaev algorithm outlined in "Breaking 
the cubic barrier in the Solovay-Kitaev algorithm" on top of LEAP for a complete synthesis
routine) Therefore, we just compare CSUM and single-qudit gate counts, depth, and runtime.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np
import numpy.linalg as LA
import json

from qudit_clifford_synthesis.CliffSyn.Environment import CliffordSynthesisEnv
from qudit_clifford_synthesis.Evaluation.SB3 import EvaluateSB3, CircCountGates, CircToUnitary
from qudit_clifford_synthesis.Evaluation.LEAP import EvaluateLEAP

# Useful Functions ############################################################################

def TargetInfo(env: CliffordSynthesisEnv) -> dict[str, Any]:
    """
    Extracts key information about the target circuit from the environment.

    Args:
        env (CliffordSynthesisEnv): The environment instance containing the target circuit.

    Returns:
        out_dict (dict[str, Any]): A dictionary containing the target's depth,
            unitary matrix, and gate counts.
    """
    target_circuit = env.initial_target_circuit
    
    target_circuit_unitary = CircToUnitary(circuit = target_circuit)
    target_gate_count = CircCountGates(circuit = target_circuit)
    out_dict = {
        "depth": env.target_depth,
        "unitary": target_circuit_unitary,
    }
    out_dict.update(target_gate_count)
    return out_dict

def SimilarityMetric(appx_unitary: np.ndarray, target_unitary: np.ndarray) -> float:
    """
    Calculates the similarity between two unitary matrices. For now, uses the absolute value
    of the Hilbert-Schmidt inner product.

    Args:
        appx_unitary (np.ndarray): The synthesized, (approximate) unitary matrix.
        
        target_unitary (np.ndarray): The target unitary matrix.

    Returns:
        similarity (float): The calculated similarity metric.
    """
    assert appx_unitary.dtype == np.complex128
    assert target_unitary.dtype == np.complex128

    appx_unitary_dag = np.conj(appx_unitary.T)
    HS_inner_prod = LA.trace(appx_unitary_dag @ target_unitary)
    return float(np.abs(HS_inner_prod))

# Main ########################################################################################

def SingleExperiment(
        num_lvs: int, 
        bi_coupling_map: list[list[int]], 
        difficulty: int, 
        max_gates: int,
        sb3_model_dir: str,
        max_layer: int
        ) -> Dict:
    """
    Evaluates the performance of the SB3 agent and LEAP for one randomly generated
    target circuit.

    Args:
        num_lvs (int): Number of qudit energy levels.
        
        bi_coupling_map (list[list[int]]): The hardware coupling graph.
        
        difficulty (int): The number of gates in the target circuit.
        
        max_gates (int): The maximum number of gates allowed for synthesis.
        
        sb3_model_dir (str): Path to the saved SB3 model.
        
        max_layer (int): The maximum number of layers for LEAP.

    Returns:
        out_dict (Dict): Contains performance metrics for both methods.
    """

    # Environment setup
    env_config = {
    "num_lvs": num_lvs,
    "num_qudits": np.max(bi_coupling_map) + 1,
    "coupling_map": bi_coupling_map,
    
    "match_reward": 200,
    "depth_reward_factor": 10,
    "inc_reward_factor": 30,
    "inc_reward_option": 3,
    
    "curriculum_difficulty": difficulty,
    "diff_ratio": None,
    "difficulty_threshold": 1e10,
    
    "max_gates": max_gates,
    "modes": ["prediction"]
    }
    env = CliffordSynthesisEnv(env_config)
    init_obs, _ = env.reset()
    target_info_dict = TargetInfo(env = env)
    
    # Run evaluations
    leap_results = EvaluateLEAP(
        num_lvs = num_lvs, 
        bi_coupling_map = bi_coupling_map, 
        target_info = target_info_dict,
        SimilarityMetric = SimilarityMetric,
        max_layer = max_layer
    )
    sb3_results = EvaluateSB3(
        model_directory = sb3_model_dir,
        env = env,
        init_obs = init_obs,
        target_info = target_info_dict,
        SimilarityMetric = SimilarityMetric
    )

    env.reset()
    out_dict = {**sb3_results, **leap_results}

    return out_dict

def MultiExperiment(
        num_lvs: int, 
        bi_coupling_map: list[list[int]], 
        max_difficulty: int, 
        max_gates: int, 
        trials_per_difficulty: int,
        sb3_model_dir: str,
        max_layer: int
        ) -> Dict:
    """
    Runs "SingleExperiment" multiple times for each difficulty level to gather statistics.

    Args:
        trials_per_difficulty (int): The number of random circuits to test at each difficulty.
        
        num_lvs (int): Number of qudit energy levels.
        
        bi_coupling_map (list[list[int]]): The hardware coupling graph.
        
        max_difficulty (int): The maximum number of gates in the target circuit for which
            we evaluate the two methods.
        
        max_gates (int): The maximum number of gates allowed for synthesis.
        
        sb3_model_dir (str): Path to the saved SB3 model.
        
        max_layer (int): The maximum number of layers for LEAP.

    Returns:
        final_dict (Dict): Contains the average and standard error of evaluation metrics, 
        for each difficulty level.
    """

    # Initializing result dictionary
    final_dict = {"difficulty": []}
    metric_names = ["runtime", "similarity_metric", "depth_saved", "1q_gates_saved", 
                    "csum_gates_saved"]
    method_names = ["sb3", "leap"]
    for metric_name in metric_names:
        for method_name in method_names:
            final_dict[method_name + "_" + metric_name + "_avg"] = []
            final_dict[method_name + "_" + metric_name + "_err"] = []
    
    # Looping over difficulties
    for difficulty in range(1, max_difficulty+1):

        # Initializing dictionary for current difficulty
        difficulty_dict = {}
        for metric_name in metric_names:
            for method_name in method_names:
                difficulty_dict[method_name + "_" + metric_name] = []
        
        # Collecting data from multiple trials of the same difficulty
        for trial in range(1, trials_per_difficulty+1):
            result = SingleExperiment(
                num_lvs = num_lvs, 
                bi_coupling_map = bi_coupling_map, 
                difficulty = difficulty, 
                max_gates = max_gates,
                sb3_model_dir = sb3_model_dir,
                max_layer = max_layer
            )
            for key, value in result.items():
                difficulty_dict[key].append(value)
        
            print(f"Trial {trial} Completed")
        
        final_dict["difficulty"].append(difficulty)
        for key, value in difficulty_dict.items():
            final_dict[key + "_avg"].append(np.average(value))
            final_dict[key + "_err"].append(np.std(value))
        
        print(f"Difficulty {difficulty} Completed")
    
    return final_dict

def PlotMetrics(data_dict: Dict, save_path: str):
    """
    Creates comparison plots showing runtime, success rate, and depth savings metrics.

    Args:
        data_dict (Dict): The results from "MultiExperiment".
        
        save_path (str): The file path to save the plot.
    """
    difficulties = data_dict["difficulty"]
    
    # 2 x 3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

    def PlotHelper(ax, metric_name, title, ylabel):
        """
        Plots a comparison of SB3 and LEAP metrics with error bars.

        Args:
            ax (matplotlib.axes.Axes): The axes object to plot on.
            
            metric_name (str): The name of the metric to plot from data_dict.
            
            title (str): The title for the plot.
            
            ylabel (str): The label for the y-axis
        """
        ax.errorbar(difficulties, 
                    data_dict[f"sb3_{metric_name}_avg"], 
                    yerr = data_dict[f"sb3_{metric_name}_err"], 
                    label="SB3", alpha=0.7, marker='s')
        ax.errorbar(difficulties, 
                    data_dict[f"leap_{metric_name}_avg"], 
                    yerr = data_dict[f"leap_{metric_name}_err"], 
                    label="LEAP", alpha=0.7, marker='s')
        ax.set_xlabel("Difficulty")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.autoscale()

    PlotHelper(ax = ax1, 
               metric_name = "runtime", 
               title = "Runtime Comparison", 
               ylabel = "Runtime (seconds)")
    ax1.set_yscale("log")

    PlotHelper(ax = ax2, 
               metric_name = "similarity_metric", 
               title = "|Hilbert-Schmidt Inner Product| Comparison", 
               ylabel = "|Hilbert-Schmidt Inner Product|")
    
    PlotHelper(ax = ax3, 
               metric_name = "depth_saved", 
               title = "Depth Savings Comparison", 
               ylabel = "Depth Saved")
    
    PlotHelper(ax = ax4, 
               metric_name = "1q_gates_saved", 
               title = "1-Qubit Gate Savings Comparison", 
               ylabel = "1Q Gates Saved")
    
    PlotHelper(ax = ax5, 
               metric_name = "csum_gates_saved", 
               title = "CSUM Gate Savings Comparison", 
               ylabel = "CSUM Gates Saved")

    # Hide the unused 6th subplot
    ax6.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# End of File #################################################################################

