"""
Script for evaluating a pre-trained Clifford synthesis agent.

The model is tested on randomly generated Clifford circuits of varying difficulty. 
We collect and plot the following metrics: reduction in circuit depth, runtime, 
the Hilbert-Schmidt inner product between the target and synthesized circuits, 
and the reduction in the number of CSUM and single-qudit gates.

The agent's evaluation metrics are compared with that of other synthesis methods, such 
as LEAP from BQSKit. However, a direct comparison is not 100% fair, as there are no models
(to my very limited knowledge) built specifically for qudit Clifford circuit synthesis. 
For instance, LEAP seems to have issues with gate sets without parameterized gates. 
Therefore, the comparison is limited to single-qudit gate counts rather than counts of
specific gates (e.g., H, P, M_a).

(TODO: apply the improved version of the Solovay-Kitaev algorithm outlined in "Breaking 
the cubic barrier in the Solovay-Kitaev algorithm" on top of LEAP for a complete synthesis
routine)
"""

from qudit_clifford_synthesis.Evaluation.MainEvaluation import MultiExperiment, PlotMetrics

if __name__ == "__main__":
    """
    Compare our SB3 model with BQSKit's LEAP.
    Independent Variables: 
        difficulty: 1 ~ 6
        num_lvs: 2, 3, 4
        num_qudits: 2, 3, 4
    Dependent Variables:
        depth, runtime, Hilbert-Schmidt inner product, 
        no. of single-qudit gates saved, no. of CSUM gates saved
    """

    # Run the evaluation function.
    data_dict = MultiExperiment(
        num_lvs = 3, 
        bi_coupling_map = [[0, 1], [1, 2]], 
        max_difficulty = 10, 
        max_gates = 20, 
        max_layer = 20,
        trials_per_difficulty = 20, 
        sb3_model_dir = "./Evaluation_Data/3Lv_3L_Diff10_model.zip")

    # Generate and save plots comparing the two models' performance
    PlotMetrics(
        data_dict = data_dict,  
        save_path = "./Evaluation_Data/LEAP vs SB3 Clifford Synthesis Comparison.png")