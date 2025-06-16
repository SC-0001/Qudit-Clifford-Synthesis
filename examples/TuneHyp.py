"""
Script for hyperparameter tuning of the Clifford synthesis agent using Optuna.

This script defines an objective function that trains an agent for a fixed number
of steps on a specific difficulty level and returns a performance metric. Optuna
is then used to run multiple trials to find the best hyperparameters.
"""

from qudit_clifford_synthesis.CliffSyn.HyperParam import SetDifficultyObjectiveMaker, TuneHyperParam

if __name__ == "__main__":

    # All user inputs -------------------------------------------------------------------------
    
    num_hypparam_trials = 100
    common_fixed_params = dict(

        # Core environment parameters
        num_lvs = 3, 
        num_qudits = 3,
        coupling_map = [[0, 1], [1, 2]], 

        # Training parameters
        log_dir = './Training_Data/3Lv_3L_Hyp',
        num_envs = 8,
        eval_ratio = 0.25,
        
        # Callback parameters
        eval_freq = 100,
        n_eval_episodes = 10, 
        hyp_window_size = 1,
        
        # Environment reward parameters
        match_reward = 200,
        depth_reward_factor = 10,
        inc_reward_option = 1,
        inc_reward_factor = 30,

        # Conditional environment parameters
        difficulty_threshold = 100,
        max_gates = 100
    )

    # Parameters for a fixed-difficulty objective
    set_difficulty_fixed_params = dict(
        difficulty = 1,
        total_timesteps = 1e6
    ) | common_fixed_params

    # Parameters for a curriculum objective (currently unusued)
    curriculum_fixed_params = dict(
        skip_to_difficulty = 1,
        max_difficulty = 6,
        succ_window_size = 30,
        success_threshold = 0.95
    ) | common_fixed_params
    #------------------------------------------------------------------------------------------
    
    # Creating the objective function
    ObjectiveFunc = SetDifficultyObjectiveMaker(set_difficulty_fixed_params)

    # Start the hyperparameter tuning process.
    TuneHyperParam(num_hypparam_trials = num_hypparam_trials, 
                   log_dir = common_fixed_params["log_dir"],  # type: ignore
                   ObjectiveFunc = ObjectiveFunc)