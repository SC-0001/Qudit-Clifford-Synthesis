"""
Main script for training the Qudit Clifford Synthesis agent using curriculum learning.
This script loads hyperparameters and initiates the training process. 
"""

import os
import torch
from itertools import combinations

from qudit_clifford_synthesis.CliffSyn.Curriculum import TrainCurriculum
from qudit_clifford_synthesis.CliffSyn.HyperParam import LoadHyperParam

# Control main process thread usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.set_num_threads(1)

# Default training parameters. Used if no existing hyperparameter file is found.
default_params = {

    # Fixed training parameters
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 1,
    "gamma": 0.95,
    "gae_lambda": 0.95,

    # Adaptive parameters ---------------------------------------------------------------------
    "initial_learning_rate": 2e-4,
    "max_learning_rate": 5e-4,
    "min_learning_rate": 1e-5,
    "learning_rate_parameter": 0.45, #learning_rate ≈ 2e-4 at success_rate = 0.8
    
    "initial_clip_range": 0.1,
    "max_clip_range": 0.2,
    "min_clip_range": 0.05,
    "clip_range_parameter": 0.7, #clip_range ≈ 0.1 at success_rate = 0.8
    
    "initial_ent_coef": 0.01,
    "max_ent_coef": 0.02,
    "min_ent_coef": 0.005,
    "ent_coef_parameter": 1,
    #-----------------------------------------------------------------------------------------

    # CNN parameters
    "conv_channels": 64,
    "fc1_units": 512,
    "features_dim": 256,
    "embedding_dim": 2, #round(2/3*num_lvs)

    # Environment parameter
    "diff_ratio": 0.5
}

if __name__ == '__main__':
    log_dir = './Training_Data/3Lv_3L_Ongoing'
    
    # Load hyperparameters from the specified log directory or use defaults.
    params = LoadHyperParam(log_dir = log_dir, default_params = default_params)
    
    # Execute the training curriculum with specified settings.
    # Monitor training progress with [tensorboard --logdir './Training_Data/3Lv_3L_Ongoing']
    TrainCurriculum(

        # Core environment parameters
        num_lvs = 3, 
        num_qudits = 3, 
        coupling_map = [[0, 1], [1, 2]], 

        # Environment reward parameters
        match_reward = 200, 
        depth_reward_factor = 10, 
        inc_reward_option = 1, 
        inc_reward_factor = 30,

        # Training parameters
        num_envs = 8, 
        eval_ratio = 0.25, 
        params = params, 
        log_dir = log_dir, 
        max_difficulty = 100, 
        difficulty_threshold = 100, 
        
        # Callback parameters
        eval_freq = 1000, 
        n_eval_episodes = 20, 
        succ_window_size = 30, 
        hyp_window_size = 1, 
        success_threshold = 0.95,

        # Conditional environment parameters
        modes = ["training"], 
        debug_difficulties = [7],
        skip_to_difficulty = 1, 
        max_gates = 100)