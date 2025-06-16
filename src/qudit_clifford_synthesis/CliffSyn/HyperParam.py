"""
Uses Optuna to tune all training and feature extractor hyperparameters
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.INFO)

import torch
import numpy as np
import glob
import time
import json
from typing import Any, Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from qudit_clifford_synthesis.CliffSyn.CNN import CustomCliffordCNN
from qudit_clifford_synthesis.CliffSyn.Curriculum import PartitionEnvs, SuccessStopCallback, AdaptiveHyperparameterCallback

# Main #######################################################################################

def SetDifficultyObjectiveMaker(fixed_params: dict[str, Any]) -> Callable:
    """
    A function that creates an Optuna objective function for a fixed difficulty.

    Args:
        fixed_params (dict[str, Any]): A dictionary of parameters that are kept constant
            during the optimization trials. For details, ref annotations on 
            CliffSyn.Curriculum.TrainCurriculum.

    Returns:
        SetDifficultyObjective (Callable): The objective function to be passed to Optuna's study.
    """
    def SetDifficultyObjective(trial: optuna.Trial) -> float:
        """
        Defines one trial of hyperparameter optimization for a fixed difficulty level.
        Requires a pre-trained model for difficulty > 1.

        Args:
            trial (optuna.Trial): An Optuna trial object used to suggest hyperparameter values.

        Returns:
            tune_metric (float): The performance metric to be maximized.
        """
        # Set Device
        if torch.cuda.is_available():
            torch_device = "cuda"
        # elif torch.backends.mps.is_available():
            # torch_device = "mps"
        else:
            torch_device = "cpu"
        device = torch.device(torch_device)

        # Unpack fixed parameters -----------------------------------------------------------

        modes: list[str] = ["training"]
        
        log_dir: str = fixed_params["log_dir"]

        num_lvs: int = fixed_params["num_lvs"]
        num_qudits: int = fixed_params["num_qudits"]
        coupling_map: list[list[int]] = fixed_params["coupling_map"]
        
        difficulty: int = fixed_params["difficulty"]
        difficulty_threshold: int = fixed_params["difficulty_threshold"]
        max_gates: int = fixed_params["max_gates"]
        num_envs: int = fixed_params["num_envs"]
        
        eval_ratio: float = fixed_params["eval_ratio"]
        eval_freq: int = fixed_params["eval_freq"]
        n_eval_episodes: int = fixed_params["n_eval_episodes"]
        hyp_window_size: int = fixed_params["hyp_window_size"]
        
        match_reward: float = fixed_params["match_reward"]
        depth_reward_factor: float = fixed_params["depth_reward_factor"]
        inc_reward_option: int = fixed_params["inc_reward_option"]
        inc_reward_factor: float = fixed_params["inc_reward_factor"]

        total_timesteps: int = fixed_params["total_timesteps"]

        # Hyperparameters to optimize -------------------------------------------------------

        n_steps = trial.suggest_int("n_steps", 512, 2048, log = True)
        batch_size = trial.suggest_int("batch_size", 128, 512, log = True)
        n_epochs = trial.suggest_int("n_epochs", 3, 10)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.9999)

        initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-3, log=True)
        learning_rate_parameter = trial.suggest_float("learning_rate_parameter", 0.01, 1)
        max_learning_rate = trial.suggest_float("max_learning_rate", 1e-4, 1e-3, log=True)
        min_learning_rate = trial.suggest_float("min_learning_rate", 1e-6, 1e-4, log=True)
        if min_learning_rate > max_learning_rate:
            min_learning_rate, max_learning_rate = max_learning_rate, min_learning_rate
        
        initial_clip_range = trial.suggest_float("initial_clip_range", 0.05, 0.3)
        clip_range_parameter = trial.suggest_float("clip_range_parameter", 0.01, 1)
        max_clip_range = trial.suggest_float("max_clip_range", 0.1, 0.5)
        min_clip_range = trial.suggest_float("min_clip_range", 0.01, 0.1)
        if min_clip_range > max_clip_range:
            min_clip_range,  max_clip_range = max_clip_range, min_clip_range
        
        initial_ent_coef = trial.suggest_float("initial_ent_coef", 0.0, 0.02)
        ent_coef_parameter = trial.suggest_float("ent_coef_parameter", 0.01, 1)
        max_ent_coef = trial.suggest_float("max_ent_coef", 0.1, 0.3)
        min_ent_coef = trial.suggest_float("min_ent_coef", 0, 0.1)
        if min_ent_coef > max_ent_coef:
            min_ent_coef, max_ent_coef = max_ent_coef, min_ent_coef

        features_dim = trial.suggest_int("features_dim", 256, 1024)
        conv_channels = trial.suggest_int("conv_channels", 64, 256)
        fc1_units = trial.suggest_int("fc1_units", 512, 1024)
        embedding_dim = trial.suggest_int("embedding_dim", 1, num_lvs)

        #------------------------------------------------------------------------------------

        # Defining policy kwargs with optimized CNN architecture
        policy_kwargs = dict(
            features_extractor_class = CustomCliffordCNN,
            features_extractor_kwargs = dict(
                features_dim = features_dim,
                conv_channels = conv_channels,
                fc1_units = fc1_units,
                embedding_dim = embedding_dim,
                num_lvs = num_lvs
            ),
            normalize_images=False
        )

        # Creating training and evaluation environments
        env_config = dict(num_lvs = num_lvs, num_qudits = num_qudits, max_gates = max_gates, 
                          curriculum_difficulty = difficulty, match_reward = match_reward, 
                          depth_reward_factor = depth_reward_factor, coupling_map = coupling_map, 
                        difficulty_threshold = difficulty_threshold, modes = modes, 
                        diff_ratio = None, inc_reward_factor = inc_reward_factor, 
                        inc_reward_option = inc_reward_option)
        train_env, eval_env = PartitionEnvs(env_config = env_config, num_envs = num_envs, 
                                            eval_ratio = eval_ratio)

        # Creating callback instances ----------------------------------------------------------
        
        eval_callback = EvalCallback(
            eval_env,
            eval_freq = eval_freq,
            n_eval_episodes = n_eval_episodes,
            deterministic = True,
            render = False,
            verbose = 0,
            best_model_save_path = f"{log_dir}/Best_Model_Difficulty{difficulty}",
            log_path = f"{log_dir}/Callback_Log_Difficulty{difficulty}"
        )
        adaptive_hyp_callback = AdaptiveHyperparameterCallback(
            eval_callback = eval_callback, 
            verbose = 0, 
            window_size = hyp_window_size, 
            
            max_learning_rate = max_learning_rate,
            min_learning_rate = min_learning_rate,
            learning_rate_paramter = learning_rate_parameter,
            
            max_ent_coef = max_ent_coef,
            min_ent_coef = min_ent_coef,
            ent_coef_parameter = ent_coef_parameter,
            
            max_clip_range = max_clip_range,
            min_clip_range = min_clip_range,
            clip_range_parameter = clip_range_parameter
        )
        callback_list = CallbackList([eval_callback, adaptive_hyp_callback])

        # Load pre-trained model or initialize-------------------------------------------------
        
        if difficulty == 1:
            # Create model with optimized hyperparameters
            model = PPO(
                "CnnPolicy",
                train_env,
                learning_rate = initial_learning_rate,
                n_steps = n_steps,
                batch_size = batch_size,
                n_epochs = n_epochs,
                gamma = gamma,
                gae_lambda = gae_lambda,
                clip_range = initial_clip_range,
                ent_coef = initial_ent_coef,
                policy_kwargs = policy_kwargs,
                verbose = 0,
                device = device
                )
        else:
            best_model_directory = glob.glob(f"{log_dir}/best_model.zip")
            if best_model_directory:
                print(f"Loading existing model")
                best_model_directory.sort()
                model = PPO.load(best_model_directory[-1], env = train_env, device = device)

                model.learning_rate = lambda progress_remaining: initial_learning_rate
                model.n_steps = n_steps
                model.batch_size = batch_size
                model.n_epochs = n_epochs
                model.gamma = gamma
                model.gae_lambda = gae_lambda
                model.clip_range = lambda progress_remaining: initial_clip_range
                model.ent_coef = initial_ent_coef
                model.policy_kwargs = policy_kwargs
            else:
                raise LookupError("No existing model found, but difficulty is not 1")
            
        # Train the model
        model.learn(total_timesteps = total_timesteps, callback = callback_list)

        #------------------------------------------------------------------------------------

        rewards_list = eval_callback.evaluations_results
        tune_metric = np.mean(rewards_list)

        train_env.close()
        eval_env.close()

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return float(tune_metric)
    return SetDifficultyObjective

def CurriculumnObjectiveMaker(fixed_params: dict[str, Any]) -> Callable:
    """
    A function that creates an Optuna objective function involving a mini curriculum.

    Args:
        fixed_params (dict[str, Any]): A dictionary of parameters that are kept constant
        during the optimization trials. For details, ref annotations on 
        CliffSyn.Curriculum.TrainCurriculum.

    Returns:
        CurriculumnObjective (Callable): The objective function for Optuna's study.
    """
    def CurriculumnObjective(trial: optuna.Trial) -> float:
        """
        Defines one trial of hyperparameter optimization over an entire (mini) curriculum.
        The goal is to minimize the total time taken to complete the curriculum.

        Args:
            trial (optuna.Trial): An Optuna trial object.

        Returns:
            objective_value (float): The negative of the total runtime, so Optuna maximizes it.
        """

        # Set Device
        if torch.cuda.is_available():
            torch_device = "cuda"
        # elif torch.backends.mps.is_available():
            # torch_device = "mps"
        else:
            torch_device = "cpu"
        device = torch.device(torch_device)
        print(device)

        # Fixed parameters --------------------------------------------------------------------

        modes = ["training"]

        log_dir = fixed_params["log_dir"]
        skip_to_difficulty = fixed_params.get("skip_to_difficulty", None)

        num_lvs = fixed_params["num_lvs"]
        num_qudits = fixed_params["num_qudits"]
        coupling_map = fixed_params["coupling_map"]
        
        max_difficulty = fixed_params["max_difficulty"]
        difficulty_threshold = fixed_params["difficulty_threshold"]
        max_gates = fixed_params["max_gates"]
        num_envs = fixed_params["num_envs"]
        
        eval_ratio = fixed_params["eval_ratio"]
        eval_freq = fixed_params["eval_freq"]
        n_eval_episodes = fixed_params["n_eval_episodes"]
        succ_window_size = fixed_params["succ_window_size"]
        hyp_window_size = fixed_params["hyp_window_size"]
        success_threshold = fixed_params["success_threshold"]
        
        match_reward = fixed_params["match_reward"]
        depth_reward_factor = fixed_params["depth_reward_factor"]
        inc_reward_option = fixed_params["inc_reward_option"]
        inc_reward_factor = fixed_params["inc_reward_factor"]

        # Hyperparameters to optimize ---------------------------------------------------------

        n_steps = trial.suggest_int("n_steps", 512, 2048, log = True)
        batch_size = trial.suggest_int("batch_size", 128, 512, log = True)
        n_epochs = trial.suggest_int("n_epochs", 3, 10)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.9999)

        initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-3, log=True)
        learning_rate_parameter = trial.suggest_float("learning_rate_parameter", 0.01, 1)
        max_learning_rate = trial.suggest_float("max_learning_rate", 1e-4, 1e-3, log=True)
        min_learning_rate = trial.suggest_float("min_learning_rate", 1e-6, 1e-4, log=True)
        if min_learning_rate > max_learning_rate:
            min_learning_rate, max_learning_rate = max_learning_rate, min_learning_rate
        
        initial_clip_range = trial.suggest_float("initial_clip_range", 0.05, 0.3)
        clip_range_parameter = trial.suggest_float("clip_range_parameter", 0.01, 1)
        max_clip_range = trial.suggest_float("max_clip_range", 0.1, 0.5)
        min_clip_range = trial.suggest_float("min_clip_range", 0.01, 0.1)
        if min_clip_range > max_clip_range:
            min_clip_range,  max_clip_range = max_clip_range, min_clip_range
        
        initial_ent_coef = trial.suggest_float("initial_ent_coef", 0.0, 0.02)
        ent_coef_parameter = trial.suggest_float("ent_coef_parameter", 0.01, 1)
        max_ent_coef = trial.suggest_float("max_ent_coef", 0.1, 0.3)
        min_ent_coef = trial.suggest_float("min_ent_coef", 0, 0.1)
        if min_ent_coef > max_ent_coef:
            min_ent_coef, max_ent_coef = max_ent_coef, min_ent_coef

        features_dim = trial.suggest_int("features_dim", 256, 1024)
        conv_channels = trial.suggest_int("conv_channels", 64, 256)
        fc1_units = trial.suggest_int("fc1_units", 512, 1024)
        embedding_dim = trial.suggest_int("embedding_dim", 1, num_lvs)

        diff_ratio = trial.suggest_float("diff_ratio", 0, 1)

        #-------------------------------------------------------------------------------------

        #cumulative_reward_list = []
        runtime_list = []

        # Define policy kwargs with optimized CNN architecture
        policy_kwargs = dict(
            features_extractor_class = CustomCliffordCNN,
            features_extractor_kwargs = dict(
                features_dim = features_dim,
                conv_channels = conv_channels,
                fc1_units = fc1_units,
                num_lvs = num_lvs, 
                embedding_dim = embedding_dim,
            ),
            normalize_images=False
        )
        # Curriculum Loop ----------------------------------------------------------------------
        if skip_to_difficulty is None: skip_to_difficulty = 1
        model = None
        for difficulty in range(skip_to_difficulty, max_difficulty+1):
        
            # Creating training and evaluation environments
            env_config = dict(num_lvs = num_lvs, num_qudits = num_qudits, max_gates = max_gates, 
                              curriculum_difficulty = difficulty, match_reward = match_reward, 
                              depth_reward_factor = depth_reward_factor, coupling_map = coupling_map, 
                              difficulty_threshold = difficulty_threshold, modes = modes, 
                              diff_ratio = diff_ratio, inc_reward_factor = inc_reward_factor, 
                              inc_reward_option = inc_reward_option)
            train_env, eval_env = PartitionEnvs(env_config, num_envs = num_envs, 
                                                eval_ratio = eval_ratio)

            # Load or initialize model --------------------------------------------------------
            
            if model is None:
                best_model_directory = glob.glob(f"{log_dir}/best_model.zip")
                if best_model_directory:
                    best_model_directory.sort()
                    model = PPO.load(best_model_directory[-1], env = train_env, device = device)

                    model.learning_rate = lambda progress_remaining: initial_learning_rate
                    model.n_steps = n_steps
                    model.batch_size = batch_size
                    model.n_epochs = n_epochs
                    model.gamma = gamma
                    model.gae_lambda = gae_lambda
                    model.clip_range = lambda progress_remaining: initial_clip_range
                    model.ent_coef = initial_ent_coef
                    model.policy_kwargs = policy_kwargs
                else:
                    model = PPO(
                        "CnnPolicy",
                        train_env,
                        learning_rate = initial_learning_rate,
                        n_steps = n_steps,
                        batch_size = batch_size,
                        n_epochs = n_epochs,
                        gamma = gamma,
                        gae_lambda = gae_lambda,
                        clip_range = initial_clip_range,
                        ent_coef = initial_ent_coef,
                        policy_kwargs = policy_kwargs,
                        verbose = 0,
                        device = device
                        )
            else:
                model.set_env(train_env)
            
            # Creating callback instances -----------------------------------------------------
            
            eval_callback = EvalCallback(
                eval_env,
                eval_freq = eval_freq,
                deterministic = True,
                render = False,
                n_eval_episodes = n_eval_episodes,
                verbose = 0,
                best_model_save_path = f"{log_dir}/Best_Model_Difficulty{difficulty}",
                log_path = f"{log_dir}/Callback_Log_Difficulty{difficulty}",
                )
            success_stop_callback = SuccessStopCallback(
                eval_callback = eval_callback, 
                verbose = 0, 
                window_size = succ_window_size, 
                success_threshold = success_threshold
            )    
            adaptive_hyp_callback = AdaptiveHyperparameterCallback(
                eval_callback = eval_callback, 
                verbose = 0, 
                window_size = hyp_window_size, 
                
                max_learning_rate = max_learning_rate,
                min_learning_rate = min_learning_rate,
                learning_rate_paramter = learning_rate_parameter,
                
                max_ent_coef = max_ent_coef,
                min_ent_coef = min_ent_coef,
                ent_coef_parameter = ent_coef_parameter,
                
                max_clip_range = max_clip_range,
                min_clip_range = min_clip_range,
                clip_range_parameter = clip_range_parameter
            ) 
            # Merge callbacks into one list
            callback_list = CallbackList([eval_callback, success_stop_callback, 
                                          adaptive_hyp_callback])

            # Calculating the objective --------------------------------------------------------
            
            start_time = time.time()
            # Training the model. Since total_timesteps is so large, 
            # the training is expected to be stopped only by callbacks.
            model.learn(total_timesteps = int(1e10), callback = callback_list)
            end_time = time.time()
            
            # TODO: The alternate objective below is theoretically more relevant for a compiler,
            # but training has been extremely slow and I've never seen the curriculum 
            # reach difficulty > difficulty_threshold for a reasonable threshold...
            """
            if difficulty > difficulty_threshold:
                rewards_list = eval_callback.evaluations_results
                average_reward = np.average(rewards_list)
                cumulative_reward_list.append(average_reward)
            """
            runtime_list.append(end_time - start_time)
        total_runtime = np.sum(runtime_list)

        train_env.close() # type: ignore
        eval_env.close() # type: ignore

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        #return np.average(cumulative_reward_list)
        return -1*total_runtime
    return CurriculumnObjective

def LoadHyperParam(log_dir: str, default_params: dict) -> dict:
    """
    Loads optimized hyperparameters from a file if available. Otherwise, returns defaults.

    Args:
        log_dir (str): The directory where the hyperparameter file is stored.

        default_params (dict): A dictionary of default parameters to use if no file is found.

    Returns:
        params (dict): The loaded or default hyperparameters.
    """
    try:
        with open(log_dir+"/best_hyperparameters.json", "r") as f:
            params = json.load(f)["best_params"]
        print("Using optimized hyperparameters")
    except FileNotFoundError:
        params = default_params
        print("No optimized hyperparameters found. Using defaults")
    return params

def TuneHyperParam(num_hypparam_trials: int, log_dir: str, ObjectiveFunc: Callable):
    """
    Tunes hyperparameters.

    Args:
        num_hypparam_trials (int): Number of hyperparameter combinations to test

        log_dir (str): The directory to store the tuned hyperparameters.

        ObjectiveFunc (Callabe): The Optuna objective to optimize-for
    """
    # Create study
    study = optuna.create_study(
        direction = "maximize",
        sampler = TPESampler(),
        pruner = MedianPruner()
    )
    # Optimize
    study.optimize(
        ObjectiveFunc, 
        n_trials = num_hypparam_trials, 
        show_progress_bar = True
    )
    trial = study.best_trial

    # Save results
    results = {
        "best_value": trial.value,
        "best_params": trial.params
    }
    with open(log_dir+"/best_hyperparameters.json", "w") as f: 
        json.dump(results, f, indent=4)
    
    return results

# End of File #################################################################################