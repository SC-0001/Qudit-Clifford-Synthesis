"""
Implements the curriculum learning training loop and associated helper callbacks
for the Clifford synthesis agent.
"""

import numpy as np
from typing import Tuple, Dict, Callable
import glob
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from qudit_clifford_synthesis.CliffSyn.Environment import CliffordSynthesisEnv
from qudit_clifford_synthesis.CliffSyn.CNN import CustomCliffordCNN

# Useful Functions ##########################################################################

def MakeEnv(env_config: Dict, seed: int) -> Callable:
    """
    Creates a helper function to generate a new training environment.

    Args:
        env_config (Dict): Configuration dictionary for the environment.

        seed (int): The reset seed for the environment.

    Returns:
        _NewEnvGenerator (Callable): A function that returns a new environment.
    """
    def _NewEnvGenerator():
        env = CliffordSynthesisEnv(env_config)
        env.reset(seed = seed)
        return env
    return _NewEnvGenerator

def MakeEvalEnv(env_config: Dict, seed: int) -> Callable:
    """
    Creates a helper function to generate a new evaluation environment.
    Wraps the new environment with Monitor.

    Args:
        env_config (Dict): Configuration dictionary for the environment.

        seed (int): The reset seed for the environment.

    Returns:
        _NewEnvGenerator (Callable): A function that returns a new environment.
    """
    def _NewEnvGenerator():
        env = CliffordSynthesisEnv(env_config)
        env.reset(seed = seed)
        return Monitor(env)
    return _NewEnvGenerator

def PartitionEnvs(env_config, num_envs: int, eval_ratio: float
                  )-> Tuple[SubprocVecEnv, SubprocVecEnv]:
    """
    Creates vectorized environments for training and evaluation.

    Args:
        env_config (Dict): Configuration dictionary for the environments.

        num_envs (int): The total number of training environments.

        eval_ratio (float): The ratio of evaluation environments relative to training ones.

    Returns:
        train_env (SubprocVecEnv): The vectorized environment for training.

        eval_env (SubprocVecEnv): The vectorized environment for evaluation.
    """
    train_env = SubprocVecEnv([
        MakeEnv(env_config = env_config, seed = i)
        for i in range(num_envs)
    ])
    eval_env_config = env_config.copy()
    eval_env_config["diff_ratio"] = None
    
    eval_env = SubprocVecEnv([
        MakeEvalEnv(eval_env_config, seed = num_envs + i)
        for i in range(round(num_envs*eval_ratio))
    ])
    return train_env, eval_env

class SuccessStopCallback(BaseCallback):
    """
    Detects that the average success rate of the last "window_size" episodes is beyond a 
    certain threshold, and stops training at the current difficulty. This makes training 
    move onto the next difficulty.
    """
    def __init__(
            self, verbose: int, 
            eval_callback: EvalCallback, 
            window_size: int, 
            success_threshold: float
        ):
        """
        Args:
            eval_callback (EvalCallback): The evaluation callback instance to get 
                success data from.

            verbose (int): The verbosity level.

            window_size (int): The number of recent evaluations to average for the success rate.

            success_threshold (float): The success rate (0.0 to 1.0) required to stop training.
        """
        super().__init__(verbose)
        self._eval_callback = eval_callback
        self._window_size = window_size
        self._success_threshold = success_threshold

    def _on_step(self) -> bool:
        """
        Called after each "env.step()". Stops training if the success condition is met.

        Returns:
            continue_training (bool): False if training should be stopped, True otherwise.
        """
        if self._SuccessRate() >= self._success_threshold:
            return False
        return True

    def _SuccessRate(self) -> float:
        """
        Calculates the average success rate over the last "window_size" evaluations.
        """
        success_list = self._eval_callback.evaluations_successes
        window_size = self._window_size
        length = len(success_list)

        if length < window_size: return 0
        latest_success_list = success_list[length-window_size : length]
        
        metric = np.average(latest_success_list)
        return float(metric)
    
class AdaptiveHyperparameterCallback(BaseCallback):
    """
    A custom callback to adapt hyperparameters based on evaluation success rate.
    """
    def __init__(
            self, verbose: int, 
            eval_callback: EvalCallback, 
            window_size: int,
            
            max_learning_rate: float, min_learning_rate: float, learning_rate_paramter: float,     
            max_clip_range: float, min_clip_range: float, clip_range_parameter: float, 
            max_ent_coef: float, min_ent_coef: float, ent_coef_parameter: float
        ):
        """
        Args:
            eval_callback (EvalCallback): The evaluation callback to get success data from.
            
            verbose (int): The verbosity level.
            
            window_size (int): The number of recent evaluations to average for success rate.
    
            max_learning_rate (float): The upper bound for the learning rate.
            min_learning_rate (float): The lower bound for the learning rate.
            learning_rate_paramter (float): The decay parameter for the learning rate.
            
            max_clip_range (float): The upper bound for the PPO clip range.
            min_clip_range (float): The lower bound for the PPO clip range.
            clip_range_parameter (float): The decay parameter for the clip range.
            
            max_ent_coef (float): The upper bound for the entropy coefficient.
            min_ent_coef (float): The lower bound for the entropy coefficient.
            ent_coef_parameter (float): The decay parameter for the entropy coefficient.
        """
        super().__init__(verbose)
        self._eval_callback = eval_callback
        self._window_size = window_size

        # Creating functions to calculate new hyperparameter values based on the 
        # current success rate.
        self._LearningRateFunc = self._HypUpdateFunc(
            max = max_learning_rate,
            min = min_learning_rate,
            parameter = learning_rate_paramter
        )
        self._ClipRangeFunc = self._HypUpdateFunc(
            max = max_clip_range,
            min = min_clip_range,
            parameter = clip_range_parameter
        )
        self._EntCoefFunc = self._HypUpdateFunc(
            max = max_ent_coef,
            min = min_ent_coef,
            parameter = ent_coef_parameter
        )

    def _on_step(self) -> bool:
        """
        Called after each step. Updates hyperparameters based on performance.
        """
        success_rate = self._SuccessRate()
        if success_rate is None: return True
        
        new_learning_rate = self._LearningRateFunc(success_rate)
        new_clip_range = self._ClipRangeFunc(success_rate)
        new_ent_coef = self._EntCoefFunc(success_rate)
        
        # Update the model's hyperparameters
        self.model.learning_rate = new_learning_rate
        self.model._setup_lr_schedule()

        self.model.clip_range = lambda progress_remaining: new_clip_range #type:ignore
        self.model.ent_coef = new_ent_coef #type:ignore
        
        return True
    
    def _HypUpdateFunc(self, max: float, min: float, parameter: float) -> Callable:
        """
        Each hyperparameter is adjusted by a function F:[0,1] -> [min, max] that:
        1. is monotonically decreasing
        2. is steeper near 1 (negative 2nd derivative)
        3. has an additional parameter that can control its curvature

        Args: 
            max (float): The maximum output value of the function (achieved at success_rate = 0).

            min (float): The minimum output value of the function (achieved at success_rate = 1).

            parameter (float): Modifies the curvature of F. If parameter = 0, F is linear.
        """
        assert parameter <= 1
        F = lambda success_rate: min + (max - min) * (1 - success_rate)**parameter
        return F
    
    def _SuccessRate(self) -> float | None:
        """
        Calculates the average success rate over the last "window_size" evaluations.
        """
        success_list = self._eval_callback.evaluations_successes
        window_size = self._window_size
        length = len(success_list)

        if length < window_size: return None
        latest_success_list = success_list[length-window_size : length]

        metric = np.average(latest_success_list)
        return float(metric)
    
# Curriculum Training Function ################################################################

def TrainCurriculum(
        params: Dict, log_dir: str, num_lvs:int, num_qudits:int, num_envs: int, 
        eval_ratio: float, eval_freq: int, n_eval_episodes: int, max_difficulty: int, 
        difficulty_threshold: int, coupling_map: list[list[int]], match_reward: int, 
        depth_reward_factor: int, modes: list[str],  inc_reward_option: int, 
        inc_reward_factor: float, success_threshold: float, succ_window_size: int, 
        hyp_window_size: int, debug_difficulties: list[int] | None = None, 
        skip_to_difficulty: int | None = None, max_gates: int | None = None) -> None:
    """
    Main training loop for curriculum learning. We iterate through difficulty levels, 
    creating and training a PPO model at each stage until a performance threshold is met.

    Args:
        # Training arguments -------------------
        log_dir (str): The directory to save logs and models.
        params (Dict): A dictionary of hyperparameters for the PPO model and training.
        num_envs (int): The number of parallel environments for training.
        max_difficulty (int): The final difficulty level for the curriculum.
        skip_to_difficulty (int | None): A difficulty level to start the curriculum from.

        # Evaluation arguments -----------------
        eval_ratio (float): Ratio of evaluation environments to training environments.
        eval_freq (int): How often to run evaluations (in steps).
        n_eval_episodes (int): The number of episodes to run for each evaluation.

        # Callback arguments -------------------
        success_threshold (float): The success rate required to advance the curriculum.
        succ_window_size (int): The window size for the SuccessStopCallback.
        hyp_window_size (int): The window size for the AdaptiveHyperparameterCallback.
        
        # Core environment arguments -----------
        num_lvs (int): Number of energy levels for each qudit.
        num_qudits (int): Number of qudits in the circuit.
        coupling_map (list): A list of undirected pairs defining allowed two-qudit interactions.
        difficulty_threshold (int): The target circuit becomes random beyond this threshold.
        
        # Environment reward arguments ----------
        match_reward (int): Reward given for successfully matching the target.
        depth_reward_factor (int): Factor for bonus reward based on depth savings.
        inc_reward_option (int): Which custom metric to use for incremental reward.
        inc_reward_factor (float): Factor for the incremental reward.
        
        # Conditional environment arguments -----
        modes (list[str]): A list of modes. "training", "prediction", "debug".
        max_gates (int | None): Maximum no. of gates the agent can apply in "prediction" mode.
        debug_difficulties (list[int] | None): A list of difficulties to log in debug mode.
    """
    # Set Device
    if torch.cuda.is_available():
        torch_device = "cuda"
    # elif torch.backends.mps.is_available():
        # torch_device = "mps"
    else:
        torch_device = "cpu"
    device = torch.device(torch_device)
    print(f"Using device: {device}")

    #-Curriculum Loop---------------------------------------------------------------------------
    
    if skip_to_difficulty is None: skip_to_difficulty = 1
    
    model = None
    for difficulty in range(skip_to_difficulty, max_difficulty+1):

        # Create training and evaluation environments
        env_config = dict(num_lvs = num_lvs, num_qudits = num_qudits, max_gates = max_gates, 
                          curriculum_difficulty = difficulty, match_reward = match_reward, 
                          depth_reward_factor = depth_reward_factor, modes = modes,
                          coupling_map = coupling_map, debug_difficulties = debug_difficulties, 
                          difficulty_threshold = difficulty_threshold,  
                          log_dir = log_dir, diff_ratio = params["diff_ratio"],
                          inc_reward_factor = inc_reward_factor, 
                          inc_reward_option = inc_reward_option)
        train_env, eval_env = PartitionEnvs(env_config = env_config, num_envs = num_envs, 
                                            eval_ratio = eval_ratio)

        #-Initialize or Update model------------------------------------------------------------

        if model is None:

            # Load previous best model if available. Otherwise, initialize.
            best_model_directory = glob.glob(f"{log_dir}/best_model.zip")
            if best_model_directory:
                print(f"Loading existing model")
                best_model_directory.sort()
                model = PPO.load(best_model_directory[-1], env = train_env, device = device)

                model.learning_rate = lambda progress: params["initial_learning_rate"]
                model.clip_range = lambda progress: params["initial_clip_range"]
                model.ent_coef = params["initial_ent_coef"]
            else:
                print("No existing model found. Creating a new one.")
                policy_kwargs = dict(
                    features_extractor_class = CustomCliffordCNN,
                    features_extractor_kwargs = dict(
                        features_dim = params['features_dim'],
                        conv_channels = params['conv_channels'],
                        fc1_units = params['fc1_units'],
                        num_lvs = num_lvs, 
                        embedding_dim = params["embedding_dim"],
                    ),
                    normalize_images = False 
                )
                model = PPO(
                            "CnnPolicy",
                            train_env,
                            learning_rate = params["initial_learning_rate"],
                            n_steps = params["n_steps"],
                            batch_size = params["batch_size"],
                            n_epochs = params["n_epochs"],
                            gamma = params["gamma"],
                            gae_lambda = params["gae_lambda"],
                            clip_range = params["initial_clip_range"],
                            ent_coef = params["initial_ent_coef"],
                            policy_kwargs = policy_kwargs,
                            verbose = 0,
                            tensorboard_log = log_dir,
                            device = device
                )
        else:
            # Update environment while keeping learned parameters
            model.set_env(train_env)

        # Define Callbacks ---------------------------------------------------------------

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path = f"{log_dir}/Best_Model_Difficulty{difficulty}",
            log_path = f"{log_dir}/Callback_Log_Difficulty{difficulty}",
            eval_freq = eval_freq,
            deterministic = False,
            render = False,
            n_eval_episodes = n_eval_episodes,
            verbose = 0
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
            
            max_learning_rate = params["max_learning_rate"],
            min_learning_rate = params["min_learning_rate"],
            learning_rate_paramter = params["learning_rate_parameter"],
            
            max_ent_coef = params["max_ent_coef"],
            min_ent_coef = params["min_ent_coef"],
            ent_coef_parameter = params["ent_coef_parameter"],
            
            max_clip_range = params["max_clip_range"],
            min_clip_range = params["min_clip_range"],
            clip_range_parameter = params["clip_range_parameter"]
        )
        
        # Merge callbacks into one list
        callback_list = CallbackList([eval_callback, 
                                      success_stop_callback, 
                                      adaptive_hyp_callback])

        # Training ---------------------------------------------------------------------------
        print(f"Starting curriculum difficulty {difficulty}")
        try:
            # Practically infinite timesteps. Learning only stops when the success rate
            # threshold is met.
            model.learn(
                total_timesteps = int(1e10), 
                callback = callback_list,
                reset_num_timesteps = False,
                tb_log_name = f"Training_Log_Difficulty{difficulty}"
            )
        finally:
            # Close environments
            train_env.close()
            eval_env.close()

    # Save final model
    if model is not None: 
        model.save(log_dir+"/Final_Model")
    
    print("Training completed!")
    return None

# End of File ################################################################################