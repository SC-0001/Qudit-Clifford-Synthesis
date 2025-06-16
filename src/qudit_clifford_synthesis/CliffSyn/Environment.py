"""
Reinforcement learning environment for (homogeneous qudit) Clifford circuit synthesis.
For now, the agent can only use a pre-defined set of generators.
"""

import gymnasium as gym 
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
from itertools import combinations
import json
from numpy import linalg as LA
import copy
from sympy import factorint

from qudit_clifford_synthesis.Essentials.QuditCirc import QuditCircuit
from qudit_clifford_synthesis.Essentials.CliffGates import SymForm, CliffGateSet

# Useful Functions ########################################################################

def CustomMetric(qc1: QuditCircuit, qc2: QuditCircuit, option: int) -> float:
    """
    Calculates a distance metric between two Clifford circuits based on their
    symplectic representations. Used for calculating incremental rewards.

    Potential metrices should satisfy Metric(A, A) ≥ Metric(A, B) for B ≠ A
    TODO: establish more rigorous criterion

    Args:
        qc1: The first QuditCircuit object.
        qc2: The second QuditCircuit object.
        option: An integer selecting the metric formula to use.

    Returns:
        A float representing the calculated distance/similarity.
    """
    # Convert to dtype = np.int32 to allow subtraction
    sym1 = qc1.rep.copy().astype(np.int32)
    sym2 = qc2.rep.copy().astype(np.int32)

    num_qudits = qc1.num_qudits
    assert num_qudits == qc2.num_qudits

    num_lvs = qc1.num_lvs
    assert num_lvs == qc2.num_lvs
    
    if option == 0:
        return 0
    
    elif option == 1:
        # TODO: modify the metric such Metric(A, A) > Metric(A, B) for B ≠ A

        # Recall S^{T}ΩS = Ω, for all S ∈ Sym(2d)
        Ω = SymForm(num_qudits = num_qudits, num_lvs = num_lvs)
        sym2_inv = (Ω @ sym2.T @ Ω) % num_lvs
        
        tr_distance = np.trace((sym1 @ sym2_inv) % num_lvs)
        normalized_distance = tr_distance / (2*num_qudits)
        return normalized_distance
    
    elif option == 2:
        # TODO: Currently problematic because the operations involved don't respect modular arithmetic
        difference = sym1 - sym2
        return float(-1*LA.norm(difference, ord='fro'))
    
    elif option == 3:
        # This metric computes a normalized L1 distance
        difference = sym1 - sym2
        modular_difference = np.minimum(np.abs(difference), num_lvs - np.abs(difference))
        l1_distance = np.sum(modular_difference)

        element_count = (2 * num_qudits)**2
        # Thinking of mod space as a circle, this makes sense
        max_element_dist = np.floor(num_lvs / 2)
        max_possible_l1_dist = element_count * max_element_dist
        assert max_possible_l1_dist != 0
        normalized_distance = l1_distance / max_possible_l1_dist

        return 1 - normalized_distance
    
    else:
        raise NotImplementedError("Not Implemented")

def Flatten(obs_array: np.ndarray) -> np.ndarray:
    """
    Converts obs to Sym. 
    For debugging purposes.
    """
    num_lvs = len(obs_array)
    num_qudits_x2 = len(obs_array[0])
    ans_array = np.zeros((num_qudits_x2, num_qudits_x2), dtype = np.uint32)

    for lv in range(num_lvs):
        for row_index in range(num_qudits_x2):
            for col_index in range(num_qudits_x2):
                if obs_array[lv, row_index, col_index] == 1:
                    ans_array[row_index, col_index] = lv
    return ans_array

def ExpSample(max: int, ratio: float) -> int:
    """
    Samples a positive integer not exceeding 'max'. 
    The probability that 'max - 1' is sampled is 'ratio' times the probability 
    that 'max' is sampled.
    """
    assert max >= 1
    option_list = range(1, max + 1)

    # Satisfies e^{α max} * r = e^{α (max-1)}
    alpha = -np.log(ratio)  

    # Create array of probabilities
    p_list = np.exp(option_list * alpha)
    p_list /= np.sum(p_list)

    # Sample
    sample = np.random.choice(option_list, size=1, p = p_list)[0]
    return sample # type: ignore

def SymTo3D(tensor2D: np.ndarray, buffer3D: np.ndarray) -> None:
    """
    Re-shapes an even-dimensional square array of shape (2n, 2n) to a 
    3D array of shape (4, n, n). This 3D array is stored in a buffer.

    Args:
        tensor2D: The (2n, 2n) array.
        buffer3D: The pre-allocated numpy array of shape (4, n, n) to store the result.
    """
    dim = tensor2D.shape[0]
    half_dim = int(dim/2)
    assert dim == half_dim*2

    buffer3D[0] = tensor2D[0:half_dim, 0:half_dim]
    buffer3D[1] = tensor2D[0:half_dim, half_dim:dim]
    buffer3D[2] = tensor2D[half_dim:dim, 0:half_dim]
    buffer3D[3] = tensor2D[half_dim:dim, half_dim:dim]

def CliffordCardinality(num_lvs: int, num_qudits: int) -> int:
    """
    Calculates the cardinality of the (projective) Clifford group 
    C_{num_lvs}^{num_qudits} / U(1). 

    Derivation comes from:
    https://quantumcomputing.stackexchange.com/questions/13643
    """
    def PrimePowerCliffordCardinality(d, n):
        number = d**(n**2+2*n)
        for i in range(1, n+1):
            number *= d**(2*i)-1
        return number
    
    final_number = 1
    factor_dict = factorint(num_lvs)
    for prime_factor in factor_dict:
        final_number *= PrimePowerCliffordCardinality(
            d = prime_factor ** factor_dict[prime_factor], 
            n = num_qudits
        )
    return final_number

# Main #####################################################################################

class CliffordSynthesisEnv(gym.Env):
    """
    A Gymnasium environment for synthesizing Clifford circuits.

    The agent's goal is to find a sequence of gates that implements a randomly
    generated target Clifford operation.
    """
    def __init__(self, env_config: Dict):
        """
        Args: 
            env_config (Dict): A dictionary containing configuration parameters:

            # Necessary parameters -------------------------------------------------------
                
                num_lvs (int): Number of energy levels for ALL qudits.
                
                num_qudits (int): Number of qudits in the circuits. 
                
                coupling_map (list[list[int]]): A list of non-directed pairs defining 
                    allowed two-qudit interactions. (e.g. due to hardware constraints)
                
                match_reward (float): Reward given for successfully matching the target.
                
                depth_reward_factor (float): Factor for bonus reward based on depth savings.
                
                inc_reward_factor (float): Factor for incremental reward based on progress.
                
                inc_reward_option (int): Which CustomMetric to use for incremental reward.
                
                curriculum_difficulty (int): The current difficulty level. Determines the 
                    number of gates in the target circuit.
                
                difficulty_threshold (int): Beyond this threshold, the target circuit becomes
                    random.
                
                diff_ratio (float): The ratio for exponential sampling of difficulty.
                
                modes (list[str]): Currently supports "training", "prediction", "debug".

            # Conditional parameters ------------------------------------------------------
                
                max_gates (int): Maximum number of gates the agent can apply under 
                    "prediction" mode. Otherwise, max_gates becomes the effective difficulty.
                
                log_dir (str): Directory path for logs. Only needed for "debug" mode.
                
                debug_difficulties (list[int]): List of curriculum difficulty levels for 
                    which to collect logs in "debug" mode. Logs for all levels aren't 
                    collected because they get big quickly.
        """
        super().__init__()
        self.env_config = env_config

        # Core parameters
        self.num_lvs: int = env_config["num_lvs"]
        self.num_qudits: int = env_config["num_qudits"]
        self.coupling_map: list[list[int]] = env_config.get(
            "coupling_map", 
            [list(elem) for elem in combinations(range(self.num_qudits), 2)]
        )
        # Reward parameters
        self.match_reward: float = env_config["match_reward"]
        self.depth_reward_factor: float = env_config["depth_reward_factor"]
        self.inc_reward_factor: float = env_config["inc_reward_factor"]
        self.inc_reward_option: int = env_config["inc_reward_option"]
        
        # Difficulty-related parameters
        self.curriculum_difficulty: int = env_config["curriculum_difficulty"]
        self.difficulty_threshold: int = env_config["difficulty_threshold"]
        self.diff_ratio: float | None = env_config.get("diff_ratio", None)
        
        # Mode-related parameters
        self.modes: list[str] = env_config["modes"]
        self.max_gates: int | None = env_config.get("max_gates", None)
        self.log_dir: str | None = env_config.get("log_dir", None)
        self.debug_difficulties: list[int] | None = env_config.get("debug_difficulties", None)
        
        # If num_qudits = 1, we'll have issues as two-qudit gates can't be applied
        assert self.num_qudits > 1
        assert self.curriculum_difficulty > 0

        # Defaults to a fully-connected coupling map
        if self.coupling_map is None:
            self.coupling_map = [list(elem) for elem in combinations(range(self.num_qudits), 2)]

        # To solve catastrophic forgetting, select a certain effective difficulty level following
        # a probability distribution, maxed at the actual difficulty. 
        # We want the distribution to bias more towards higher difficulties to make sure 
        # generalization occurs at a decent rate during training.
        if self.diff_ratio is not None:
            assert self.diff_ratio >= 0
            self.effective_difficulty = ExpSample(max = self.curriculum_difficulty, ratio = self.diff_ratio)
        else:
            self.effective_difficulty = self.curriculum_difficulty

        # Make the target circuit random upon hitting the difficulty threshold.
        if self.curriculum_difficulty > self.difficulty_threshold:
            # TODO: cardinality increases VERY quickly. Implement a more efficient way to generate random clifford gates
            self.effective_difficulty = CliffordCardinality(num_qudits = self.num_qudits, num_lvs = self.num_lvs)
        
        # While training, we max the number of gates the agent can place to the number of 
        # gates the target circuit has. This way, we prevent the agent from learning sub-optimal
        # policies for synthesis.
        if "prediction" in self.modes:
            assert self.max_gates is not None
        else:
            self.max_gates = self.effective_difficulty
        
        # Pre-allocate circuits & observation arrays for better performance
        self.buffer3D = np.empty([4, self.num_qudits, self.num_qudits], dtype = np.uint32)
        self.current_circuit = QuditCircuit(
            num_qudits = self.num_qudits, 
            num_lvs = self.num_lvs, 
            coupling_map = self.coupling_map, 
            rep_type = "Symplectic", 
            gate_set = CliffGateSet(self.num_lvs)
            )
        self.target_circuit = QuditCircuit(
            num_qudits = self.num_qudits, 
            num_lvs = self.num_lvs, 
            coupling_map = self.coupling_map, 
            rep_type = "Symplectic", 
            gate_set = CliffGateSet(self.num_lvs))
        
        # These attributes are needed because we need more Clifford generators for non-prime num_lvs
        self.is_prime = self.target_circuit.gate_set.is_prime  # type: ignore
        if not self.is_prime:
            self.multiplicative_inverse = self.target_circuit.gate_set.multiplicative_inverse # type: ignore
        
        # Prevent unwanted actions (e.g. target qudit = control qudit) without MaskPPO, etc
        # The 1st 'num_qudits' actions correspond to F. The 1st element is H applied to qudit 1, etc.
        # The 2nd 'num_qudits' actions (so 1+self.num_qudits ~ 2*num_qudits th actions) correspond to P.
        # The next 2*len(coupling_map) actions correspond to SUM applied to possible control-target pairs.
        #     Even-numbered pairs correspond to control-target pairs in the order in the coupling map.
        #     Odd-numbered pairs correspond to control-target pairs in flipped order wrt coupling map.
        # The remaining actions correspond to M gates. The first num_qudits actions correspond to Mi gates. 
        #     The next num_qudits actions correspond to Mj gates, etc. (where i and j are smallest 
        #     multiplicative inverses of num_lvs)
        
        # Action space:
        if self.is_prime:
            self.action_space = spaces.MultiDiscrete([
                2*self.num_qudits + 2*len(self.coupling_map), 
            ])
        else:
            self.action_space = spaces.MultiDiscrete([
                2*self.num_qudits + 2*len(self.coupling_map) + len(self.multiplicative_inverse)*self.num_qudits, 
            ])
        # Observation space: 
        self.observation_space = spaces.Box(
            low = 0, high = self.num_lvs - 1,
            shape = (4, self.num_qudits, self.num_qudits),
            dtype = np.uint32
        )
        self.reset()

    # Core Functions -----------------------------------------------------------------------

    def step(self, action):
        """
        Executes one time step within the environment. Applies a gate specified by the 
        action, calculates the reward, and checks if the episode has terminated.

        Args:
            action (np.ndarray): An array containing the action chosen by the agent. Ref 
                annotation on __init__ for details.

        Returns:
            observation (np.ndarray): The next state observation.
            
            reward (float): The reward for the action.
            
            terminated (bool): Whether the episode has ended.
            
            truncated (bool): Whether the episode was truncated (not used).
            
            info (dict): Carries optional information. Here, we store whether 
                (1) the agent's proposed ciruit matches with the target's, and 
                (2) the depth-saved by the agent IF the agent's circuit matches
        """
        # Updates in time ------------------------------------------------------------------

        # Converts the action to a gate application. 
        action_index = action[0]
        if action_index < self.num_qudits:
            gate_type = 'F'
            target = action_index
            control = None
        elif action_index < 2*self.num_qudits:
            gate_type = 'P'
            target = action_index - self.num_qudits
            control = None
        elif action_index < 2*self.num_qudits + 2*len(self.coupling_map):
            gate_type = 'SUM'
            coupling_index = action_index - 2*self.num_qudits
            selected_pair = self.coupling_map[coupling_index // 2]
            if coupling_index % 2:
                control, target = selected_pair[1], selected_pair[0]
            else:
                control, target = selected_pair[0], selected_pair[1]
        elif action_index < 2*self.num_qudits + 2*len(self.coupling_map) \
        + len(self.multiplicative_inverse)*self.num_qudits:
            reduced_action_index = action_index - 2*self.num_qudits - 2*len(self.coupling_map)
            
            M_index = reduced_action_index // self.num_qudits
            a_value = self.multiplicative_inverse[M_index]
            gate_type = 'M'+str(a_value)
            
            target = reduced_action_index % self.num_qudits
            control = None
        else:
            raise ValueError("Invalid action index")

        # Apply the agent's chosen gate to its circuit
        self.current_circuit.ApplyGate(gate_id = gate_type, control = control, target = target)
        
        # We update self.target_circuit itself. This is the "effective" target at the current
        # timestep. The target unitary follows U_{t+1} = U_{t} G_{t}^{-1} where G_{t}
        # is the gate corresponding to the action at timestep t.
        # Unlike the case d = 2, G_{t}^{-1} = G_{t} is generally not true for higher d.
        self.target_circuit.ApplyGate(gate_id = gate_type, control = control, target = target, reverse_order = True, inverse_gate = True)
        
        reward, match = self._RewardCalc()
        
        # Termination ----------------------------------------------------------------------
        
        # Check for termination conditions
        if match:
            self.info["is_success"] = True
            self.info["depth_saved"] = self.target_depth - self.current_circuit.Depth()
            self.done = True
        elif np.sum(self.current_circuit.gate_count) == self.max_gates:
            self.info["is_success"] = False
            self.done = True

        # In "debug" mode, update log
        if "debug" in self.modes and self.done == True:
            with open(self.log_dir+"/Comprehensive_Log.txt", 'a') as f:  # type: ignore
                f.write('\n'*3)
                f.write(f"Effective Difficulty: {self.effective_difficulty}\n")
                f.write(f"Configuration Dictionary: {self.env_config}\n")
                f.write(f"성공여부: {self.info["is_success"]} \n")
                f.write(f"Reward: {reward} \n")
                f.write(f"Target Depth: {self.target_depth} \n")
                f.write(f"Current Depth: {self.current_circuit.Depth()} \n")
                
                if self.info["is_success"] == True:
                    f.write(f"Depth saved: {self.info["depth_saved"]} \n")
                else:
                    f.write("Depth saved: N/A \n")
                
                f.write("Current Gate Placement History: \n")
                json.dump(self.current_circuit.gate_placement_history, f)
                f.write('\n')

                f.write("Target Gate Placement History: \n")
                json.dump(self.target_circuit.gate_placement_history, f)
                f.write('\n'*3)

        return self._CircToObs(), reward, self.done, False, self.info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        """
        Resets the environment to its initial state. Generate a new target Clifford circuit
        based on the current effective difficulty.

        Args:
            seed (int | None): A seed for the random number generator.
            
            options (dict | None): Additional options for resetting the environment.

        Returns:
            initial_target_obs (np.ndarray): Symplectic rep of the target circuit, re-shaped
                to a 3D array
            
            empty_dict (dict): Empty list
        """
        super().reset(seed = seed, options = options)

        if "debug" in self.modes: 
            assert self.log_dir is not None
            assert self.debug_difficulties is not None

            if (self.curriculum_difficulty not in self.debug_difficulties):
                self.modes.remove("debug")
            else:
                with open(self.log_dir+"/Comprehensive_Log.txt", 'a') as f:
                    f.write('\n'*3)
                    f.write(f"New Environment"+"#"*80+'\n')
                    f.write('\n'*3)

        self.done = False
        self.info = {}

        # Clear circuits instead of creating new ones
        self.current_circuit.Reset()
        self.target_circuit.Reset()

        # Generate target Clifford based on curriculum level
        for _ in range(self.effective_difficulty):
            if self.is_prime:
                # F, P, SUM
                gate_type = np.random.randint(3) 
            else:
                # F, P, SUM, all M gates
                number_of_M_gates = len(self.multiplicative_inverse)
                gate_type = np.random.randint(number_of_M_gates + 3)

            if gate_type == 0:
                target = np.random.randint(self.num_qudits)
                gate_id = 'F'
                control = None
            elif gate_type == 1:
                target = np.random.randint(self.num_qudits)
                gate_id = 'P'
                control = None
            elif gate_type == 2:
                index = np.random.randint(len(self.coupling_map))
                flip = np.random.randint(2)
                pair = self.coupling_map[index]
                if flip:
                    control, target = pair[1], pair[0]
                else:
                    control, target = pair[0], pair[1]
                gate_id = 'SUM'
            else:
                # M gates
                target = np.random.randint(self.num_qudits)
                a_value = self.multiplicative_inverse[gate_type-3]
                gate_id = 'M'+str(a_value)
                control = None
            self.target_circuit.ApplyGate(gate_id = gate_id, control = control, target = target)
        
        # Copy information about the initial target because self.target_circuit will be 
        # modified upon the agent applying gates.
        self.target_depth = self.target_circuit.Depth()
        self.initial_target_circuit = copy.deepcopy(self.target_circuit)

        # Calculate initial distance. Used for calculating the incremental reward.
        self.current_distance = CustomMetric(self.current_circuit, self.initial_target_circuit, option = self.inc_reward_option)

        initial_target_obs = self._CircToObs()
        empty_dict = {}
        return initial_target_obs, empty_dict

    # Non-Core Functions -------------------------------------------------------------------

    def _RewardCalc(self) -> tuple[float, bool]:
        """
        Calculates the reward for the most recent step.

        The reward consists of:
        - A small penalty for each gate used.
        - An incremental reward proportional to the progress made towards the target.
        - A large bonus for successfully matching the target.
        - An additional bonus for matching the target with a smaller depth than the original.

        Returns:
            reward (float): RL reward given to the agent

            match (bool): Whether the agent perfectly synthesized the initial target circuit
        """
        # Gate usage penalty
        reward = -1

        # Incremental reward
        previous_distance = self.current_distance
        self.current_distance = CustomMetric(self.current_circuit, self.initial_target_circuit, option = self.inc_reward_option)
        distance_change = self.current_distance - previous_distance
        reward += distance_change * self.inc_reward_factor
        
        match = False
        if np.array_equal(self.target_circuit.rep, np.identity(2 * self.num_qudits, dtype = np.uint32)):
            match = True
            # Bonus reward for matching the target circuit
            reward += self.match_reward
            # Extra reward for matching beating the target depth
            reward += (self.target_depth - self.current_circuit.Depth()) * self.depth_reward_factor
        return reward, match

    def _CircToObs(self) -> np.ndarray:
        """
        Converts the current target circuit's symplectic rep into a 3D observation tensor.
        (i.e. (2*self.num_qudits, 2*self.num_qudits) -> (4, self.num_qudits, self.num_qudits))
        The result is stored in the pre-allocated "self.buffer3D" for efficiency.

        Returns:
            self.buffer3D (np.ndarray): stores the reshaped target circuit's sym rep
        """
        sym_matrix = self.target_circuit.rep % self.num_lvs # type: ignore
        SymTo3D(sym_matrix, self.buffer3D)
        return copy.deepcopy(self.buffer3D)
        
    def GetAttr(self, attr_name):
        """
        Exists for debugging purposes.
        """
        return getattr(self, attr_name)

# End of File ##############################################################################