# Qudit Clifford Circuit Synthesis using Reinforcement Learning

In this project we attempt to develop a reinforcement learning (RL) agent capable of synthesizing Clifford circuits on homogeneous qudits. The agent simply observes the (symplectic) matrix representation of the target and places a gate from a fixed Clifford gate set onto its own circuit. This continues until the synthesis is complete or the agent uses up its maximum gate allowance.

Although we currently work with the symplectic representation of Clifford gates—due to the matrix size growing linearly with respect to the qudit count, rather than the exponential growth of unitary representations—the simple structure of the training environment allows for easy generalization to alternate represenations and gate sets.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
    * [Key Dependencies](#key-dependencies)
    * [Training the Agent](#training-the-agent)
    * [Tuning Hyperparameters](#tuning-hyperparameters)
    * [Evaluating the Model](#evaluating-the-model)
3. [Implementation Details](#implementation-details)
    * [Environment Basics](#environment-basics)
    * [Reward](#reward)
    * [Training Strategy](#training-strategy)
4. [Future Work](#future-work)
5. [References](#references)
6. [License](#license)

## Project Structure

The project is organized into a main `src` directory containing the core logic, and an `examples` directory with scripts to run experiments.

```text
.
├── examples/
│   ├── EvaluateModel.py        # Main script to evaluate a trained model
│   ├── Train.py                # Main script to train an agent
│   └── TuneHyp.py              # Main script to tune hyperparameters
│
└── src/qudit_clifford_synthesis/
    ├── Essentials/             # Core data structures for gates and circuits
    │   ├── Gates.py            # Base classes for quantum gates and gate sets
    │   ├── CliffGates.py       # Clifford generators (F, P, SUM, M_a)
    │   └── QuditCirc.py        # Main class for simulating qudit circuits
    │
    ├── CliffSyn/               # RL agent implementation
    │   ├── Environment.py      # The custom Gymnasium environment for Clifford synthesis
    │   ├── CNN.py              # Custom CNN feature extractor for the PPO agent
    │   ├── Curriculum.py       # Curriculum learning loop and callbacks
    │   └── HyperParam.py       # Hyperparameter tuning using Optuna
    │
    └── Evaluation/             # Scripts for model evaluation and comparison
        ├── MainEvaluation.py   # Core logic for running experiments and plotting results
        ├── SB3.py              # Wrapper to evaluate the trained Stable-Baselines3 agent
        ├── LEAP.py             # Wrapper to evaluate BQSKit's LEAP algorithm for comparison
        ├── SK (WIP).py         # (WIP) Solovay-Kitaev algorithm implementation
        └── QSD (WIP).py        # (WIP) Quantum Shannon Decomposition implementation
```

## Getting Started

### Key Dependencies

* `stable-baselines3[extra]`
* `gymnasium`
* `torch`
* `optuna`
* `bqskit`
* `numpy`
* `matplotlib`
* `sympy`

### Training the Agent

To start training a new agent or continue training a pre-existing model, run the training script from the `examples` directory. This script uses the `TrainCurriculum` function from `src/qudit_clifford_synthesis/CliffSyn/Curriculum.py`.

```bash
python examples/Train.py
```

There are too many parameters for on-the-fly keyboard inputs, so these should be modified manually within the Train.py file. By default, training progress, models, and logs are saved in the `./Training_Data/3Lv_3L_Ongoing` directory.

You can monitor the training process in real-time using TensorBoard:

  ```bash
  tensorboard --logdir './Training_Data/3Lv_3L_Ongoing'
  ```

### Tuning Hyperparameters

To find the optimal hyperparameters for training, run the tuning script from the `examples` directory. By default, training logs, best models, and the optimization result are stored in `./Training_Data/3Lv_3L_Hyp`.

```bash
python examples/TuneHyp.py
```

The default objective function is created using `SetDifficultyObjectiveMaker` from `src/qudit_clifford_synthesis/CliffSyn/HyperParam.py`. It trains the agent for a fixed number of steps at a single diffculty level and uses its performance as the metric to optimize. For a more useful (but much more time-consuming) optimization function, `CurriculumnObjectiveMaker` from `src/qudit_clifford_synthesis/CliffSyn/HyperParam.py` can be used.

### Evaluating the Model

To evaluate a trained agent and compare its performance against BQSKit's LEAP algorithm, run the evaluation script from the `examples` directory.

```bash
python examples/EvaluateModel.py
```

This script runs multiple trials for target circuits of varying difficulty. It collects metrics such as runtime, circuit depth saved, gate count saved, and the absolute value of the Hilbert-Schmidt inner product (a measure of synthesis accuracy). The results are plotted and saved as a PNG image in the `./Evaluation_Data` directory.

## Implementation Details

The agent is trained using the Proximal Policy Optimization (PPO) algorithm from Stable-Baselines3 within a custom `gymnasium` environment, `CliffordSynthesisEnv`.

### Environment Basics

* **Goal:** The synthesis problem is framed as a sequence of choices where the agent's goal is to reverse a target Clifford operation back to the identity matrix. If the agent applies a sequence of gates `G_1, G_2, ..., G_k`, the initial target `U_target` is updated to `U_target @ G_1_inv @ G_2_inv @, ... G_k_inv`. This way, instead of having to pass-through both the initial target and the agent's current circuit at each step, the agent can merely observe the updated target and predict which gate, placed on which qudit(s) is most likely to lead to correct synthesis.
* **State/Observation:** At each step, the agent observes the symplectic matrix of the remaining target operation. This `(2n, 2n)` integer matrix is reshaped into a `(4, n, n)` tensor to be fed into a CNN feature extractor.
* **Action Space:** The agent's action space is a single discrete integer, which maps to a specific gate (e.g., Fourier `F`, Phase `P`, `SUM`) applied to a specific qudit or pair of qudits.

### Reward

The reward function is composed of several components:

* A large, sparse reward (`match_reward`) is given upon successful synthesis of the target.
* A bonus reward (adjusted by `depth_reward_factor`) is added if the agent finds a solution with a lower circuit depth than the original target circuit, encouraging efficiency.
* A small penalty (fixed to -1) is applied for every gate used, encouraging shorter solutions.
* An incremental reward (adjusted by `inc_reward_factor`) is provided at each step. This is based on a custom distance metric that calculates how "close" the current circuit is to the target. This gives more frequent feedback to the agent, stabilizing the training process.

### Training Strategy

* **Curriculum Learning:** The agent begins by training on simple circuits (`difficulty` = 1). The `TrainCurriculum` uses `SuccessStopCallback` to automatically increase the difficulty after the agent achieves a consistent success rate (e.g., > 95%) on the current level. This ensures a smooth learning progression.
* **Mixing Difficulties** To fight against catastrophic forgetting, we randomly select an "effective difficulty" when creating an environment instance, maxed at the actual difficulty. We currently follow an exponential distribution for the random selection to bias more towards higher difficulties. The intention is to make sure generalization occurs at a decent rate during training.
* **Adaptive Hyperparameters:** Key PPO hyperparameters like the `learning_rate` and `clip_range` are dynamically adjusted via `AdaptiveHyperparameterCallback`, based on the agent's recent performance. When the agent is struggling, exploration is encouraged by increasing these values; when it is succeeding, they are lowered to fine-tune the policy.
* **Custom Feature Extractor:** The agent uses a `CustomCliffordCNN` feature extractor with an `nn.Embedding` layer to create dense vector representations (as opposed to when using one-hot encoding) of the sparse integer values in the observation tensor. This is followed by a convolutional layer and a MLP layer. The hope is that the convolutional layer will allow the model to learn about the simulated hardware's qudit couplings.

## Future Work

Several areas for future development have been identified:

* **Implement Improved Solovay-Kitaev (SK) Algorithm**: Complete the `SK (WIP).py` implementation to provide a method for decomposing arbitrary single-qudit unitaries into a sequence of gates from our fixed gate set. This would allow for a more complete synthesis pipeline when combined with LEAP. (to compare against)
* **Enhance Reward Function**: Refine the incremental reward metric in `Environment.py` to provide a more informative signal to the agent during training. Current metrics are based on intuition rather than mathematical rigor.
* **Further Stabilize Learning**: Despite existing measures, we still observe sudden drops in success rate even after the agent goes through multiple difficulties smoothly.

## References

* [Kremer, D., Villar, V., Paik, H., Duran, I., Faro, I., & Cruz-Benito, J. (2024). Practical and efficient quantum circuit synthesis and transpiling with Reinforcement Learning. *arXiv preprint arXiv:2405.13196*.](https://arxiv.org/abs/2405.13196)
* [Hostens, E., Dehaene, J., & De Moor, B. (2004). Stabilizer states and Clifford operations for systems of arbitrary dimensions, and modular arithmetic. *arXiv preprint quant-ph/0408190*.](https://arxiv.org/abs/quant-ph/0408190)
* [Smith, E., Davis, M. G., Larson, J. M., Younis, E., Bassman, L., Lavrijsen, W., & Iancu, C. (2022). LEAP: Scaling numerical optimization based synthesis using an incremental approach. *ACM Transactions on Quantum Computing, 4(1), 5.*](https://dl.acm.org/doi/10.1145/3548693)
* [Di, Y. M., & Wei, H. R. (2013). Synthesis of multivalued quantum logic circuits by elementary gates. *Physical Review A, 87(1), 012325.*](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.012325)
* [Shende, V. V., Bullock, S. S., & Markov, I. L. (2004). Synthesis of Quantum Logic Circuits. *arXiv preprint quant-ph/0406176.*](https://arxiv.org/abs/quant-ph/0406176)
* [Kuperberg, G. (2016). Breaking the cubic barrier in the Solovay-Kitaev algorithm. *arXiv preprint arXiv:2306.13158*.](https://arxiv.org/abs/2306.13158)
* [Stack Exchange: Cardinality of the Clifford Group](https://quantumcomputing.stackexchange.com/questions/13643)

## License

### Third-Party Libraries

* The core reinforcement learning framework is provided by **Stable-Baselines3**, which is distributed under the **MIT License**.
* The neural network models are built and trained using **PyTorch**, which is distributed under a permissive **BSD-style license**.
* Hyperparameter tuning is managed by **Optuna**, which is distributed under the **MIT License**.
* Performance benchmarking is conducted against **BQSKit**, which is distributed under the **BSD free software license**.

Please refer to the documentation of each respective project for their full license terms.
