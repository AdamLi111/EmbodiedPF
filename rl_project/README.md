# RL Friction Policy for PONDER

Reinforcement-learning pipeline that learns **when** and **which type** of communicative friction to apply during human-robot dialogue, built on top of the existing PONDER simulation framework.

## Overview

In embodied human-robot interaction, blindly executing ambiguous commands leads to errors and safety violations. *Positive friction* — brief dialogue acts such as probing questions, assumption reveals, or overspecifications — can resolve ambiguity before the robot acts. This project trains an RL agent (PPO) to choose the optimal friction type (or to execute immediately) at each dialogue turn, maximising task success while minimising unnecessary delays.

The pipeline **wraps** existing PONDER classes (`WorldModel`, `SimulatedUser`, `TaskEvaluator`, etc.) inside a Gymnasium environment. No existing source code is modified.

## Directory Structure

```
rl_project/
├── config/
│   └── default.yaml            # All hyperparameters
├── envs/
│   └── friction_env.py         # Gymnasium env wrapping PONDER simulation
├── prompts/                    # LLM prompt scripts, one per friction type
│   ├── base_prompt.py
│   ├── execute_prompt.py       # action 0 — physical execution
│   ├── probing_prompt.py       # action 1
│   ├── assumption_reveal_prompt.py  # action 2
│   ├── overspecification_prompt.py  # action 3
│   ├── reflective_pause_prompt.py   # action 4
│   └── reinforcement_prompt.py      # action 5
├── models/
│   ├── state_encoder.py        # Sentence-transformer text encoder
│   ├── flat_policy.py          # 6-way MLP policy
│   ├── hierarchical_policy.py  # Gate (exec/friction) + selector (5 types)
│   └── value_network.py        # Scalar value MLP
├── agents/
│   ├── ppo_agent.py            # PPO update logic
│   └── rollout_buffer.py       # GAE rollout storage
├── training/
│   ├── reward.py               # Reward shaping from env info
│   └── trainer.py              # Main training loop
├── evaluation/
│   ├── evaluator.py            # Greedy rollout evaluator
│   └── metrics.py              # Per-episode metrics tracker
├── utils/
│   ├── helpers.py              # Seed, device, config loading
│   └── logger.py               # TensorBoard + console logger
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation entry point
│   └── compare.py              # Multi-policy comparison + plots
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r rl_project/requirements.txt
```

## Action Space

| ID | Name | Description |
|----|------|-------------|
| 0 | execute | Parse command into physical action and execute immediately |
| 1 | probe | Ask a targeted clarifying question |
| 2 | assumption_reveal | State the robot's assumption explicitly |
| 3 | overspecification | Confirm plan with extra confirmatory detail |
| 4 | reflective_pause | Verbally signal uncertainty / re-assessment |
| 5 | reinforcement | Restate key information for emphasis |

## Training

```bash
python -m rl_project.scripts.train \
    --config rl_project/config/default.yaml \
    --policy hierarchical

# Quick test run (1000 steps):
python -m rl_project.scripts.train \
    --config rl_project/config/default.yaml \
    --policy hierarchical \
    --total_timesteps 1000
```

Options:
- `--policy flat|hierarchical` — flat 6-way MLP or hierarchical gate+selector
- `--use_encoder` — load sentence-transformer for real text embeddings (slower)
- `--openai_api_key <key>` — use real LLM prompts instead of rule-based placeholders
- `--total_timesteps N` — override training length

## Evaluation

```bash
python -m rl_project.scripts.evaluate \
    --checkpoint checkpoints/checkpoint_final.pt \
    --policy hierarchical \
    --num_episodes 500
```

## Comparison

```bash
python -m rl_project.scripts.compare \
    --hierarchical_checkpoint checkpoints/hier_final.pt \
    --flat_checkpoint checkpoints/flat_final.pt \
    --num_episodes 500
```

Generates plots in `rl_project/results/`:
- `success_by_ambiguity.png` — grouped bar chart by ambiguity type
- `avg_turns.png` — average episode length
- `friction_distribution.png` — pie charts of friction type usage

## Configuration

All hyperparameters are in `config/default.yaml`:

| Section | Key parameters |
|---------|---------------|
| `env` | `max_turns` |
| `reward` | `task_success`, `task_failure`, `safety_violation`, `per_turn_penalty`, ... |
| `encoder` | `model_name`, `embedding_dim`, `freeze` |
| `policy` | `hidden_dims`, `activation`, `dropout` |
| `ppo` | `lr`, `gamma`, `gae_lambda`, `clip_epsilon`, `entropy_coef`, `update_epochs`, ... |
| `training` | `log_interval`, `eval_interval`, `save_interval`, `checkpoint_dir` |
