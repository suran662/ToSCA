# ToSCA

ToSCA is a Hierarchical Reinforcement Learning (HRL) framework built on top of OpenRLHF 0.3.8.

## Overview

This repository implements a two-level hierarchical reinforcement learning architecture:

- **High-level (Upper layer)**: [`cli/train_straQ.py`](cli/train_straQ.py) - Strategic Q-learning trainer
- **Low-level (Lower layer)**: [`cli/train_ppo.py`](cli/train_ppo.py) - PPO trainer with enhancements

## Key Features

### Hierarchical Reinforcement Learning (HRL)
- Simultaneous reward-based updates for both upper and lower layers
- Two-tier decision making for more sophisticated policy learning

### Enhanced PPO with PPL Loss
The lower layer trainer incorporates a novel PPL (perplexity) loss component:
- In addition to standard reward and KL divergence, the loss includes: **log(∏ P(token)) × coefficient**
- This term is the logarithm of the product of token probabilities, scaled by a coefficient
- Helps maintain language modeling quality during RL fine-tuning

## Installation

This project is built upon **OpenRLHF 0.3.8**. To use ToSCA:

1. First, set up and verify OpenRLHF 0.3.8:
```bash
pip install openrlhf==0.3.8
```

2. Test the base OpenRLHF installation to ensure it works correctly

3. Replace the relevant files in your OpenRLHF installation with the files from this repository

## Project Structure

- `cli/` - Command-line training scripts
  - `train_straQ.py` - High-level strategic trainer
  - `train_ppo.py` - Low-level PPO trainer with PPL loss
  - `interactive_chat.py` - Interactive chat interface
- `models/` - Model architecture definitions
- `Trainer/` - Training logic and utilities
  - `ppo_trainer.py` - PPO training implementation
  - `q_stra_trainer.py` - Strategic Q-learning implementation
  - `ppo_utils/` - Experience replay, KL control, etc.
- `datasets/` - Dataset utilities (see its README for details)
- `utils/` - Distributed training, logging, and helper functions

## Usage

### Prerequisites

Ensure you have installed and tested OpenRLHF 0.3.8 as described in the Installation section.

### Training Scripts

ToSCA provides three main training scripts:

1. **Lower Layer Training (PPO with PPL Loss)**: [`cli/train_ppo.py`](cli/train_ppo.py)
2. **Upper Layer Training (Strategic Q-Learning)**: [`cli/train_straQ.py`](cli/train_straQ.py)
3. **HRL Joint Training**: [`cli/train_hrl.py`](cli/train_hrl.py)

### Training Examples

#### 1. Lower Layer Training Only (PPO with PPL Loss)

Train the lower layer with PPO and perplexity loss:

```bash
deepspeed --module cli.train_ppo \
    --pretrain meta-llama/Llama-2-7b-hf \
    --reward_pretrain your-reward-model-path \
    --critic_pretrain your-critic-model-path \
    --prompt_data your-prompt-dataset \
    --save_path ./ckpt/lower_layer \
    --num_episodes 1 \
    --rollout_batch_size 512 \
    --micro_rollout_batch_size 8 \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --ppl_coef 0.1 \
    --ptx_coef 0.05 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing
```

**Key Parameters:**
- `--ppl_coef`: Coefficient for PPL loss (log(∏ P(token))). Default: 0.0 (disabled). Recommended: 0.05-0.2
- `--ptx_coef`: Coefficient for pretrain loss
- `--init_kl_coef`: Initial KL divergence coefficient

#### 2. Upper Layer Training Only (Strategic Q-Learning)

Train the upper layer with Q-learning:

```bash
deepspeed --module cli.train_straQ \
    --pretrain meta-llama/Llama-2-7b-hf \
    --dataset your-sft-dataset \
    --save_path ./ckpt/upper_layer \
    --max_epochs 2 \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --learning_rate 5e-6 \
    --max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn
```

#### 3. HRL Joint Training (Recommended)

Train both layers simultaneously with shared rewards:

```bash
deepspeed --module cli.train_hrl \
    --upper_pretrain meta-llama/Llama-2-7b-hf \
    --lower_pretrain meta-llama/Llama-2-7b-hf \
    --reward_pretrain your-reward-model-path \
    --prompt_data your-prompt-dataset \
    --dataset your-sft-dataset \
    --upper_save_path ./ckpt/upper_layer \
    --lower_save_path ./ckpt/lower_layer \
    --hrl_mode joint \
    --ppl_coef 0.1 \
    --reward_share_weight 0.5 \
    --num_episodes 1 \
    --upper_max_epochs 2 \
    --upper_learning_rate 5e-6 \
    --lower_actor_lr 1e-6 \
    --lower_critic_lr 9e-6 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --lora_rank 8 \
    --lora_alpha 16
```

**HRL-Specific Parameters:**
- `--hrl_mode`: Training mode (`joint` or `alternate`)
- `--reward_share_weight`: Weight for sharing rewards between layers (0.0-1.0)
- `--ppl_coef`: PPL loss coefficient for lower layer

#### 4. Using LoRA for Efficient Training

All training scripts support LoRA for parameter-efficient fine-tuning:

```bash
# Add these flags to any training command
--lora_rank 8 \
--lora_alpha 16 \
--target_modules all-linear \
--lora_dropout 0.05
```

#### 5. Training with Weights & Biases Logging

Enable W&B logging for experiment tracking:

```bash
# Add these flags to any training command
--use_wandb YOUR_WANDB_API_KEY \
--wandb_org YOUR_ORG \
--wandb_project tosca \
--wandb_run_name experiment_name
```

### Interactive Chat

After training, use the interactive chat interface:

```bash
python cli/interactive_chat.py \
    --model_path ./ckpt/lower_layer \
    --max_new_tokens 512
```

## Advanced Configuration

### PPL Loss Explained

The PPL (Perplexity) loss in the lower layer is calculated as:

$$\text{PPL Loss} = -\text{coef} \times \sum_{t=1}^{T} \log P(token_t)$$

This is equivalent to:

$$\text{PPL Loss} = -\text{coef} \times \log \prod_{t=1}^{T} P(token_t)$$

Where:
- $P(token_t)$ is the probability of token $t$
- `coef` is controlled by `--ppl_coef` parameter
- This term helps maintain language modeling quality during RL fine-tuning
