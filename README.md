# ToSCA

ToSCA is a Hierarchical Reinforcement Learning (HRL) framework built on top of OpenRLHF 0.3.8.

## Overview

This repository implements a two-level hierarchical reinforcement learning architecture:

- **High-level (Upper layer)**: [`cli/train_straQ.py`](cli/train_straQ.py) - Strategic Q-learning trainer
- **Low-level (Lower layer)**: [`cli/train_ppo.py`](cli/train_ppo.py) - PPO trainer with enhancements

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

#### 1. Lower Layer Training Only (PPO)

Train the lower layer with PPO:

```bash
deepspeed --module cli.train_ppo \
    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --reward_pretrain your-reward-model-path \
    --critic_pretrain your-critic-model-path \
    --prompt_data your-prompt-dataset \
    --save_path ./ckpt/lower_layer \
    --num_episodes 1 \
    --rollout_batch_size 512 \
    --micro_rollout_batch_size 8 \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --actor_learning_rate 9e-7 \
    --critic_learning_rate 9e-4 \
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

#### 2. Upper Layer Training Only (Strategic Q-Learning)

Train the upper layer with Q-learning:

```bash
deepspeed --module cli.train_straQ \
    --pretrain meta-llama/meta-llama/Llama-3.2-1B-Instruct \
    --dataset your-q-dataset \
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
    --upper_pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --lower_pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --reward_pretrain your-reward-model-path \
    --prompt_data your-prompt-dataset \
    --dataset your-sft-dataset \
    --upper_save_path ./ckpt/upper_layer \
    --lower_save_path ./ckpt/lower_layer \
    --hrl_mode joint \
    --ppl_coef 0.01 \
    --num_episodes 2 \
    --upper_max_epochs 2 \
    --upper_learning_rate 5e-6 \
    --lower_actor_lr 9e-7 \
    --lower_critic_lr 9e-4 \
    --zero_stage 3 \
    --bf16 \
    --flash_attn \
    --lora_rank 8 \
    --lora_alpha 16
```


#### 4. Using LoRA for Efficient Training

All training scripts support LoRA for parameter-efficient fine-tuning:

```bash
# Add these flags to any training command
--lora_rank 8 \
--lora_alpha 16 \
--target_modules all-linear \
--lora_dropout 0.05
```


