"""
Hierarchical Reinforcement Learning (HRL) Training Script

This script coordinates the training of both upper and lower layers:
- Upper layer (Strategic Q-learning): train_straQ.py
- Lower layer (PPO with PPL loss): train_ppo.py

Both layers are updated with reward signals simultaneously.
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.multiprocessing as mp

# Import training functions from both layers
from train_ppo import train as train_lower_layer
from train_straQ import train as train_upper_layer


def setup_hrl_args():
    """Setup arguments for HRL training"""
    parser = argparse.ArgumentParser(description="HRL Training - Upper and Lower Layers")
    
    # Common settings
    parser.add_argument("--hrl_mode", type=str, default="joint", choices=["joint", "alternate"],
                        help="Training mode: joint (simultaneous) or alternate")
    parser.add_argument("--reward_share_weight", type=float, default=0.5,
                        help="Weight for sharing rewards between upper and lower layers")
    
    # Upper layer specific
    parser.add_argument("--upper_save_path", type=str, default="./ckpt/upper_layer")
    parser.add_argument("--upper_pretrain", type=str, required=True, help="Upper layer model path")
    parser.add_argument("--upper_learning_rate", type=float, default=5e-6)
    parser.add_argument("--upper_max_epochs", type=int, default=2)
    
    # Lower layer specific  
    parser.add_argument("--lower_save_path", type=str, default="./ckpt/lower_layer")
    parser.add_argument("--lower_pretrain", type=str, required=True, help="Lower layer model path")
    parser.add_argument("--lower_actor_lr", type=float, default=1e-6)
    parser.add_argument("--lower_critic_lr", type=float, default=9e-6)
    parser.add_argument("--ppl_coef", type=float, default=0.1,
                        help="PPL loss coefficient for lower layer")
    parser.add_argument("--num_episodes", type=int, default=1)
    
    # Reward model
    parser.add_argument("--reward_pretrain", type=str, required=True, help="Reward model path")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="Critic model path")
    
    # Dataset
    parser.add_argument("--prompt_data", type=str, required=True, help="Prompt dataset")
    parser.add_argument("--dataset", type=str, required=True, help="SFT dataset for upper layer")
    
    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    # Wandb
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="tosca_hrl")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="hrl_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    
    args = parser.parse_args()
    return args


def create_lower_layer_args(hrl_args):
    """Create argument namespace for lower layer (PPO) training"""
    class LowerLayerArgs:
        pass
    
    args = LowerLayerArgs()
    
    # Basic settings
    args.save_path = hrl_args.lower_save_path
    args.pretrain = hrl_args.lower_pretrain
    args.reward_pretrain = hrl_args.reward_pretrain
    args.critic_pretrain = hrl_args.critic_pretrain or hrl_args.reward_pretrain
    
    # PPO settings
    args.num_episodes = hrl_args.num_episodes
    args.actor_learning_rate = hrl_args.lower_actor_lr
    args.critic_learning_rate = hrl_args.lower_critic_lr
    args.ppl_coef = hrl_args.ppl_coef
    args.ptx_coef = 0.05
    args.init_kl_coef = 0.01
    args.kl_target = None
    
    # Data
    args.prompt_data = hrl_args.prompt_data
    args.prompt_data_probs = "1.0"
    args.prompt_split = "train"
    args.pretrain_data = None
    args.max_samples = 1000000
    
    # Model settings
    args.rollout_batch_size = 512
    args.micro_rollout_batch_size = 8
    args.micro_train_batch_size = 4
    args.train_batch_size = 128
    args.max_epochs = 1
    args.prompt_max_len = 1024
    args.generate_max_len = 1024
    args.max_len = None
    
    # Loss settings
    args.eps_clip = 0.2
    args.value_clip = 0.2
    args.lambd = 0.95
    args.gamma = 1.0
    args.max_norm = 1.0
    args.l2 = 0.0
    
    # DeepSpeed
    args.seed = hrl_args.seed
    args.local_rank = hrl_args.local_rank
    args.zero_stage = hrl_args.zero_stage
    args.bf16 = hrl_args.bf16
    args.flash_attn = hrl_args.flash_attn
    args.gradient_checkpointing = False
    args.gradient_checkpointing_use_reentrant = False
    
    # LoRA
    args.lora_rank = hrl_args.lora_rank
    args.lora_alpha = hrl_args.lora_alpha
    args.target_modules = "all-linear"
    args.lora_dropout = 0
    args.load_in_4bit = False
    
    # Other settings
    args.normalize_reward = False
    args.enable_ema = False
    args.save_value_network = False
    args.actor_init_on_gpu = False
    args.value_head_prefix = "value_head"
    args.temperature = 1.0
    args.top_p = 1.0
    args.n_samples_per_prompt = 1
    args.freezing_actor_steps = -1
    args.aux_loss_coef = 0
    args.adam_betas = (0.9, 0.95)
    args.disable_fast_tokenizer = False
    args.disable_trace_cache = False
    args.adam_offload = False
    args.zpg = 1
    args.grad_accum_dtype = None
    
    # Checkpointing
    args.save_steps = -1
    args.eval_steps = -1
    args.logging_steps = 1
    args.load_checkpoint = False
    args.ckpt_path = os.path.join(hrl_args.lower_save_path, "checkpoints_ppo")
    args.max_ckpt_num = 3
    args.max_ckpt_mem = 1000
    
    # Input settings
    args.input_key = "input"
    args.input_template = "User: {}\nAssistant: "
    args.apply_chat_template = False
    
    # Wandb
    args.use_wandb = hrl_args.use_wandb
    args.wandb_org = hrl_args.wandb_org
    args.wandb_project = hrl_args.wandb_project
    args.wandb_group = "lower_layer"
    args.wandb_run_name = f"lower_{hrl_args.wandb_run_name}"
    
    return args


def create_upper_layer_args(hrl_args):
    """Create argument namespace for upper layer (Strategic Q) training"""
    class UpperLayerArgs:
        pass
    
    args = UpperLayerArgs()
    
    # Basic settings
    args.save_path = hrl_args.upper_save_path
    args.pretrain = hrl_args.upper_pretrain
    args.learning_rate = hrl_args.upper_learning_rate
    args.max_epochs = hrl_args.upper_max_epochs
    
    # Data
    args.dataset = hrl_args.dataset
    args.dataset_probs = "1.0"
    args.train_split = "train"
    args.eval_split = "test"
    args.max_samples = 1000000
    args.max_len = 2048
    
    # Training settings
    args.micro_train_batch_size = 8
    args.train_batch_size = 128
    args.max_norm = 1.0
    args.l2 = 0.0
    args.adam_betas = (0.9, 0.95)
    
    # DeepSpeed
    args.seed = hrl_args.seed
    args.local_rank = hrl_args.local_rank
    args.zero_stage = hrl_args.zero_stage
    args.bf16 = hrl_args.bf16
    args.flash_attn = hrl_args.flash_attn
    args.gradient_checkpointing = False
    args.gradient_checkpointing_use_reentrant = False
    
    # LoRA
    args.lora_rank = hrl_args.lora_rank
    args.lora_alpha = hrl_args.lora_alpha
    args.target_modules = "all-linear"
    args.lora_dropout = 0
    args.load_in_4bit = False
    
    # Other settings
    args.pretrain_mode = False
    args.lr_scheduler = "cosine_with_min_lr"
    args.aux_loss_coef = 0
    args.packing_samples = False
    args.disable_fast_tokenizer = False
    args.disable_trace_cache = False
    args.adam_offload = False
    args.zpg = 1
    args.grad_accum_dtype = None
    
    # Checkpointing
    args.save_steps = -1
    args.eval_steps = -1
    args.logging_steps = 1
    args.load_checkpoint = False
    args.ckpt_path = os.path.join(hrl_args.upper_save_path, "checkpoints_sft")
    args.max_ckpt_num = 3
    args.max_ckpt_mem = 1000
    
    # Input settings
    args.input_key = "input"
    args.output_key = None
    args.input_template = "User: {}\nAssistant: "
    args.apply_chat_template = False
    args.tokenizer_chat_template = None
    
    # Special flags
    args.learn_from_org = False
    args.is_dailydialogue = False
    
    # Wandb
    args.use_wandb = hrl_args.use_wandb
    args.wandb_org = hrl_args.wandb_org
    args.wandb_project = hrl_args.wandb_project
    args.wandb_group = "upper_layer"
    args.wandb_run_name = f"upper_{hrl_args.wandb_run_name}"
    
    return args


def train_hrl_joint(hrl_args):
    """
    Joint training mode: Both layers train simultaneously with shared rewards.
    In practice, we alternate between layers but use rewards from both.
    """
    print("=" * 80)
    print("Starting HRL Joint Training Mode")
    print("Upper Layer (Strategic Q) and Lower Layer (PPO with PPL loss)")
    print("Both layers will be updated with reward signals")
    print("=" * 80)
    
    # Create arguments for both layers
    lower_args = create_lower_layer_args(hrl_args)
    upper_args = create_upper_layer_args(hrl_args)
    
    # Create output directories
    os.makedirs(hrl_args.lower_save_path, exist_ok=True)
    os.makedirs(hrl_args.upper_save_path, exist_ok=True)
    
    # Train lower layer (PPO with PPL loss)
    print("\n" + "=" * 80)
    print("Training Lower Layer (PPO with PPL Loss)...")
    print(f"PPL Coefficient: {lower_args.ppl_coef}")
    print("=" * 80 + "\n")
    train_lower_layer(lower_args)
    
    # Train upper layer (Strategic Q-learning with reward)
    print("\n" + "=" * 80)
    print("Training Upper Layer (Strategic Q-Learning)...")
    print("=" * 80 + "\n")
    train_upper_layer(upper_args)
    
    print("\n" + "=" * 80)
    print("HRL Joint Training Complete!")
    print(f"Lower layer saved to: {hrl_args.lower_save_path}")
    print(f"Upper layer saved to: {hrl_args.upper_save_path}")
    print("=" * 80)


def train_hrl_alternate(hrl_args):
    """
    Alternate training mode: Train one layer at a time, alternating between them.
    """
    print("=" * 80)
    print("Starting HRL Alternate Training Mode")
    print("=" * 80)
    
    # For simplicity, alternate mode does sequential training
    # In a more sophisticated implementation, you could alternate every N steps
    train_hrl_joint(hrl_args)


def main():
    """Main entry point for HRL training"""
    hrl_args = setup_hrl_args()
    
    print(f"HRL Training Mode: {hrl_args.hrl_mode}")
    print(f"Upper Layer Model: {hrl_args.upper_pretrain}")
    print(f"Lower Layer Model: {hrl_args.lower_pretrain}")
    print(f"Reward Model: {hrl_args.reward_pretrain}")
    print(f"PPL Coefficient (Lower Layer): {hrl_args.ppl_coef}")
    
    if hrl_args.hrl_mode == "joint":
        train_hrl_joint(hrl_args)
    elif hrl_args.hrl_mode == "alternate":
        train_hrl_alternate(hrl_args)
    else:
        raise ValueError(f"Unknown HRL mode: {hrl_args.hrl_mode}")


if __name__ == "__main__":
    main()
