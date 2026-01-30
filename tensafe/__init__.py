"""
TenSafe - Homomorphically Encrypted LoRA Adaptation

This package provides extensions for TenSafe training including:
- Pluggable loss functions
- RLVR (Reinforcement Learning with Verifiable Rewards)
- Benchmark and evaluation tools
- Cookbook: Practical examples and utilities for fine-tuning LLMs
  - Renderers for different model families (Llama3, DeepSeek, Qwen, etc.)
  - Completers for token and message-level sampling
  - Hyperparameter utilities for LoRA configuration
  - Training recipes for common use cases:
    - Chat supervised learning
    - Math reasoning with RL
    - Preference learning (RLHF/DPO)
    - Tool use training
    - Prompt distillation
    - Multi-agent training
  - Evaluation abstractions and benchmark integrations
"""

__version__ = "0.1.0"
