# TenSafe Cookbook

The TenSafe Cookbook provides practical examples and utilities for fine-tuning language models using the TenSafe training API. It includes realistic examples and common abstractions to help you customize training environments.

## Overview

The cookbook is built on top of the TenSafe training API and provides:

- **Renderers**: Convert between tokens and structured chat messages for different model families
- **Completers**: Token and message-level sampling abstractions
- **Hyperparameter Utilities**: LoRA configuration and learning rate scaling
- **Evaluation**: Benchmark integrations and metrics
- **Recipes**: Complete training examples for various use cases

## Quick Start

```python
from tensafe.cookbook import get_tokenizer, get_lora_lr
from tensafe.cookbook.renderers import get_renderer, Llama3Renderer
from tensafe.cookbook.recipes import ChatSLConfig, run_chat_sl

# Get tokenizer and renderer for your model
tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
renderer = get_renderer("llama3", tokenizer)

# Configure training
config = ChatSLConfig(
    model_name="meta-llama/Llama-3.1-8B",
    dataset="HuggingFaceH4/no_robots",
    batch_size=128,
    learning_rate=2e-4,
)

# Run training
metrics = await run_chat_sl(config, client)
```

## Available Recipes

### 1. Chat Supervised Learning

Fine-tune models on conversational datasets using standard cross-entropy loss.

```python
from tensafe.cookbook.recipes import ChatSLConfig, run_chat_sl

config = ChatSLConfig(
    model_name="meta-llama/Llama-3.1-8B",
    dataset="HuggingFaceH4/no_robots",
    batch_size=128,
    learning_rate=2e-4,
    num_epochs=1,
)

metrics = await run_chat_sl(config, client)
```

### 2. Math Reasoning with RL

Improve mathematical problem-solving using GRPO-style reward centering.

```python
from tensafe.cookbook.recipes import MathRLConfig, run_math_rl

config = MathRLConfig(
    model_name="meta-llama/Llama-3.1-8B",
    dataset="gsm8k",
    batch_size=128,
    group_size=16,  # Responses per problem
    learning_rate=4e-5,
    max_steps=1000,
)

metrics = await run_math_rl(config, training_client, sampling_client)
```

### 3. Preference Learning (RLHF)

Three-stage pipeline: SFT → Reward Model → PPO/DPO.

```python
from tensafe.cookbook.recipes import PreferenceConfig, run_preference_learning

config = PreferenceConfig(
    model_name="meta-llama/Llama-3.1-8B",
    use_dpo=True,  # Use DPO instead of PPO
)

# Configure each stage
config.sft_config.dataset = "HuggingFaceH4/no_robots"
config.dpo_config.beta = 0.1

results = await run_preference_learning(config, client)
```

### 4. Tool Use Training

Train models to effectively use tools and APIs.

```python
from tensafe.cookbook.recipes import ToolUseConfig, run_tool_use

config = ToolUseConfig(
    model_name="meta-llama/Llama-3.1-8B",
    tools=[
        {"name": "search", "description": "Search the web", "parameters": {"query": {"type": "string"}}},
        {"name": "calculator", "description": "Perform calculations", "parameters": {"expression": {"type": "string"}}},
    ],
)

metrics = await run_tool_use(config, client)
```

### 5. Prompt Distillation

Internalize complex instructions into model weights.

```python
from tensafe.cookbook.recipes import DistillationConfig, run_distillation

config = DistillationConfig(
    model_name="meta-llama/Llama-3.1-8B",
    long_prompt="You are a helpful assistant specialized in...",  # Full system prompt
    short_prompt="Be helpful.",  # Shortened version
)

metrics = await run_distillation(config, training_client, sampling_client)
```

### 6. Multi-Agent Training

Train models through competitive or cooperative interactions.

```python
from tensafe.cookbook.recipes import MultiAgentConfig, GameType, run_multi_agent

config = MultiAgentConfig(
    model_name="meta-llama/Llama-3.1-8B",
    game_type=GameType.DEBATE,
    num_agents=2,
    max_turns=6,
    self_play=True,
)

metrics = await run_multi_agent(config, training_client, sampling_client)
```

## Renderers

Renderers handle token-level formatting for different model families:

| Renderer | Models | Features |
|----------|--------|----------|
| `Llama3Renderer` | Llama 3.x | Instruct format, EOT tokens |
| `DeepSeekV3Renderer` | DeepSeek V3 | Thinking mode support |
| `Qwen3Renderer` | Qwen 3.x | ChatML format |
| `KimiK2Renderer` | Kimi K2 | Thinking mode with `<think>` tags |
| `RoleColonRenderer` | Generic | Simple `Role: Content` format |

```python
from tensafe.cookbook.renderers import get_renderer, Llama3Renderer
from tensafe.cookbook import get_tokenizer

tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
renderer = get_renderer("llama3", tokenizer)

# Build generation prompt
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
]
tokens = renderer.build_generation_prompt(messages)

# Build supervised example with loss masking
messages_with_response = messages + [
    {"role": "assistant", "content": "Hi there!"}
]
tokens, weights = renderer.build_supervised_example(messages_with_response)
```

## Hyperparameter Utilities

Calculate optimal LoRA configurations:

```python
from tensafe.cookbook.hyperparam_utils import (
    LoRAConfig,
    get_lora_lr,
    get_lora_param_count,
    get_recommended_batch_size,
)

# Configure LoRA
lora_config = LoRAConfig(
    rank=32,
    alpha=64.0,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Get optimal learning rate
lr = get_lora_lr("meta-llama/Llama-3.1-8B", rank=32)

# Count trainable parameters
params = get_lora_param_count("meta-llama/Llama-3.1-8B", rank=32)

# Get recommended batch size
micro_batch, accum = get_recommended_batch_size("meta-llama/Llama-3.1-8B")
```

## Evaluation

Run benchmarks and compute metrics:

```python
from tensafe.cookbook.eval import Evaluator, EvaluatorConfig
from tensafe.cookbook.eval.metrics import accuracy, f1_score, bleu

# Configure evaluator
config = EvaluatorConfig(
    model_name="meta-llama/Llama-3.1-8B",
    benchmarks=["mmlu", "gsm8k"],
)

evaluator = Evaluator(config, sampling_client)
results = await evaluator.evaluate()

# Or compute metrics directly
acc = accuracy(predictions, references)
f1 = f1_score(predictions, references)
```

## Completers

Abstractions for sampling at different levels:

```python
from tensafe.cookbook.completers import (
    TinkerTokenCompleter,
    TinkerMessageCompleter,
    MockMessageCompleter,
)

# Token-level completion
token_completer = TinkerTokenCompleter(client, tokenizer)
result = await token_completer.complete(prompt_tokens, max_tokens=128)

# Message-level completion
message_completer = TinkerMessageCompleter(client, renderer)
response = await message_completer.complete(messages)

# Mock for testing
mock_completer = MockMessageCompleter(responses=["Test response"])
response = await mock_completer.complete(messages)
```

## Model Info

Get model metadata and recommendations:

```python
from tensafe.cookbook.model_info import (
    get_model_attributes,
    get_recommended_renderer_name,
    is_chat_model,
    get_context_length,
)

# Get model attributes
attrs = get_model_attributes("meta-llama/Llama-3.1-8B")
print(f"Context length: {attrs.context_length}")
print(f"Is instruct: {attrs.is_instruct}")

# Get recommended renderer
renderer_name = get_recommended_renderer_name("meta-llama/Llama-3.1-8B")  # "llama3"

# Check model capabilities
is_chat = is_chat_model("meta-llama/Llama-3.1-8B-Instruct")  # True
context = get_context_length("meta-llama/Llama-3.1-8B")  # 131072
```

## Integration with TenSafe

The cookbook integrates seamlessly with TenSafe's privacy features:

```python
from tg_tinker import ServiceClient
from tensafe.cookbook.recipes import ChatSLConfig, ChatSLTrainer

# Create TenSafe client with privacy settings
service = ServiceClient(
    api_key="your-key",
    dp_config={"epsilon": 8.0, "delta": 1e-5},
)

# Create training client
tc = service.create_training_client(
    model_ref="meta-llama/Llama-3.1-8B",
    lora_config={"rank": 32, "alpha": 64.0},
)

# Use cookbook trainer with TenSafe client
config = ChatSLConfig(
    model_name="meta-llama/Llama-3.1-8B",
    use_dp=True,  # Enable differential privacy
)

trainer = ChatSLTrainer(config, tc)
metrics = await trainer.train()
```

## Contributing

We welcome contributions! See the main CONTRIBUTING.md for guidelines.
