# Implementations of Various LLM Technologies

This repository contains implementations of various building blocks for Large Language Models (LLMs).

## Files:

### Core Modules

*   `mha.py`: Multi-Head Attention (with cross-attention variant).
*   `res_net.py`: ResNet Module, used by the classical encoder-decoder transformer model.
*   `encoder.py`: Classical Transformer Encoder module.
*   `decoder.py`: Classical Transformer Decoder module.
*   `feed_forward.py`: Position-wise Feed-Forward Network.
*   `device.py`: Sets up the torch device for training (CPU or GPU).

### Mixture of Experts (MoE)

*   `moe/classic_moe.py`: A classic Mixture of Experts (MoE) implementation.

### Positional Encoding

*   `positional_encoding/rope.py`: Rotary Positional Encoding (RoPE).
*   `positional_encoding/sinusoidal.py`: Sinusoidal Positional Encoding.

### Configuration

*   `config.py`: Contains dataclasses for configuring different model architectures.
*   `shared_config.py`: Defines a shared configuration dataclass for model parameters like `d_model` and `vocab_size`.

### Training

*   `training/basic_trainer.py`: A basic training loop for the models.

## Getting Started

Get started by customizing your environment (defined in the `.idx/dev.nix` file) with the tools and IDE extensions you'll need for your project!

Learn more at https://developers.google.com/idx/guides/customize-idx-env
