# Gemma3 270M from Scratch

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains a from-scratch implementation of a Small Language Model (SLM) inspired by Gemma3, with approximately 270 million parameters. The model is trained on the TinyStories dataset to generate creative and coherent short stories suitable for 3-4 year olds.

## Features
- **Modular Code Structure**: Separated into data preparation, model architecture, training, and inference modules.
- **RoPE Embeddings**: Supports Rotary Position Embeddings (RoPE) for both local (sliding window) and global attention.
- **Grouped Query Attention**: Efficient attention mechanism with query-key-value grouping.
- **Training Optimizations**: Includes gradient accumulation, mixed-precision training (bfloat16/float16), learning rate scheduling (warmup + cosine decay), and gradient clipping.
- **Inference**: Generate text from prompts using top-k sampling and temperature control.
- **Dataset**: Uses the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset from Hugging Face.

