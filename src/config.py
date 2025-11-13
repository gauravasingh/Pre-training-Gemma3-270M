import torch

# Model configuration for Gemma3 270M
GEMMA3_CONFIG_270M = {
    "vocab_size": 50257,
    "context_length": 32768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10000.0,
    "rope_base": 1000000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}

# Training configuration
TRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "max_iters": 150000,
    "warmup_steps": 1000,
    "min_lr": 5e-4,
    "eval_iters": 500,
    "batch_size": 32,
    "block_size": 128,
    "gradient_accumulation_steps": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
}