import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_rope_params(head_dim, theta_base=10000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
        if self.shift is not None:
            out = out + self.shift.float()
        return out.to(input_dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, query_pre_attn_scalar=None, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        if head_dim is None:
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None
        self.scaling = (query_pre_attn_scalar or head_dim) ** -0.5

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm: queries = self.q_norm(queries)
        if self.k_norm: keys = self.k_norm(keys)
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        queries = queries * self.scaling
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = F.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg, attn_type):
        super().__init__()
        self.attn_type = attn_type
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"], num_heads=cfg["n_heads"], num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"], qk_norm=cfg["qk_norm"], query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["emb_dim"])
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"])
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"])
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"])

    def forward(self, x, mask_global, mask_local, cos_global, sin_global, cos_local, sin_local):
        shortcut = x
        x = self.input_layernorm(x)
        attn_mask = mask_local if self.attn_type == "sliding_attention" else mask_global
        cos, sin = (cos_local, sin_local) if self.attn_type == "sliding_attention" else (cos_global, sin_global)
        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x

class Gemma3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.blocks = nn.ModuleList([TransformerBlock(cfg, attn_type) for attn_type in cfg["layer_types"]])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        self.cfg = cfg
        cos_local, sin_local = compute_rope_params(cfg["head_dim"], cfg["rope_local_base"], cfg["context_length"])
        cos_global, sin_global = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        mask_global = torch.triu(ones, diagonal=1)
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)
        for block in self.blocks:
            x = block(x, mask_global, mask_local, self.cos_global, self.sin_global, self.cos_local, self.sin_local)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            ctx_len = self.cfg["context_length"]
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx