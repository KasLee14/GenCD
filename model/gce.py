# model/gce.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCEBlock(nn.Module):
    def __init__(self, embedding_size, dropout_ratio):
        super(GCEBlock, self).__init__()
        self.embedding_size = embedding_size
        self.qkvu_proj = nn.Linear(embedding_size, 4 * embedding_size)
        self.attn_mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 2),
            nn.SiLU(),
            nn.Linear(embedding_size * 2, embedding_size),
        )
        self.out_proj = nn.Linear(embedding_size, embedding_size)
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, hidden_states, attention_bias, attention_mask):
        norm_hidden = self.layer_norm_1(hidden_states)

        qkvu = self.qkvu_proj(norm_hidden)
        query, key, value, gate = qkvu.chunk(4, dim=-1)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.embedding_size)
        scores = scores + attention_bias
        scores = F.silu(scores)
        scores = scores.masked_fill(attention_mask, 0.0)

        updated_value = torch.matmul(scores, value)
        gated_value = self.layer_norm_2(updated_value * gate)
        gated_out = self.attn_mlp(gated_value)

        return hidden_states + self.dropout(self.out_proj(gated_out))


class GCE(nn.Module):
    def __init__(self, config):
        super(GCE, self).__init__()
        self.embedding_size = config.embedding_size
        self.max_seq_len = config.max_seq_len
        self.total_seq_len = 1 + self.max_seq_len + 1
        self.num_block = int(getattr(config, "num_block", 1))
        if self.num_block < 0:
            raise ValueError(f"num_block must be non-negative, got {self.num_block}.")

        self.response_embedding = nn.Embedding(3, self.embedding_size)
        self.rel_pos_bias = nn.Embedding(2 * self.total_seq_len + 1, 1)
        self.time_bias_proj = nn.Linear(1, 1)

        rel_pos_index = torch.arange(self.total_seq_len).unsqueeze(1) - torch.arange(self.total_seq_len).unsqueeze(0)
        rel_pos_index = rel_pos_index + self.total_seq_len
        self.register_buffer("rel_pos_index", rel_pos_index, persistent=False)

        self.blocks = nn.ModuleList(
            [GCEBlock(self.embedding_size, config.dropout_ratio) for _ in range(self.num_block)]
        )

    def forward(self, user_emb, seq_item_emb, seq_score, seq_item_id, target_item_emb, seq_time=None):
        batch_size = user_emb.size(0)
        device = user_emb.device

        seq_score_idx = (seq_score + 1.0).long()
        resp_emb = self.response_embedding(seq_score_idx)
        seq_fused_emb = seq_item_emb + resp_emb

        user_emb = user_emb.unsqueeze(1)
        target_item_emb = target_item_emb.unsqueeze(1)
        combined_seq = torch.cat([user_emb, seq_fused_emb, target_item_emb], dim=1)

        seq_mask = seq_item_id == 0
        false_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        combined_mask = torch.cat([false_mask, seq_mask, false_mask], dim=1)

        attention_bias = self.rel_pos_bias(self.rel_pos_index).squeeze(-1).unsqueeze(0)
        if seq_time is not None:
            time_diff = seq_time.unsqueeze(2) - seq_time.unsqueeze(1)
            time_bias = self.time_bias_proj(time_diff.unsqueeze(-1)).squeeze(-1)
            attention_bias = attention_bias + time_bias

        causal_mask = torch.triu(
            torch.ones(self.total_seq_len, self.total_seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        attention_mask = causal_mask.unsqueeze(0) | combined_mask.unsqueeze(1)

        hidden_states = combined_seq
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_bias, attention_mask)

        final_target_rep = hidden_states[:, -1, :]
        seq_reps = hidden_states[:, 1:-1, :]
        return final_target_rep, seq_reps
