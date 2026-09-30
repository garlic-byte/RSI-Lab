"""Pointwise Transformer velocity field using position, time and class tokens."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from config.train_config import ModelConfig


def manual_multi_head_attention(tokens: torch.Tensor, attention: nn.MultiheadAttention) -> torch.Tensor:
    """练习入口：自己实现多头自注意力，暂不需要 mask、dropout 或 KV cache。

    Args:
        tokens: [B, T, D]，B 个独立样本，每个样本 T 个 token。
            本模型 T=3，但实现应支持任意 T。不要在 B 维度计算注意力。
        attention: 提供已初始化的参数，便于与 PyTorch 使用相同权重对照。
            attention.num_heads: 头数 H，D 必须能被 H 整除。
            attention.in_proj_weight: [3*D, D]，按 Q、K、V 顺序排列。
            attention.in_proj_bias: [3*D]，顺序同上，也可能为 None。
            attention.out_proj.weight: [D, D]。
            attention.out_proj.bias: [D]，也可能为 None。

    Returns:
        [B, T, D]，完成多头注意力和输出投影后的结果。
        不包含残差连接、LayerNorm 和前馈网络，外部 EncoderLayer 会处理。

    允许使用 F.linear、reshape、transpose、matmul、softmax 等基础算子。
    请复用传入的参数，不要在这里创建新的 Linear 或 Parameter。
    不要直接调用 attention(...) 或 scaled_dot_product_attention 来代替实现。

    写完后在项目根目录运行：
        /Users/garlic/Desktop/venv/torch/bin/python run_check_attention.py
    检查通过后，可在模型构建后调用 model.set_manual_attention(True) 接入训练。
    """
    # TODO 1: 对输入进行 Q、K、V 投影。
    # TODO 2: 拆分 H 个头，理清每个维度代表什么。
    # TODO 3: 计算缩放后的注意力分数，在正确的维度归一化。
    # TODO 4: 用注意力权重聚合 V，再合并各个头。
    # TODO 5: 完成输出投影，返回 [B, T, D]。
    batch_size, seq_len, hidden_dim = tokens.shape
    assert hidden_dim % attention.num_heads == 0
    head_dim = hidden_dim // attention.num_heads
    W_q, W_k, W_v = attention.in_proj_weight.chunk(3, dim=0)
    if attention.in_proj_bias is not None:
        b_q, b_k, b_v = attention.in_proj_bias.chunk(3, dim=0)
    else:
        b_q = b_k = b_v =None
    # shape [B, H, N, Dh]
    Q = F.linear(tokens, W_q, b_q).reshape(batch_size, seq_len, attention.num_heads, head_dim).permute(0, 2, 1, 3)
    K = F.linear(tokens, W_k, b_k).reshape(batch_size, seq_len, attention.num_heads, head_dim).permute(0, 2, 1, 3)
    V = F.linear(tokens, W_v, b_v).reshape(batch_size, seq_len, attention.num_heads, head_dim).permute(0, 2, 1, 3)
    attention_score = (Q @ K.transpose(-1, -2)) * (head_dim ** -0.5)
    attention_score = (attention_score.softmax(dim=-1) @ V).permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
    return F.linear(attention_score, attention.out_proj.weight, attention.out_proj.bias)




class PracticeEncoderLayer(nn.TransformerEncoderLayer):
    """保留原始参数和检查点格式，仅为自注意力提供可选的练习入口。"""

    use_manual_attention: bool = False

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # 当前模型使用 SiLU，不走 PyTorch 的 fused Encoder 快速路径，
        # 所以训练和 eval 均会进入此处。修改激活函数时需重新验证这一点。
        if not self.use_manual_attention:
            return super()._sa_block(x, attn_mask, key_padding_mask, is_causal=is_causal)
        if attn_mask is not None or key_padding_mask is not None or is_causal:
            raise ValueError("手写注意力练习暂不支持 mask 或 causal attention")
        return self.dropout1(manual_multi_head_attention(x, self.self_attn))


class VelocityTransformer(nn.Module):
    """Attend over three feature tokens per point, never across batch samples.

    Fourier features and label embeddings match the MLP input information.
    Separate projections and learned token types identify position/time/class.
    The position token predicts the two-dimensional velocity after attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.class_embedding = nn.Embedding(config.num_classes, config.embedding_dim)
        self.register_buffer("frequencies", torch.arange(1, config.time_frequencies + 1).float() * math.pi)
        self.register_buffer("spatial_frequencies", torch.arange(1, config.spatial_frequencies + 1).float() * math.pi)
        width = config.transformer_dim
        self.position_projection = nn.Linear(2 + 4 * config.spatial_frequencies, width)
        self.time_projection = nn.Linear(1 + 2 * config.time_frequencies, width)
        self.class_projection = nn.Linear(config.embedding_dim, width)
        self.token_types = nn.Parameter(torch.empty(1, 3, width))
        nn.init.normal_(self.token_types, std=0.02)
        # Independently initialized blocks; dropout is disabled so ODE sampling
        # and same-point predictions remain deterministic.
        self.blocks = nn.ModuleList([
            PracticeEncoderLayer(d_model=width, nhead=config.num_heads,
                                       dim_feedforward=config.hidden_dim, dropout=0.,
                                       activation=F.silu, batch_first=True, norm_first=True)
            for _ in range(config.num_layers)
        ])
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, 2)

    def set_manual_attention(self, enabled: bool = True) -> None:
        """切换所有层的注意力实现；默认关闭，此开关不写入 checkpoint。

        开启前请完成上方 TODO；重新加载模型后需要再次显式开启。
        """
        for block in self.blocks:
            block.use_manual_attention = enabled

    def forward(self, points: torch.Tensor, time: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Map [B,2], [B,1], [B] into [B,2] via [B,3,D] attention."""
        spatial_phase = (points[:, :, None] * self.spatial_frequencies).flatten(1)
        time_phase = time * self.frequencies
        position = self.position_projection(torch.cat((points, spatial_phase.sin(), spatial_phase.cos()), -1))
        time_token = self.time_projection(torch.cat((time, time_phase.sin(), time_phase.cos()), -1))
        class_token = self.class_projection(self.class_embedding(labels))
        tokens = torch.stack((position, time_token, class_token), dim=1) + self.token_types
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_projection(self.output_norm(tokens[:, 0]))
