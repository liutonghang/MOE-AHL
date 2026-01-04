'''
* @name: ahl_layer.py
* @description: Basic layer of AHL. Note: Some code references Vision Transformer (VIT, https://github.com/gupta-abhay/pytorch-vit).
'''

import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


class PreNormAHL(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_t, h_a, h_v, h_hyper):
        h_t = self.norm1(h_t)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)

        return self.fn(h_t, h_a, h_v, h_hyper)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Expert(nn.Module):
    def __init__(self, dim,dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(dim, num_experts)
        self.noise_linear = nn.Linear(dim, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)

        noise_logits = self.noise_linear(mh_output)

        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k=4):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        # 1. 输入进入router得到两个输出
        gating_output, indices = self.router(x)
        # 2.初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros_like(x)

        # 3.展平，即把每个batch拼接到一起，这里对输入x和router后的结果都进行了展平
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # 以每个专家为单位进行操作，即把当前专家处理的所有token都进行加权
        for i, expert in enumerate(self.experts):
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            expert_mask = (indices == i).any(dim=-1)
            # 5. 展平操作
            flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if flat_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                expert_input = flat_x[flat_mask]
                # 7. 将token输入expert得到输出
                expert_output = expert(expert_input)

                # 8. 计算当前专家对于有作用的token的权重分数
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                # 9. 将expert输出乘上权重分数
                weighted_output = expert_output * gating_scores

                # 10. 循环进行做种的结果叠加
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v)) # [batch_size,seq_len,d_model] -> [batch_size,head,seq_len,d_model/head]
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class HhyperLearningLayer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, h_t, h_a, h_v, h_hyper):
        b, n, _, h = *h_t.shape, self.heads

        q = self.to_q(h_t)
        k_ta = self.to_k_ta(h_a)
        k_tv = self.to_k_tv(h_v)
        v_ta = self.to_v_ta(h_a)
        v_tv = self.to_v_tv(h_v)

        q, k_ta, k_tv, v_ta, v_tv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_ta, k_tv, v_ta, v_tv))

        dots_ta = einsum('b h i d, b h j d -> b h i j', q, k_ta) * self.scale
        attn_ta = self.attend(dots_ta)
        out_ta = einsum('b h i j, b h j d -> b h i d', attn_ta, v_ta)
        out_ta = rearrange(out_ta, 'b h n d -> b n (h d)')

        dots_tv = einsum('b h i d, b h j d -> b h i j', q, k_tv) * self.scale
        attn_tv = self.attend(dots_tv)
        out_tv = einsum('b h i j, b h j d -> b h i d', attn_tv, v_tv)
        out_tv = rearrange(out_tv, 'b h n d -> b n (h d)')

        h_hyper_shift = self.to_out(out_ta + out_tv)
        h_hyper += h_hyper_shift

        return h_hyper


class HhyperLearningEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAHL(dim, HhyperLearningLayer(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, h_t_list, h_a, h_v, h_hyper):
        for i, attn in enumerate(self.layers):
            h_hyper = attn[0](h_t_list[i], h_a, h_v, h_hyper)
        return h_hyper


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_experts,dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, SparseMoE(dim, num_experts, top_k=2))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_experts,dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, SparseMoE(dim, num_experts, top_k=2))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64,num_experts=4, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, num_experts,dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, num_experts=4, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim,num_experts, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b = b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, : n_s+1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, : n_t+1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t

