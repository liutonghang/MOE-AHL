import torch
from torch import nn
from models.moe_ahl_layer import Transformer, CrossTransformer, HhyperLearningEncoder
from models.bert import BertTextEncoder
from einops import repeat
import torch.nn.functional as F


class NoisyTop1Router(nn.Module):
    def __init__(self, input_dim, num_experts, noise_epsilon=0.1):
        super().__init__()
        self.noise_epsilon = noise_epsilon
        self.w_gate = nn.Linear(input_dim, num_experts, bias=False)
        self.w_noise = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x, training=True):
        # 基础路由权重
        gate_logits = self.w_gate(x)

        # 训练时添加噪声
        if training:
            noise_logits = self.w_noise(x)
            noise = torch.rand_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + (noise_logits * noise)

        # 计算softmax权重
        gates = F.softmax(gate_logits, dim=1)

        # 返回权重和路由指标（用于计算负载均衡损失）
        return gates


class MOE_AHL(nn.Module):
    def __init__(self, args):
        super(MOE_AHL, self).__init__()

        args = args.model
        # h_hyper:[1, seq_len=8, d_model=128]
        self.h_hyper = nn.Parameter(torch.ones(1, args.token_len, args.token_dim))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args.bert_pretrained)

        self.proj_l = nn.Sequential(
            nn.Linear(args.l_input_dim, args.l_proj_dst_dim),
            Transformer(num_frames=args.l_input_length, save_hidden=False, token_len=args.token_length,
                        dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads,
                        mlp_dim=args.proj_mlp_dim, num_experts=args.moe_number_expert)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(args.a_input_dim, args.a_proj_dst_dim),
            Transformer(num_frames=args.a_input_length, save_hidden=False, token_len=args.token_length,
                        dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads,
                        mlp_dim=args.proj_mlp_dim, num_experts=args.moe_number_expert)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(args.v_input_dim, args.v_proj_dst_dim),
            Transformer(num_frames=args.v_input_length, save_hidden=False, token_len=args.token_length,
                        dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads,
                        mlp_dim=args.proj_mlp_dim, num_experts=args.moe_number_expert)
        )

        self.l_encoder = Transformer(num_frames=args.token_length, save_hidden=True, token_len=None,
                                     dim=args.proj_input_dim, depth=args.AHL_depth - 1, heads=args.l_enc_heads,
                                     mlp_dim=args.l_enc_mlp_dim, num_experts=args.moe_number_expert)
        self.a_encoder = Transformer(num_frames=args.token_length, save_hidden=True, token_len=None,
                                     dim=args.proj_input_dim, depth=args.AHL_depth - 1, heads=args.l_enc_heads,
                                     mlp_dim=args.l_enc_mlp_dim, num_experts=args.moe_number_expert)
        self.v_encoder = Transformer(num_frames=args.token_length, save_hidden=True, token_len=None,
                                     dim=args.proj_input_dim, depth=args.AHL_depth - 1, heads=args.l_enc_heads,
                                     mlp_dim=args.l_enc_mlp_dim, num_experts=args.moe_number_expert)

        self.h_l_hyper_layer = HhyperLearningEncoder(dim=args.token_dim, depth=args.AHL_depth, heads=args.ahl_heads,
                                                     dim_head=args.ahl_dim_head, dropout=args.ahl_droup)
        self.h_a_hyper_layer = HhyperLearningEncoder(dim=args.token_dim, depth=args.AHL_depth, heads=args.ahl_heads,
                                                     dim_head=args.ahl_dim_head, dropout=args.ahl_droup)
        self.h_v_hyper_layer = HhyperLearningEncoder(dim=args.token_dim, depth=args.AHL_depth, heads=args.ahl_heads,
                                                     dim_head=args.ahl_dim_head, dropout=args.ahl_droup)

        self.l_fusion_layer = CrossTransformer(source_num_frames=args.token_len, tgt_num_frames=args.token_len,
                                               dim=args.proj_input_dim, depth=args.fusion_layer_depth,
                                               heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim,
                                               num_experts=args.moe_cross_number_expert)
        self.a_fusion_layer = CrossTransformer(source_num_frames=args.token_len, tgt_num_frames=args.token_len,
                                               dim=args.proj_input_dim, depth=args.fusion_layer_depth,
                                               heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim,
                                               num_experts=args.moe_cross_number_expert)
        self.v_fusion_layer = CrossTransformer(source_num_frames=args.token_len, tgt_num_frames=args.token_len,
                                               dim=args.proj_input_dim, depth=args.fusion_layer_depth,
                                               heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim,
                                               num_experts=args.moe_cross_number_expert)

        self.l_regression_layer = nn.Sequential(
            nn.Linear(args.token_dim, 1)
        )
        self.a_regression_layer = nn.Sequential(
            nn.Linear(args.token_dim, 1)
        )
        self.v_regression_layer = nn.Sequential(
            nn.Linear(args.token_dim, 1)
        )

        # 新增：改进的路由器（带有噪声）
        router_dim = args.proj_input_dim * 3  # 三个模态特征拼接后的维度
        self.router = NoisyTop1Router(router_dim, num_experts=args.moe_moisy_top1_router_experts, noise_epsilon=0.1)

        # 保存路由权重用于计算负载均衡损失
        self.route_weights = None

    def forward(self, x_visual, x_audio, x_text):
        b = x_visual.size(0)
        # h_hyper:[1, seq_len=8, d_model=128] -> [batch_size,seq_len=8, d_model=128]
        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)
        # x_text:[batch_size,3,seq_len] -> [batch_size,seq_len,d_model]
        x_text = self.bertmodel(x_text)

        h_v = self.proj_v(x_visual)[:, :self.h_hyper.shape[1]]
        h_a = self.proj_a(x_audio)[:, :self.h_hyper.shape[1]]
        h_l = self.proj_l(x_text)[:, :self.h_hyper.shape[1]]

        # 路由输入特征（增强版）
        # 使用每个模态的CLS token（假设第一个token是CLS）
        router_input = torch.cat([h_l[:, 0], h_a[:, 0], h_v[:, 0]], dim=-1)

        # 计算路由权重（带噪声）
        self.route_weights = self.router(router_input, training=self.training)

        # 三个专家分别处理
        # 专家1：文本处理链路
        h_l_list = self.l_encoder(h_l)
        h_l_hyper = self.h_l_hyper_layer(h_l_list, h_a, h_v, h_hyper)
        feat_l = self.l_fusion_layer(h_l_hyper, h_l_list[-1])[:, 0]
        l_output = self.l_regression_layer(feat_l)

        # 专家2：音频处理链路
        h_a_list = self.a_encoder(h_a)
        h_a_hyper = self.h_a_hyper_layer(h_a_list, h_l, h_v, h_hyper)
        feat_a = self.a_fusion_layer(h_a_hyper, h_a_list[-1])[:, 0]
        a_output = self.a_regression_layer(feat_a)

        # 专家3：视频处理链路
        h_v_list = self.v_encoder(h_v)
        h_v_hyper = self.h_v_hyper_layer(h_v_list, h_a, h_l, h_hyper)
        feat_v = self.v_fusion_layer(h_v_hyper, h_v_list[-1])[:, 0]
        v_output = self.v_regression_layer(feat_v)

        # 加权聚合专家输出
        experts_output = torch.stack([l_output, a_output, v_output], dim=1).squeeze(-1)  # [b, 3]
        final_output = torch.sum(experts_output * self.route_weights, dim=1, keepdim=True)  # [b, 1]

        return final_output

    def get_load_balancing_loss(self):
        """计算负载均衡损失，防止专家使用不均衡"""
        if self.route_weights is None:
            return 0

        # 计算路由概率的均值（期望负载）
        expected_load = torch.mean(self.route_weights, dim=0)

        # 计算实际负载（每个专家被选择的频率）
        expert_mask = (self.route_weights > 0).float()
        actual_load = torch.mean(expert_mask, dim=0)

        # 计算KL散度作为负载均衡损失
        load_balancing_loss = F.kl_div(
            expected_load.log(), actual_load, reduction='batchmean'
        )

        return load_balancing_loss


def build_model(args):
    model = MOE_AHL(args)

    return model
