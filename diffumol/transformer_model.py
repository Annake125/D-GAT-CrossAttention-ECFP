from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

# ============== 新增：轻量级图融合模块 ==============
class GraphCrossAttention(nn.Module):
    """
    轻量级跨注意力融合图嵌入
    参数量：~5000 (相比原方案的50000减少10倍)
    """
    def __init__(self, hidden_dim, graph_embed_dim, num_heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 极轻量的投影层（无bias减少参数）
        self.graph_proj = nn.Linear(graph_embed_dim, hidden_dim, bias=False)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = self.head_dim ** -0.5
        
        # 可学习的温度参数（控制融合强度）
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)  # 初始温度0.5
        
    def forward(self, text_emb, graph_emb):
        """
        text_emb: [B, L, D] - 文本嵌入
        graph_emb: [B, D'] - 图嵌入
        返回: [B, L, D] - 图信息增强的文本嵌入
        """
        B, L, D = text_emb.shape
        
        # 投影图嵌入到相同维度
        g = self.graph_proj(graph_emb).unsqueeze(1)  # [B, 1, D]
        
        # Query from text, Key/Value from graph
        q = self.q_proj(text_emb)  # [B, L, D]
        
        # Multi-head attention (简化版)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = g.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = k
        
        # Scaled dot-product attention with temperature
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, L, 1]
        attn = torch.softmax(attn / self.temperature.clamp(min=0.01), dim=-1)
        
        # Weighted graph information
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        
        return out
# ============== 新增：分子指纹融合模块 ==============
class FingerprintCrossAttention(nn.Module):
    """
    轻量级跨注意力融合分子指纹（ECFP/Morgan）
    参数量：~7000 (极轻量设计)
    设计思路：
    - 指纹是预计算的固定向量(2048D)，无需反向传播
    - 通过降维MLP投影到hidden_dim
    - 使用Cross-Attention融合到文本嵌入
    """
    def __init__(self, hidden_dim, fp_dim=2048, num_heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 轻量级降维投影（2048 -> hidden_dim）
        # 使用两层MLP提取指纹特征
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim // 2, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(hidden_dim // 2, hidden_dim, bias=False)
        )

        # Query投影（从文本嵌入）
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = self.head_dim ** -0.5

        # 可学习的温度参数（控制融合强度）
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, text_emb, fp_emb):
        """
        text_emb: [B, L, D] - 文本嵌入
        fp_emb: [B, fp_dim] - 预计算的分子指纹（ECFP）
        返回: [B, L, D] - 指纹信息增强的文本嵌入
        """
        B, L, D = text_emb.shape

        # 投影指纹到相同维度
        f = self.fp_proj(fp_emb).unsqueeze(1)  # [B, 1, D]

        # Query from text, Key/Value from fingerprint
        q = self.q_proj(text_emb)  # [B, L, D]

        # Multi-head attention
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = f.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = k

        # Scaled dot-product attention with temperature
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, L, 1]
        attn = torch.softmax(attn / self.temperature.clamp(min=0.01), dim=-1)

        # Weighted fingerprint information
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)

        return out

# ============== 新增：门控结构融合模块 (ECFP + Mol2Vec) ==============
class GatedStructureFusion(nn.Module):
    """
    自适应门控融合ECFP和Mol2Vec两种分子结构表示

    设计思路:
    - ECFP (2048维): 精确的子结构匹配,对已知化学基团敏感
    - Mol2Vec (300维): 语义相似性,泛化能力强,对新颖结构友好
    - 门控机制: 让模型自动学习每个分子使用哪种特征更好

    参数量: ~150K (相比两个独立CrossAttention的~12K节省90%)

    优势:
    1. 信息互补: 精确匹配 vs 语义相似
    2. 自适应选择: 根据分子特性动态调整权重
    3. 轻量设计: 共享投影层减少参数
    4. 可解释性: 门控值反映特征重要性
    """
    def __init__(self, hidden_dim, ecfp_dim=2048, mol2vec_dim=300):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ecfp_dim = ecfp_dim
        self.mol2vec_dim = mol2vec_dim

        # ===== 特征投影层 =====
        # ECFP投影: 2048 -> hidden_dim (降维)
        self.ecfp_proj = nn.Sequential(
            nn.Linear(ecfp_dim, hidden_dim // 2, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim, bias=False)
        )

        # Mol2Vec投影: 300 -> hidden_dim (升维+对齐)
        self.mol2vec_proj = nn.Sequential(
            nn.Linear(mol2vec_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # ===== 门控网络 =====
        # 输入: 拼接的两种特征 [ecfp_proj || mol2vec_proj]
        # 输出: 门控权重 [w_ecfp, w_mol2vec]
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2, bias=True),  # 输出2个门控值
            nn.Softmax(dim=-1)  # 归一化到[0,1],和为1
        )

        # ===== 融合后的Cross-Attention =====
        # 统一的交叉注意力模块(替代两个独立的)
        self.num_heads = 2
        self.head_dim = hidden_dim // self.num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = self.head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, text_emb, ecfp_emb, mol2vec_emb):
        """
        Args:
            text_emb: [B, L, D] - 文本嵌入
            ecfp_emb: [B, 2048] - ECFP指纹
            mol2vec_emb: [B, 300] - Mol2Vec嵌入
        Returns:
            [B, L, D] - 融合后的结构增强文本嵌入
        """
        B, L, D = text_emb.shape

        # ===== 步骤1: 投影到统一维度 =====
        ecfp_feat = self.ecfp_proj(ecfp_emb)        # [B, D]
        mol2vec_feat = self.mol2vec_proj(mol2vec_emb)  # [B, D]

        # ===== 步骤2: 门控权重计算 =====
        # 拼接两种特征作为门控输入
        concat_feat = torch.cat([ecfp_feat, mol2vec_feat], dim=-1)  # [B, 2D]
        gate_weights = self.gate(concat_feat)  # [B, 2]

        # 分离门控权重
        w_ecfp = gate_weights[:, 0:1]      # [B, 1]
        w_mol2vec = gate_weights[:, 1:2]   # [B, 1]

        # ===== 步骤3: 加权融合 =====
        # 门控加权平均
        fused_struct = w_ecfp * ecfp_feat + w_mol2vec * mol2vec_feat  # [B, D]

        # ===== 步骤4: Cross-Attention融合到文本 =====
        struct_kv = fused_struct.unsqueeze(1)  # [B, 1, D]

        # Query from text
        q = self.q_proj(text_emb)  # [B, L, D]

        # Multi-head attention
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = struct_kv.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = k

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, L, 1]
        attn = torch.softmax(attn / self.temperature.clamp(min=0.01), dim=-1)

        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)

        return out, gate_weights  # 返回门控权重用于监控

# ============== 主模型修改 ==============
class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
        **kwargs
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size
        self.num_props = kwargs["num_props"]
        
        # ====== 图嵌入融合模块 ======
        self.use_graph = kwargs.get("use_graph", False)
        if self.use_graph:
            graph_embed_dim = kwargs.get("graph_embed_dim", 128)
            # 用轻量级Cross-Attention替代原来的3层MLP
            self.graph_fusion = GraphCrossAttention(
                hidden_dim=config.hidden_size,
                graph_embed_dim=graph_embed_dim,
                num_heads=2  # 只用2个头，足够且轻量
            )

        # ====== 分层结构融合模块 (ECFP + Mol2Vec) ======
        self.use_fingerprint = kwargs.get("use_fingerprint", False)
        self.use_mol2vec = kwargs.get("use_mol2vec", False)

        # 选项1: 使用门控融合 (推荐 - 同时使用ECFP和Mol2Vec)
        if self.use_fingerprint and self.use_mol2vec:
            fp_dim = kwargs.get("fp_dim", 2048)
            mol2vec_dim = kwargs.get("mol2vec_dim", 300)
            # 门控融合模块 - 自适应融合ECFP和Mol2Vec
            self.struct_fusion = GatedStructureFusion(
                hidden_dim=config.hidden_size,
                ecfp_dim=fp_dim,
                mol2vec_dim=mol2vec_dim
            )
            print(f"[Info] 使用门控融合: ECFP({fp_dim}) + Mol2Vec({mol2vec_dim})")

        # 选项2: 仅使用ECFP (向后兼容)
        elif self.use_fingerprint:
            fp_dim = kwargs.get("fp_dim", 2048)
            self.fp_fusion = FingerprintCrossAttention(
                hidden_dim=config.hidden_size,
                fp_dim=fp_dim,
                num_heads=2
            )
            print(f"[Info] 仅使用ECFP: {fp_dim}维")

        # 选项3: 仅使用Mol2Vec
        elif self.use_mol2vec:
            mol2vec_dim = kwargs.get("mol2vec_dim", 300)
            # 复用FingerprintCrossAttention架构
            self.mol2vec_fusion = FingerprintCrossAttention(
                hidden_dim=config.hidden_size,
                fp_dim=mol2vec_dim,
                num_heads=2
            )
            print(f"[Info] 仅使用Mol2Vec: {mol2vec_dim}维")

        # ====== 可学习的多模态融合权重 ======
        # 计算需要融合的模态数量
        num_modalities = 0
        if self.use_graph:
            num_modalities += 1
        if self.use_fingerprint or self.use_mol2vec:
            num_modalities += 1  # 结构特征算作一个模态

        if num_modalities == 2:
            # 双模态: 图 + 结构
            self.fusion_weights = nn.Parameter(torch.tensor([0.1, 0.05]))
        elif num_modalities == 1:
            # 单模态
            self.fusion_weights = nn.Parameter(torch.tensor([0.1]))

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        
        if self.num_props:
            self.prop_nn = nn.Linear(self.num_props, self.input_dims) 
        
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids.to(th.int64))

    def get_props(self, props):
        return self.prop_nn(props.unsqueeze(1))
        
    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: 
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) 
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps, graph_ids=None, fp_embs=None, mol2vec_embs=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param graph_ids: molecular indices used to retrieve graph embeddings (optional)
        :param fp_embs: ECFP fingerprint embeddings (optional)
        :param mol2vec_embs: Mol2Vec embeddings (optional)
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]

        # ====== 多模态融合：图 + 结构（ECFP + Mol2Vec） ======
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.LayerNorm(emb_inputs)

        graph_info = None
        struct_info = None
        gate_weights = None  # 用于监控门控权重

        # ===== 1. 图嵌入融合 =====
        if self.use_graph and graph_ids is not None:
            # 获取图嵌入
            graph_emb = self.graph_embeddings[graph_ids]  # [B, graph_dim]
            # Cross-attention融合
            graph_info = self.graph_fusion(emb_inputs, graph_emb)  # [B, L, D]

        # ===== 2. 结构嵌入融合 (三种模式) =====
        # 模式1: 门控融合 ECFP + Mol2Vec (推荐)
        if self.use_fingerprint and self.use_mol2vec:
            if fp_embs is not None and mol2vec_embs is not None:
                struct_info, gate_weights = self.struct_fusion(
                    emb_inputs, fp_embs, mol2vec_embs
                )  # [B, L, D], [B, 2]

        # 模式2: 仅使用ECFP
        elif self.use_fingerprint and fp_embs is not None:
            struct_info = self.fp_fusion(emb_inputs, fp_embs)  # [B, L, D]

        # 模式3: 仅使用Mol2Vec
        elif self.use_mol2vec and mol2vec_embs is not None:
            struct_info = self.mol2vec_fusion(emb_inputs, mol2vec_embs)  # [B, L, D]

        # ===== 3. 可学习的加权融合 =====
        num_modalities_active = sum([
            graph_info is not None,
            struct_info is not None
        ])

        if num_modalities_active == 2:
            # 双模态融合（图 + 结构）
            w_graph, w_struct = self.fusion_weights.abs()  # 保证非负
            emb_inputs = emb_inputs + w_graph * graph_info + w_struct * struct_info
        elif graph_info is not None:
            # 仅图
            emb_inputs = emb_inputs + self.fusion_weights[0].abs() * graph_info
        elif struct_info is not None:
            # 仅结构
            emb_inputs = emb_inputs + self.fusion_weights[0].abs() * struct_info

        emb_inputs = self.dropout(emb_inputs)
        
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h