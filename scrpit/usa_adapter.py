import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateEncoder(nn.Module):
    """
    物理坐标编码器
    将 2D 物理绝对坐标 (X_mm, Y_mm) 升维到与 ViT 特征相同的维度。
    这使得模型能够理解每个视觉 Token 在真实世界中的物理位置。
    """
    def __init__(self, coord_dim=2, embed_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, coords):
        # coords shape: (Batch, N, 2)
        # N 是 Token 的数量，例如 16x16 = 256
        return self.mlp(coords)

class USACrossAttentionBlock(nn.Module):
    """
    通用尺度适配器核心块 (USA Cross-Attention Block)
    基于目标物理坐标进行特征重采样的关键模块。
    """
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        # 交叉注意力层：根据 Query 和 Key 的相似度，对 Value 进行加权聚合
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 前馈神经网络 (FFN)，增强特征表达能力
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, source_feat, source_pos_enc, target_pos_enc):
        """
        参数:
            source_feat: (B, N, D) 当前异形传感器提取的视觉特征
            source_pos_enc: (B, N, D) 当前异形传感器对应的物理坐标编码
            target_pos_enc: (B, M, D) 目标基座模型期望的标准物理坐标编码
        返回:
            重采样后的特征 (B, M, D)
        """
        # Query: 目标模型期望的物理位置 (例如：我想知道标准 GelSight 在这个位置应该看到什么？)
        Q = target_pos_enc
        
        # Key: 异形传感器的视觉特征 + 其真实的物理位置
        # 告诉模型：这是一种特定的视觉形变，并且它发生在当前的这个物理坐标上
        K = source_feat + source_pos_enc
        
        # Value: 纯粹的异形传感器视觉特征
        V = source_feat
        
        # 交叉注意力计算：自动根据目标坐标与源坐标的物理距离，聚合源特征
        attn_out, _ = self.cross_attn(query=Q, key=K, value=V)
        
        # 归一化
        out = self.norm1(attn_out)
        
        # FFN 残差连接
        out = out + self.ffn(self.norm2(out))
        return out

class UniversalScaleAdapter(nn.Module):
    """
    完整的即插即用 Universal Scale Adapter (USA)
    解耦了物理尺度与视觉特征，实现异构触觉传感器的零样本迁移。
    """
    def __init__(self, embed_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        self.coord_encoder = CoordinateEncoder(coord_dim=2, embed_dim=embed_dim)
        self.layers = nn.ModuleList([
            USACrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads) 
            for _ in range(num_layers)
        ])

    def forward(self, source_feat, source_coords, target_coords):
        """
        参数:
            source_feat: (B, 16, 16, 768) 或 (B, 256, 768) 冻结的 ViT 提取的 Patch 特征
            source_coords: (B, 16, 16, 2) 或 (B, 256, 2) 由 TPS 模型推导出的当前传感器真实物理网格 (X_mm, Y_mm)
            target_coords: (B, 16, 16, 2) 或 (B, 256, 2) 目标大模型(如 OpenVLA)期望的物理网格坐标
        返回:
            与 target_coords 尺度一致的重构特征矩阵
        """
        # 自动处理四维张量，将其展平为序列，方便 Transformer 处理
        if source_feat.dim() == 4:
            b, h, w, c = source_feat.shape
            source_feat = source_feat.view(b, h*w, c)
            source_coords = source_coords.view(b, h*w, 2)
            target_coords = target_coords.view(b, h*w, 2)
            reshape_back = True
        else:
            reshape_back = False

        # 1. 编码源坐标和目标坐标
        src_pos_enc = self.coord_encoder(source_coords)
        tgt_pos_enc = self.coord_encoder(target_coords)

        # 2. 逐层穿过 Cross-Attention 网络进行空间特征重采样
        x = source_feat
        for layer in self.layers:
            x = layer(x, src_pos_enc, tgt_pos_enc)
            
        # 3. 恢复原始的空间形状 (如果输入是四维的)
        if reshape_back:
            x = x.view(b, h, w, c)
            
        return x

# =====================================================================
# 本地测试代码 (确保模型结构正确)
# =====================================================================
if __name__ == "__main__":
    # 模拟环境参数
    batch_size = 4
    num_patches = 256 # 对应 16x16 的特征图
    embed_dim = 768
    
    # 实例化 Adapter
    adapter = UniversalScaleAdapter(embed_dim=embed_dim, num_layers=2)
    
    # 生成假数据
    # 假设这是从一个 25mm 视野的传感器提取的特征
    dummy_source_feat = torch.randn(batch_size, num_patches, embed_dim)
    # 假设这是 25mm 传感器的物理坐标
    dummy_source_coords = torch.randn(batch_size, num_patches, 2) 
    # 假设目标大模型期望的是 15mm 的物理坐标
    dummy_target_coords = torch.randn(batch_size, num_patches, 2) 
    
    # 执行前向传播
    out_feat = adapter(dummy_source_feat, dummy_source_coords, dummy_target_coords)
    
    print(f"✅ Adapter 初始化成功!")
    print(f"📥 输入源特征形状: {dummy_source_feat.shape}")
    print(f"📤 输出对齐特征形状: {out_feat.shape}")
    print(f"🎯 模型参数量: {sum(p.numel() for p in adapter.parameters())}")