import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------
# 1. 基础组件 (Basic Blocks)
# -----------------------------------------------------------------------

class ResNetBlock(nn.Module):
    """基础残差块，保持分辨率不变，用于特征提取。"""
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module。
    空洞率设为 [2, 4, 6]，适配训练时 16×16 的瓶颈特征图
    （原始 [6, 12, 18] 在 16×16 上感受野超出特征图，退化为 1×1 卷积）。
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        modules = []

        # Branch 1: 1×1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Branch 2-4: 不同空洞率的 3×3 卷积
        for rate in [2, 4, 6]:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Branch 5: 全局平均池化
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.convs = nn.ModuleList(modules)

        # 5 路拼接后投影回 out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.size()[2:], mode='bilinear', align_corners=False)
        res.append(global_feat)
        return self.project(torch.cat(res, dim=1))


# -----------------------------------------------------------------------
# 2. 编码器 (Encoder Branch)
# -----------------------------------------------------------------------

class EncoderBranch(nn.Module):
    """
    通用编码器分支：16× 下采样（H → H/16），输出 4 级 skip feature。
    SAR 分支与纹理分支均使用同一结构，因为 MyDS 所有变量分辨率一致。
    """
    def __init__(self, in_channels):
        super(EncoderBranch, self).__init__()

        # Stage 1: 7×7 conv, stride 2 → 1/2
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 2: stride 2 + 3×ResBlock → 1/4
        self.stage2_down = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage2_res = nn.Sequential(*[ResNetBlock(64, 64) for _ in range(3)])

        # Stage 3: stride 2 + 3×ResBlock → 1/8
        self.stage3_down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage3_res = nn.Sequential(*[ResNetBlock(128, 128) for _ in range(3)])

        # Stage 4: stride 2 + 3×ResBlock → 1/16
        self.stage4_down = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage4_res = nn.Sequential(*[ResNetBlock(256, 256) for _ in range(3)])

    def forward(self, x):
        f1 = self.stage1(x)                      # [B,  32, H/2,  W/2 ]
        f2 = self.stage2_res(self.stage2_down(f1))  # [B,  64, H/4,  W/4 ]
        f3 = self.stage3_res(self.stage3_down(f2))  # [B, 128, H/8,  W/8 ]
        f4 = self.stage4_res(self.stage4_down(f3))  # [B, 256, H/16, W/16]
        return f1, f2, f3, f4


# -----------------------------------------------------------------------
# 3. 完整的 DBU-Net 模型
# -----------------------------------------------------------------------

class DBUNet_ASPP(nn.Module):
    """
    双分支 U-Net + ASPP，适配 MyDS 数据集。

    通道划分（由 options['dbunet_sar_channels'] 控制，默认 3）：
      Branch 1 (SAR)    : nersc_sar_primary, nersc_sar_secondary, sar_incidenceangle  (3通道)
      Branch 2 (Texture): glcm_contrast, glcm_dissimilarity, glcm_homogeneity, global_valid_mask (4通道)

    接口与 UNet 完全一致：
      forward(x: Tensor) → {'SIC': Tensor, 'SOD': Tensor, 'FLOE': Tensor}
    """

    def __init__(self, options):
        super(DBUNet_ASPP, self).__init__()

        n_total   = len(options['train_variables'])
        self.n_sar = options.get('dbunet_sar_channels', 3)   # 前 n_sar 通道 → SAR 分支
        n_tex      = n_total - self.n_sar                     # 剩余通道 → 纹理分支

        # --- 双编码器（均为全分辨率 EncoderBranch）---
        self.encoder_sar = EncoderBranch(in_channels=self.n_sar)
        self.encoder_tex = EncoderBranch(in_channels=n_tex)

        # --- 瓶颈：cat(s4, t4) = 512 ch → ASPP ---
        self.aspp = ASPP(in_channels=512, out_channels=512)

        # --- 解码器 ---
        # Up1: 1/16 → 1/8  | cat(up(256), s3(128), t3(128)) = 512
        self.up1  = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(ResNetBlock(512, 256), ResNetBlock(256, 256))

        # Up2: 1/8  → 1/4  | cat(up(128), s2(64), t2(64)) = 256
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(ResNetBlock(256, 128), ResNetBlock(128, 128))

        # Up3: 1/4  → 1/2  | cat(up(64), s1(32), t1(32)) = 128
        self.up3  = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Sequential(ResNetBlock(128, 64), ResNetBlock(64, 64))

        # Up4: 1/2  → full | 无 skip
        self.up4  = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 共享末端特征层
        self.last_stage = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # --- 三个独立输出头，与 UNet 对齐 ---
        self.sic_head  = nn.Conv2d(16, options['n_classes']['SIC'],  kernel_size=1)
        self.sod_head  = nn.Conv2d(16, options['n_classes']['SOD'],  kernel_size=1)
        self.floe_head = nn.Conv2d(16, options['n_classes']['FLOE'], kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x):
        # 1. 通道切分
        x_sar = x[:, :self.n_sar, :, :]   # [B, 3, H, W]
        x_tex = x[:, self.n_sar:, :, :]   # [B, 4, H, W]

        # 2. 双分支编码
        s1, s2, s3, s4 = self.encoder_sar(x_sar)
        t1, t2, t3, t4 = self.encoder_tex(x_tex)

        # 3. 融合 + ASPP
        feat_aspp = self.aspp(torch.cat([s4, self._match(t4, s4)], dim=1))

        # 4. 逐级解码，拼接双分支 skip
        d1 = self.up1(feat_aspp)
        d1 = self.dec1(torch.cat([d1, self._match(s3, d1), self._match(t3, d1)], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, self._match(s2, d2), self._match(t2, d2)], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, self._match(s1, d3), self._match(t1, d3)], dim=1))

        d4 = self.up4(d3)
        feat = self.last_stage(d4)  # [B, 16, H, W]

        # 5. 对齐输入尺寸（防止奇数 patch 造成的 off-by-one）
        if feat.size()[2:] != x.size()[2:]:
            feat = F.interpolate(feat, size=x.size()[2:], mode='bilinear', align_corners=False)

        return {
            'SIC':  self.sic_head(feat),
            'SOD':  self.sod_head(feat),
            'FLOE': self.floe_head(feat),
        }

    @staticmethod
    def _match(src, ref):
        """将 src 的空间尺寸双线性对齐到 ref（处理奇数尺寸的 off-by-one）。"""
        if src.size()[2:] != ref.size()[2:]:
            src = F.interpolate(src, size=ref.size()[2:], mode='bilinear', align_corners=True)
        return src


# -----------------------------------------------------------------------
# 4. 单元测试
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from configs._base_.base import train_options as base_options

    # 构造最小 options，模拟流水线传入的 train_options
    options = {
        'train_variables': [
            'nersc_sar_primary', 'nersc_sar_secondary', 'sar_incidenceangle',
            'glcm_sigma0_hh_contrast', 'glcm_sigma0_hh_dissimilarity',
            'glcm_sigma0_hh_homogeneity', 'global_valid_mask'
        ],
        'n_classes': {'SIC': 12, 'SOD': 5, 'FLOE': 8},
        'dbunet_sar_channels': 3,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = DBUNet_ASPP(options=options).to(device)

    # 模拟流水线输入：单张量 [B, 7, 256, 256]
    dummy_x = torch.randn(2, 7, 256, 256).to(device)

    print("模型已构建，开始前向传播测试...")
    output = model(dummy_x)

    print(f"\n测试成功！")
    print(f"Input  shape : {dummy_x.shape}")
    for k, v in output.items():
        print(f"Output {k:4s} : {v.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量 : {total_params / 1e6:.2f} M")
