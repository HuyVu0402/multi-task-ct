# src/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net cơ bản 2D (segmentation 1 kênh)"""
    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_channels
        # Encoder
        for feat in features:
            self.downs.append(DoubleConv(ch, feat))
            ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        rev_feats = list(reversed(features))
        for feat in rev_feats:
            self.ups.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feat * 2, feat))

        # Output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        x = self.final_conv(x)
        return torch.sigmoid(x)


class UNetMultiTask(nn.Module):
    """
    U-Net multi-task:
      - Segmentation: 1 kênh (lesion mask)
      - Classification: num_cls_classes (vd 2: normal/covid)
    """
    def __init__(self,
                 in_channels=1,
                 num_seg_classes=1,
                 num_cls_classes=2,
                 features=(64, 128, 256, 512)):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.downs = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(ch, feat))
            ch = feat

        # Bottleneck chung
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder cho segmentation
        self.ups = nn.ModuleList()
        rev_feats = list(reversed(features))
        for feat in rev_feats:
            self.ups.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feat * 2, feat))

        # Head segmentation
        self.final_conv = nn.Conv2d(features[0], num_seg_classes, kernel_size=1)

        # Head classification (từ bottleneck feature)
        bottleneck_ch = features[-1] * 2
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_cls_classes)  # logits (không softmax)
        )

    def forward(self, x):
        skips = []
        out = x

        # Encoder
        for down in self.downs:
            out = down(out)
            skips.append(out)
            out = self.pool(out)

        # Bottleneck
        out = self.bottleneck(out)
        bottleneck_feat = out  # [B, C, H, W]

        # Decoder (seg)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            out = self.ups[i](out)
            skip = skips[i // 2]
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = self.ups[i + 1](out)

        seg = self.final_conv(out)
        seg = torch.sigmoid(seg)  # [B,1,H,W] prob 0-1

        # Classification từ bottleneck
        cls = self.cls_head(bottleneck_feat)  # [B,num_cls_classes] logits

        return seg, cls
