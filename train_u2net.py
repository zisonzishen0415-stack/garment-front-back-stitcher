"""U-2-Net 微调 — 用人手标注 mask 训练模型区分衣服 vs 人台

用法:
    python train_u2net.py                  # 从下载的预训练权重开始微调
    python train_u2net.py --epochs 50      # 指定训练轮数
    python train_u2net.py --resume ckpt.pt # 从断点续训

数据: 素材/7-21p图/*.JPG + *_mask.png（自动配对）
输出: models/u2net_finetuned.onnx
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

# ============================================================
# U-2-Net 模型架构 (PyTorch)
# 与 Carve/u2net-universal (HuggingFace) 预训练权重完全兼容
# ============================================================

class _REBNCONV(nn.Module):
    """单个 Conv+BN+ReLU 模块，命名与 HF 权重一致"""

    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn_s1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn_s1(self.conv_s1(x)), inplace=True)


class _RSU(nn.Module):
    """Residual U-Block。
    depth=N 表示有 N 个 encoder 层（含输入+bottleneck），decoder 层数为 N-1。
    每个 encoder 层的输出（pooling 前）作为同索引 decoder 层的 skip connection。
    最终输出与 rebnconvin 的输出相加（残差连接）。
    """

    def __init__(self, in_ch, mid_ch, out_ch, depth, dilated=False):
        super().__init__()
        self.depth = depth
        self.rebnconvin = _REBNCONV(in_ch, out_ch)

        # Encoder: rebnconv1 到 rebnconv{depth}
        for i in range(1, depth + 1):
            ic = out_ch if i == 1 else mid_ch
            oc = mid_ch
            dl = 2 if (dilated and i == depth) else 1
            setattr(self, f"rebnconv{i}", _REBNCONV(ic, oc, dl))

        # Decoder: rebnconv{depth-1}d 到 rebnconv{1}d
        for i in range(depth - 1, 0, -1):
            ic = 2 * mid_ch
            oc = out_ch if i == 1 else mid_ch
            setattr(self, f"rebnconv{i}d", _REBNCONV(ic, oc))

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        hx = self.rebnconvin(x)
        h = hx

        # Encoder
        enc = []
        for i in range(1, self.depth + 1):
            h = getattr(self, f"rebnconv{i}")(h)
            enc.append(h)
            if i < self.depth:
                h = self.pool(h)

        # Decoder
        for i in range(self.depth - 1, 0, -1):
            h = F.interpolate(h, enc[i - 1].shape[2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, enc[i - 1]], 1)
            h = getattr(self, f"rebnconv{i}d")(h)

        return h + hx


class U2Net(nn.Module):
    """U^2-Net — 从 HuggingFace 预训练权重加载，可用于微调"""

    def __init__(self):
        super().__init__()

        # Encoder
        self.stage1 = _RSU(3, 32, 64, 7)       # RSU-7
        self.pool12 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage2 = _RSU(64, 32, 128, 6)      # RSU-6
        self.pool23 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage3 = _RSU(128, 64, 256, 5)     # RSU-5
        self.pool34 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage4 = _RSU(256, 128, 512, 4)    # RSU-4
        self.pool45 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage5 = _RSU(512, 256, 512, 4, dilated=True)   # RSU-4F
        self.pool56 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage6 = _RSU(512, 256, 512, 4, dilated=True)   # RSU-4F

        # Decoder (RSU-4F 用 dilated)
        self.stage5d = _RSU(1024, 256, 512, 4, dilated=True)
        self.stage4d = _RSU(1024, 128, 256, 4)
        self.stage3d = _RSU(512, 64, 128, 5)
        self.stage2d = _RSU(256, 32, 64, 6)
        self.stage1d = _RSU(128, 16, 64, 7)

        # Side outputs (deep supervision)
        self.side1 = nn.Conv2d(64, 1, 3, padding=1)
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side4 = nn.Conv2d(256, 1, 3, padding=1)
        self.side5 = nn.Conv2d(512, 1, 3, padding=1)
        self.side6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # Encoder
        h1 = self.stage1(x)                                     # 1/1
        h2 = self.stage2(self.pool12(h1))                       # 1/2
        h3 = self.stage3(self.pool23(h2))                       # 1/4
        h4 = self.stage4(self.pool34(h3))                       # 1/8
        h5 = self.stage5(self.pool45(h4))                       # 1/16
        h6 = self.stage6(self.pool56(h5))                       # 1/32

        # Decoder
        h5d = self.stage5d(torch.cat([F.interpolate(h6, h5.shape[2:], mode='bilinear', align_corners=False), h5], 1))
        h4d = self.stage4d(torch.cat([F.interpolate(h5d, h4.shape[2:], mode='bilinear', align_corners=False), h4], 1))
        h3d = self.stage3d(torch.cat([F.interpolate(h4d, h3.shape[2:], mode='bilinear', align_corners=False), h3], 1))
        h2d = self.stage2d(torch.cat([F.interpolate(h3d, h2.shape[2:], mode='bilinear', align_corners=False), h2], 1))
        h1d = self.stage1d(torch.cat([F.interpolate(h2d, h1.shape[2:], mode='bilinear', align_corners=False), h1], 1))

        # Side outputs
        d1 = F.interpolate(self.side1(h1d), (h, w), mode='bilinear', align_corners=False)
        d2 = F.interpolate(self.side2(h2d), (h, w), mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.side3(h3d), (h, w), mode='bilinear', align_corners=False)
        d4 = F.interpolate(self.side4(h4d), (h, w), mode='bilinear', align_corners=False)
        d5 = F.interpolate(self.side5(h5d), (h, w), mode='bilinear', align_corners=False)
        d6 = F.interpolate(self.side6(h6), (h, w), mode='bilinear', align_corners=False)

        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), \
               torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)


# ============================================================
# 数据集
# ============================================================

class MaskDataset(Dataset):
    """从素材目录自动配对图片 + mask

    目录下 *.JPG + *_mask.png → (image, mask) pair
    图片 → PIL → resize 320×320 → tensor [0,1]
    mask → PIL → resize 320×320 → tensor {0,1}
    """

    def __init__(self, src_dir: str, size: int = 320, augment: bool = True):
        self.size = size
        self.augment = augment

        self.pairs = []
        # 支持逗号分隔的多目录
        for d in src_dir.split(","):
            d = d.strip()
            if not d:
                continue
            for mp in sorted(Path(d).glob("*_mask.png")):
                if "_mask_mask" in mp.name:
                    continue
                stem = mp.stem.replace("_mask", "")
                for ext in [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"]:
                    orig = Path(d) / f"{stem}{ext}"
                    if orig.exists():
                        self.pairs.append((orig, mp))
                        break

        if not self.pairs:
            raise RuntimeError(f"未找到配对数据: {src_dir}/*_mask.png + 对应原图")
        print(f"数据集: {len(self.pairs)} 对 (目录: {src_dir})")

    def __len__(self):
        return len(self.pairs) * 8 if self.augment else len(self.pairs)

    def __getitem__(self, idx):
        i = idx % len(self.pairs)
        img_path, mask_path = self.pairs[i]

        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 随机正方形 crop
        if self.augment:
            w, h = img.size
            crop_sz = min(w, h)
            x = random.randint(0, max(0, w - crop_sz))
            y = random.randint(0, max(0, h - crop_sz))
            img = img.crop((x, y, x + crop_sz, y + crop_sz))
            mask = mask.crop((x, y, x + crop_sz, y + crop_sz))

        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.BILINEAR)

        img_t = torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1)) / 255.0
        mask_t = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_t = (mask_t > 0.5).float()

        # 数据增强
        if self.augment:
            if random.random() < 0.5:
                img_t = torch.flip(img_t, [-1])
                mask_t = torch.flip(mask_t, [-1])
            if random.random() < 0.3:
                k = random.choice([0, 1, 2, 3])
                img_t = torch.rot90(img_t, k, [1, 2])
                mask_t = torch.rot90(mask_t, k, [1, 2])
            # 颜色抖动
            if random.random() < 0.2:
                img_t = img_t * random.uniform(0.85, 1.15)
                img_t = img_t.clamp(0, 1)

        return img_t, mask_t


# ============================================================
# 损失函数
# ============================================================

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def muti_loss(d0, d1, d2, d3, d4, d5, d6, target):
    """Deep supervision: 所有侧输出的加权 BCE + Dice"""
    losses = []
    for d in [d0, d1, d2, d3, d4, d5, d6]:
        bce = F.binary_cross_entropy(d, target)
        dice = dice_loss(d, target)
        losses.append(bce + dice)
    return sum(losses) / len(losses)


# ============================================================
# 训练
# ============================================================

def download_pretrained(dst: str) -> bool:
    """尝试从多个源下载 u2net 预训练权重。"""
    import urllib.request

    urls = [
        # HuggingFace mirror (推荐)
        "https://huggingface.co/davidfant/rembg-u2net/resolve/main/u2net.pth",
        # 官方 Google Drive 直链 (via gdown)
        # gdown fallback
    ]

    dst_path = Path(dst)
    if dst_path.exists():
        print(f"预训练权重已存在: {dst}")
        return True

    for url in urls:
        try:
            print(f"下载预训练权重: {url}")
            urllib.request.urlretrieve(url, dst)
            print("下载完成")
            return True
        except Exception as e:
            print(f"  …失败: {e}")

    # gdown fallback
    try:
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
            dst, quiet=False)
        return True
    except Exception:
        pass

    return False


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"数据: {args.data}")

    # 数据集
    dataset = MaskDataset(args.data, size=args.size, augment=not args.no_augment)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    print(f"样本: {len(dataset)} (原图 × {len(dataset.pairs)}对 × 增强)")

    # 模型
    model = U2Net().to(device)

    # 加载预训练权重
    pretrained = args.pretrained or "models/u2net.pth"
    if os.path.exists(pretrained):
        print(f"加载预训练权重: {pretrained}")
        state = torch.load(pretrained, map_location=device, weights_only=True)
        # 处理 checkpoint 格式 {'model': {...}, 'loss': ..., 'epoch': ...}
        if "model" in state:
            state = state["model"]
        # 处理 key 前缀差异 (module. 前缀来自 DDP)
        filtered = {}
        for k, v in state.items():
            filtered[k.replace("module.", "")] = v
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"  缺失 keys: {len(missing)}")
        if unexpected:
            print(f"  多余 keys: {len(unexpected)}")
        print("预训练权重加载完成")
    else:
        print(f"预训练权重不存在 ({pretrained})，尝试下载...")
        if download_pretrained(pretrained):
            return train(args)  # 递归重试
        print("无法获取预训练权重，将从头训练（不推荐）")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_loss = float("inf")

    # 断点续训
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", best_loss)
        print(f"从 epoch {start_epoch} 续训")

    os.makedirs("models", exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (img, mask) in enumerate(loader):
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()
            d0, d1, d2, d3, d4, d5, d6 = model(img)
            loss = muti_loss(d0, d1, d2, d3, d4, d5, d6, mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} "
                      f"Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"--- Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f} ---")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": model.state_dict(), "loss": avg_loss, "epoch": epoch},
                       "models/best.pt")
            print(f"    新最佳模型保存")

        # 每 N 轮保存检查点
        if (epoch + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "loss": avg_loss, "epoch": epoch},
                       f"models/ckpt_epoch{epoch+1}.pt")

    # 导出 ONNX
    print("导出 ONNX...")
    model.eval()
    dummy = torch.randn(1, 3, args.size, args.size, device=device)
    onnx_path = "models/u2net_finetuned.onnx"

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        export_params=True,
    )
    print(f"ONNX 模型已导出: {onnx_path}")

    # 验证
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    test_out = sess.run(None, {"input": dummy.cpu().numpy()})
    print(f"ONNX 验证通过: output shape = {test_out[0].shape}")

    print("\n训练完成！")
    print(f"  最佳模型: models/best.pt")
    print(f"  ONNX 模型: models/u2net_finetuned.onnx")
    print(f"  用 reviewer.py 测试: 将 {onnx_path} 放入 models/ 目录")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-2-Net 微调")
    parser.add_argument("--data", default="素材/7-21p图", help="标注数据目录")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--size", type=int, default=320, help="输入分辨率")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--save-every", type=int, default=10, help="每 N 轮保存检查点")
    parser.add_argument("--no-augment", action="store_true", help="禁用数据增强")
    parser.add_argument("--pretrained", default="", help="预训练权重路径")
    parser.add_argument("--resume", default="", help="从检查点续训")
    args = parser.parse_args()
    train(args)
