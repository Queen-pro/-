"""
支持 prompt 类型：
  --prompt point
  --prompt box
  --prompt text 

使用方式：
  # point prompt
  python inference_ma_clean.py \
      --image-dir /path/to/images \
      --prompts   /path/to/prompts.json \
      --prompt    point \
      --output    /path/to/output_ma

  # box prompt
  python inference_ma_clean.py \
      --image-dir /path/to/images \
      --prompts   /path/to/prompts.json \
      --prompt    box \
      --output    /path/to/output_ma

  # text prompt（自动检测目标，最接近 zero-shot）
  python inference_ma_clean.py \
      --image-dir /path/to/images \
      --prompt    text \
      --text      "plant" \
      --output    /path/to/output_ma
"""

import os
import sys
import cv2
import json
import toml
import argparse
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from tqdm import tqdm

import utils
from utils import CONFIG
import networks

sys.path.insert(0, "./segment-anything")
sys.path.insert(0, "./GroundingDINO")

from segment_anything.utils.transforms import ResizeLongestSide
from groundingdino.util.inference import Model

TARGET_SIZE = 1024
transform   = ResizeLongestSide(TARGET_SIZE)

def inference(model, image_dict, os8_width=10, os4_width=20, os1_width=10):
    """
    完整的多尺度精修流程：
      1. os8：粗粒度 alpha，以 SAM post_mask 的未知区为参考范围
      2. os4：中粒度修正
      3. os1：细粒度边缘精修
    返回 uint8 灰度图，值域 [0, 255]
    """
    with torch.no_grad():
        feas, pred, post_mask = model.forward_inference(image_dict)

        alpha_os1 = pred["alpha_os1"]
        alpha_os4 = pred["alpha_os4"]
        alpha_os8 = pred["alpha_os8"]

        pad_h, pad_w = image_dict["pad_shape"]

        # 裁掉 padding
        alpha_os8 = alpha_os8[..., :pad_h, :pad_w]
        alpha_os4 = alpha_os4[..., :pad_h, :pad_w]
        alpha_os1 = alpha_os1[..., :pad_h, :pad_w]

        # 上采样回原始尺寸
        ori_shape = image_dict["ori_shape"]
        alpha_os8 = F.interpolate(alpha_os8, ori_shape, mode="bilinear", align_corners=False)
        alpha_os4 = F.interpolate(alpha_os4, ori_shape, mode="bilinear", align_corners=False)
        alpha_os1 = F.interpolate(alpha_os1, ori_shape, mode="bilinear", align_corners=False)

        # os8：纯模型输出，不混入 SAM mask
        alpha_pred = alpha_os8.clone()

        # os4 精修
        weight_os4 = utils.get_unknown_tensor_from_pred_oneside(
            alpha_pred, rand_width=os4_width, train_mode=False
        )
        alpha_pred[weight_os4 > 0] = alpha_os4[weight_os4 > 0]

        # os1 精修
        weight_os1 = utils.get_unknown_tensor_from_pred_oneside(
            alpha_pred, rand_width=os1_width, train_mode=False
        )
        alpha_pred[weight_os1 > 0] = alpha_os1[weight_os1 > 0]

    alpha = alpha_pred[0].cpu().numpy() * 255          # (1, H, W) → (H, W)
    return alpha.squeeze().astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# 图像预处理
# ─────────────────────────────────────────────────────────────

def load_and_preprocess(image_path):
    """
    读取 RGB 图像，归一化并 pad 到 1024x1024
    返回 (image_tensor, original_size, pad_size)
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w  = image.shape[:2]

    if h != TARGET_SIZE or w != TARGET_SIZE:
        raise ValueError(
            f"图像尺寸必须为 {TARGET_SIZE}x{TARGET_SIZE}，实际 {w}x{h}：{image_path}\n"
            f"请先用 gen_resize.py 将图像 resize 到 1024x1024。"
        )

    original_size = (h, w)
    image_t = transform.apply_image(image)                       # ndarray
    image_t = torch.as_tensor(image_t).cuda().permute(2, 0, 1)  # (3, H, W)

    pixel_mean = torch.tensor([123.675, 116.28,  103.53]).view(3,1,1).cuda()
    pixel_std  = torch.tensor([58.395,  57.12,   57.375]).view(3,1,1).cuda()
    image_t    = (image_t - pixel_mean) / pixel_std

    th, tw   = image_t.shape[-2:]
    pad_size = (th, tw)
    image_t  = F.pad(image_t, (0, TARGET_SIZE - tw, 0, TARGET_SIZE - th))

    return image_t, original_size, pad_size


# ─────────────────────────────────────────────────────────────
# 构建 image_dict（point / box prompt）
# ─────────────────────────────────────────────────────────────

def build_image_dict_point(image_path, points):
    """
    points: [[x, y, label], ...]   label: 1=前景 0=背景
    """
    image_t, original_size, pad_size = load_and_preprocess(image_path)

    pts = np.asarray(points, dtype=np.float32)          # (N, 3)
    point_coords = pts[:, :2]                           # (N, 2)
    point_labels = pts[:, 2]                            # (N,)

    point_coords = transform.apply_coords(point_coords, original_size)
    point_coords = torch.as_tensor(point_coords, dtype=torch.float).cuda()
    point_labels = torch.as_tensor(point_labels, dtype=torch.float).cuda()

    return {
        "image":     image_t[None],           # (1, 3, 1024, 1024)
        "point":     point_coords[None],      # (1, N, 2)
        "label":     point_labels[None],      # (1, N)
        "ori_shape": original_size,
        "pad_shape": pad_size,
    }


def build_image_dict_box(image_path, bbox):
    """
    bbox: [x1, y1, x2, y2]，坐标范围 [0, 1024]
    """
    image_t, original_size, pad_size = load_and_preprocess(image_path)

    bbox_arr = np.asarray(bbox, dtype=np.float32)[None, :]      # (1, 4)
    bbox_t   = transform.apply_boxes(bbox_arr, original_size)   # (1, 4)
    bbox_t   = torch.as_tensor(bbox_t, dtype=torch.float).cuda()

    return {
        "image":     image_t[None],       # (1, 3, 1024, 1024)
        "bbox":      bbox_t[None],        # (1, 1, 4)
        "ori_shape": original_size,
        "pad_shape": pad_size,
    }


def build_image_dict_text(image_path, text, dino_model, box_threshold, text_threshold, nms_threshold):
    """
    text prompt：用 GroundingDINO 检测目标框，再转 box prompt
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取: {image_path}")

    detections, _ = dino_model.predict_with_caption(
        image=image_bgr,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if len(detections.xyxy) == 0:
        return None   # 未检测到目标

    if len(detections.xyxy) > 1:
        keep = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold,
        ).numpy().tolist()
        detections.xyxy       = detections.xyxy[keep]
        detections.confidence = detections.confidence[keep]

    bbox = detections.xyxy[np.argmax(detections.confidence)].astype(np.float32)
    return build_image_dict_box(image_path, bbox.tolist())


# ─────────────────────────────────────────────────────────────
# 文件查找
# ─────────────────────────────────────────────────────────────

def find_image(image_dir, base, exts=('.png', '.jpg', '.jpeg')):
    stem = os.path.splitext(base)[0]
    for e in exts:
        p = os.path.join(image_dir, stem + e)
        if os.path.isfile(p):
            return p
    # 原始 base（可能已带扩展名）
    p = os.path.join(image_dir, base)
    if os.path.isfile(p):
        return p
    return None


# ─────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="MattingAnything 推理（精简版，用于与 ZIM 对比）"
    )
    parser.add_argument("--config",     type=str,
                        default="config/MAM-ViTB-8gpu.toml")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/mam_vitb.pth")
    parser.add_argument("--image-dir",  type=str, required=True,
                        help="输入图像目录（图像须为 1024x1024）")
    parser.add_argument("--output",     type=str, required=True,
                        help="输出 alpha 目录")
    parser.add_argument("--prompt",     type=str, default="point",
                        choices=["point", "box", "text"])
    # point / box prompt
    parser.add_argument("--prompts",    type=str, default=None,
                        help="prompts.json 路径（point/box prompt 必须提供）")
    # text prompt
    parser.add_argument("--text",       type=str, default="plant",
                        help="GroundingDINO 文本描述（text prompt 使用）")
    parser.add_argument("--box-threshold",  type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold",  type=float, default=0.8)
    parser.add_argument("--os8-width",  type=int, default=10)
    parser.add_argument("--os4-width",  type=int, default=20)
    parser.add_argument("--os1-width",  type=int, default=10)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU")

    if args.prompt in ("point", "box") and args.prompts is None:
        raise ValueError("--prompt point/box 必须指定 --prompts prompts.json")

    # 加载配置
    with open(args.config) as f:
        utils.load_config(toml.load(f))
    if CONFIG.is_default:
        raise ValueError("config 未正确加载")

    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    print("加载模型...")
    model = networks.get_generator_m2m(
        seg=CONFIG.model.arch.seg,
        m2m=CONFIG.model.arch.m2m,
    )
    model.cuda()
    ckpt = torch.load(args.checkpoint)
    model.m2m.load_state_dict(
        utils.remove_prefix_state_dict(ckpt["state_dict"]), strict=True
    )
    model.eval()
    print(f"参数量: {sum(p.numel() for p in model.m2m.parameters() if p.requires_grad):,}")

    skipped = []

    # ── point / box prompt ──────────────────────────────────
    if args.prompt in ("point", "box"):
        with open(args.prompts, encoding="utf-8") as f:
            prompts = json.load(f)

        for base, prompt in tqdm(prompts.items(), desc=f"[{args.prompt}]"):
            image_path = find_image(args.image_dir, base)
            if image_path is None:
                print(f"  ⚠ 找不到图像: {base}")
                skipped.append(base)
                continue

            try:
                if args.prompt == "point":
                    image_dict = build_image_dict_point(image_path, prompt["point"])
                else:
                    image_dict = build_image_dict_box(image_path, prompt["bbox"])

                alpha = inference(
                    model, image_dict,
                    os8_width=args.os8_width,
                    os4_width=args.os4_width,
                    os1_width=args.os1_width,
                )

                save_path = os.path.join(args.output, os.path.splitext(base)[0] + ".png")
                cv2.imwrite(save_path, alpha)

            except Exception as e:
                print(f"  ✗ {base}: {e}")
                skipped.append(base)

    # ── text prompt ─────────────────────────────────────────
    else:
        print("加载 GroundingDINO...")
        dino = Model(
            model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path="checkpoints/groundingdino_swint_ogc.pth",
        )

        image_exts = {".png", ".jpg", ".jpeg"}
        image_files = sorted(
            f for f in os.listdir(args.image_dir)
            if os.path.splitext(f)[1].lower() in image_exts
        )

        for fname in tqdm(image_files, desc="[text]"):
            image_path = os.path.join(args.image_dir, fname)
            base       = os.path.splitext(fname)[0]

            try:
                image_dict = build_image_dict_text(
                    image_path, args.text, dino,
                    args.box_threshold, args.text_threshold, args.nms_threshold,
                )

                if image_dict is None:
                    print(f"  ⚠ {base}: 未检测到目标（text='{args.text}'）")
                    skipped.append(base)
                    continue

                alpha = inference(
                    model, image_dict,
                    os8_width=args.os8_width,
                    os4_width=args.os4_width,
                    os1_width=args.os1_width,
                )

                cv2.imwrite(os.path.join(args.output, base + ".png"), alpha)

            except Exception as e:
                print(f"  ✗ {base}: {e}")
                skipped.append(base)

    # ── 汇总 ────────────────────────────────────────────────
    total = (len(prompts) if args.prompt in ("point", "box") else len(image_files))
    print(f"\n完成：{total - len(skipped)}/{total} 张成功，跳过 {len(skipped)} 张")
    print(f"结果保存至: {args.output}")
    if skipped:
        for s in skipped[:20]:
            print(f"  - {s}")
        if len(skipped) > 20:
            print(f"  ... 还有 {len(skipped)-20} 个")