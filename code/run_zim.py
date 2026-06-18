"""
run_zim.py

在 PM 数据集上批量运行 ZIM 推理。

只保留：
  1) box prompt only
  2) point prompt only

前提：
  image 和 alpha / prompts 均为 1024x1024 坐标系。

输出目录：
  out-dir/
    box/
    point/
"""

import os
import json
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from zim_anything import zim_model_registry, ZimPredictor


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def find_image_file(image_dir, base, exts=(".jpg", ".jpeg", ".png")):
    # 如果 base 本身带后缀
    direct_path = os.path.join(image_dir, base)
    if os.path.exists(direct_path):
        return direct_path

    # 如果 base 不带后缀
    stem, ext = os.path.splitext(base)
    if ext:
        return None

    for ext in exts:
        p = os.path.join(image_dir, base + ext)
        if os.path.exists(p):
            return p

    return None


def points_to_arrays(points):
    """
    points:
        [[x, y, label], ...]

    label:
        1 = positive foreground
        0 = negative background
    """
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid point format: expected Nx3, got {points.shape}")

    point_coords = points[:, :2].astype(np.float32)
    point_labels = points[:, 2].astype(np.int32)

    return point_coords, point_labels


def check_image_size(image, base, expected_size=1024):
    h, w = image.shape[:2]

    if h != expected_size or w != expected_size:
        print(f"\n[ERROR] {base}: image size is not {expected_size}x{expected_size}")
        print(f"        got: w={w}, h={h}")
        return False

    return True


def check_prompt_valid(bbox, point_coords, point_labels, base, image_w=1024, image_h=1024):
    ok = True

    bbox = np.asarray(bbox, dtype=np.float32)

    if bbox.shape != (4,):
        print(f"\n[ERROR] {base}: invalid bbox shape: {bbox.shape}")
        return False

    x1, y1, x2, y2 = bbox.tolist()

    if not (0 <= x1 < image_w and 0 <= y1 < image_h and 0 < x2 <= image_w and 0 < y2 <= image_h):
        print(f"\n[ERROR] {base}: bbox out of bounds")
        print(f"        bbox: {bbox.tolist()}")
        print(f"        image size: {image_w}x{image_h}")
        ok = False

    if x2 <= x1 or y2 <= y1:
        print(f"\n[ERROR] {base}: invalid bbox order")
        print(f"        bbox: {bbox.tolist()}")
        ok = False

    if point_coords is not None:
        if point_coords.ndim != 2 or point_coords.shape[1] != 2:
            print(f"\n[ERROR] {base}: invalid point_coords shape: {point_coords.shape}")
            ok = False
        else:
            xs = point_coords[:, 0]
            ys = point_coords[:, 1]

            if xs.min() < 0 or xs.max() >= image_w or ys.min() < 0 or ys.max() >= image_h:
                print(f"\n[ERROR] {base}: point coords out of bounds")
                print(f"        x range: {xs.min()} ~ {xs.max()}")
                print(f"        y range: {ys.min()} ~ {ys.max()}")
                print(f"        image size: {image_w}x{image_h}")
                ok = False

    if point_labels is not None:
        unique_labels = set(point_labels.tolist())
        if not unique_labels.issubset({0, 1}):
            print(f"\n[ERROR] {base}: point labels must be 0 or 1")
            print(f"        labels: {sorted(unique_labels)}")
            ok = False

    return ok


def mask_to_uint8(mask, invert=False):
    mask = np.asarray(mask)

    if mask.dtype == np.bool_:
        alpha = mask.astype(np.uint8) * 255
    else:
        mask = mask.astype(np.float32)

        if mask.max() <= 1.0:
            alpha = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        else:
            alpha = np.clip(mask, 0, 255).astype(np.uint8)

    if invert:
        alpha = 255 - alpha

    return alpha


def run_predict(predictor, point_coords=None, point_labels=None, box=None, invert=False):
    with torch.inference_mode():
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )

    alpha = mask_to_uint8(masks[0], invert=invert)

    score = None
    if scores is not None:
        score_arr = np.asarray(scores).reshape(-1)
        if len(score_arr) > 0:
            score = float(score_arr[0])

    return alpha, score


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dir",
        type=str,
        default="/home/liangyin/zbk/rzkeshe/hardcase/image",
        help="PM 图像目录，要求图像为 1024x1024",
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default="/home/liangyin/zbk/rzkeshe/prompts_hard.json",
        help="prompts.json 路径",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="results/zim_vit_l_2092",
        help="ZIM checkpoint 路径",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="vit_l",
        help="ZIM backbone",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs_zim_hard/l_2092",
        help="输出根目录",
    )

    parser.add_argument(
        "--image-ext",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="图像后缀尝试顺序",
    )

    parser.add_argument(
        "--expected-size",
        type=int,
        default=1024,
        help="期望输入尺寸，默认 1024",
    )

    parser.add_argument(
        "--invert-output",
        action="store_true",
        help="如果发现输出前景/背景极性反了，开启该选项",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_exts = tuple(
        ext if ext.startswith(".") else "." + ext
        for ext in args.image_ext
    )

    box_out_dir = os.path.join(args.out_dir, "box")
    point_out_dir = os.path.join(args.out_dir, "point")

    os.makedirs(box_out_dir, exist_ok=True)
    os.makedirs(point_out_dir, exist_ok=True)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print("加载 ZIM 模型...")
    model = zim_model_registry[args.backbone](checkpoint=args.ckpt)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    predictor = ZimPredictor(model)

    processed = 0
    skipped = []

    for base, p in tqdm(prompts.items()):
        image_path = find_image_file(args.image_dir, base, image_exts)

        if image_path is None:
            print(f"\n[WARN] 找不到图像: {base}")
            skipped.append(base)
            continue

        image = load_image(image_path)

        if image is None:
            print(f"\n[WARN] 无法读取图像: {image_path}")
            skipped.append(base)
            continue

        if not check_image_size(image, base, expected_size=args.expected_size):
            skipped.append(base)
            continue

        if "bbox" not in p:
            print(f"\n[WARN] {base}: prompts 中缺少 bbox")
            skipped.append(base)
            continue

        if "point" not in p:
            print(f"\n[WARN] {base}: prompts 中缺少 point")
            skipped.append(base)
            continue

        bbox = np.asarray(p["bbox"], dtype=np.float32)
        point_coords, point_labels = points_to_arrays(p["point"])

        valid = check_prompt_valid(
            bbox=bbox,
            point_coords=point_coords,
            point_labels=point_labels,
            base=base,
            image_w=args.expected_size,
            image_h=args.expected_size,
        )

        if not valid:
            skipped.append(base)
            continue

        predictor.set_image(image)

        save_name = os.path.splitext(base)[0] + ".png"

        # 1. box prompt only
        alpha_box, score_box = run_predict(
            predictor=predictor,
            box=bbox,
            invert=args.invert_output,
        )

        cv2.imwrite(
            os.path.join(box_out_dir, save_name),
            alpha_box,
        )

        # 2. point prompt only
        alpha_point, score_point = run_predict(
            predictor=predictor,
            point_coords=point_coords,
            point_labels=point_labels,
            invert=args.invert_output,
        )

        cv2.imwrite(
            os.path.join(point_out_dir, save_name),
            alpha_point,
        )

        processed += 1

        if score_box is not None and score_point is not None:
            print(
                f"{base}: "
                f"score_box={score_box:.4f}, "
                f"score_point={score_point:.4f}"
            )

    print("\n完成。")
    print(f"成功处理: {processed}")
    print(f"跳过: {len(skipped)}")
    print(f"box 结果保存至: {box_out_dir}")
    print(f"point 结果保存至: {point_out_dir}")

    if len(skipped) > 0:
        print("\n跳过样本:")
        for name in skipped[:50]:
            print("  -", name)
        if len(skipped) > 50:
            print(f"  ... 还有 {len(skipped) - 50} 个")