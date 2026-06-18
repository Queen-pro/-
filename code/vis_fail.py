"""
vis_failure_cases.py
基于逐图指标 CSV（eval.py 输出的 results/per_image/{model}.csv），
找出指定方法在某指标上表现最差的 Top-N 张图，
生成可视化对比图：原图 | GT alpha | 预测 alpha | 误差热力图，
用于 failure case 分析。

用法：
python vis_failure_cases.py \
    --csv results/per_image/ZIM_b_2043_point.csv \
    --metric SAD \
    --topn 10 \
    --image-dir PM/image \
    --alpha-dir PM/alpha \
    --pred-dir outputs/zim/point \
    --out-dir failure_vis/ZIM_b_2043_point_SAD
"""
import os
import argparse
import csv
import numpy as np
import cv2
from tqdm import tqdm


def find_file(directory, base, exts=('.jpg', '.jpeg', '.png'), suffixes=('',)):
    """
    在 directory 下按 suffixes（如 '', '_alpha'）和 exts 组合依次尝试匹配
    base + suffix + ext
    """
    for suffix in suffixes:
        for ext in exts:
            p = os.path.join(directory, base + suffix + ext)
            if os.path.exists(p):
                return p
    return None


def load_per_image_csv(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def error_heatmap(pred, gt, trimap=None):
    """
    pred, gt: float64 灰度图 (0~255)
    返回: BGR 热力图 (uint8)，误差越大越红
    若提供 trimap，仅在未知区域(128)显示误差，其余区域置黑
    """
    err = np.abs(pred - gt) / 255.0  # 0~1
    err_uint8 = (err * 255).astype(np.uint8)
    heat = cv2.applyColorMap(err_uint8, cv2.COLORMAP_JET)

    if trimap is not None:
        mask = (trimap == 128)
        heat_masked = np.zeros_like(heat)
        heat_masked[mask] = heat[mask]
        return heat_masked

    return heat


def put_title(img, text, color=(0, 0, 0)):
    """在图像下方添加标题条"""
    h, w = img.shape[:2]
    title_h = 30
    title = np.full((title_h, w, 3), 255, dtype=np.uint8)
    cv2.putText(title, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return np.concatenate([img, title], axis=0)


def to_bgr(gray):
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default="./results/per_image/ViTMatte_B_DIS.csv", help='单个方法的逐图指标 csv (eval.py 输出)')
    parser.add_argument('--metric', type=str, default='SAD',
                         choices=['SAD', 'MSE', 'MAD', 'Grad', 'Conn'], help='按该指标排序，取最差的 topn 张')
    parser.add_argument('--topn', type=int, default=150, help='可视化数量')
    parser.add_argument('--image-dir', type=str, default="./ViTMatte/data/keshe/image", help='原图目录')
    parser.add_argument('--alpha-dir', type=str, default="./ViTMatte/data/keshe/alpha", help='GT alpha 目录')
    parser.add_argument('--pred-dir', type=str, default="./ViTMatte/data/result/output_B_DIS", help='预测 alpha 目录')
    parser.add_argument('--trimap-dir', type=str, default=None,
                         help='trimap 目录（可选，提供则误差热力图仅显示未知区域）')
    parser.add_argument('--out-dir', type=str, default="./failcase/ViTMatte_B_DIS", help='可视化输出目录')
    parser.add_argument('--image-ext', type=str, nargs='+', default=['.jpg', '.jpeg', '.png'])
    parser.add_argument('--alpha-ext', type=str, nargs='+', default=['.png', '.jpg', '.jpeg'])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_per_image_csv(args.csv)

    # 按指标降序排序，取最差的 topn
    rows_sorted = sorted(rows, key=lambda r: float(r[args.metric]), reverse=True)
    top_rows = rows_sorted[:args.topn]

    print(f"Top-{args.topn} 最差样本（按 {args.metric} 降序）：")
    for r in top_rows:
        print(f"  {r['image']}  {args.metric}={r[args.metric]}")

    summary_lines = [f"Top-{args.topn} worst samples by {args.metric}", ""]

    for rank, r in enumerate(tqdm(top_rows), start=1):
        gt_name = r['image']
        base = os.path.splitext(gt_name)[0]

        image_path = find_file(args.image_dir, base, tuple(args.image_ext))
        alpha_path = find_file(args.alpha_dir, base, tuple(args.alpha_ext))
        pred_path = find_file(args.pred_dir, base, tuple(args.alpha_ext), suffixes=('', '_alpha'))

        if image_path is None or alpha_path is None or pred_path is None:
            print(f"  ⚠ {base}: 缺少文件 (image/alpha/pred)，跳过")
            continue

        image = cv2.imread(image_path)
        gt = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # 尺寸对齐
        h, w = gt.shape
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

        trimap = None
        if args.trimap_dir is not None:
            trimap_path = find_file(args.trimap_dir, base, tuple(args.alpha_ext))
            if trimap_path is not None:
                trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
                if trimap.shape != (h, w):
                    trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)

        gt_f = gt.astype(np.float64)
        pred_f = pred.astype(np.float64)

        heat = error_heatmap(pred_f, gt_f, trimap)

        # 拼图：原图 | GT alpha | 预测 alpha | 误差热力图
        panels = [
            put_title(image, 'Image'),
            put_title(to_bgr(gt), 'GT alpha'),
            put_title(to_bgr(pred), 'Pred alpha'),
            put_title(heat, 'Error heatmap'),
        ]

        sep = np.full((panels[0].shape[0], 4, 3), 255, dtype=np.uint8)
        combined = panels[0]
        for p in panels[1:]:
            combined = np.concatenate([combined, sep, p], axis=1)

        # 顶部信息条
        metric_val = r[args.metric]
        info_text = f"#{rank}  {gt_name}  {args.metric}={metric_val}"
        info_h = 32
        info = np.full((info_h, combined.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(info, info_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        combined = np.concatenate([info, combined], axis=0)

        out_path = os.path.join(args.out_dir, f"rank{rank:02d}_{base}.png")
        cv2.imwrite(out_path, combined)

        summary_lines.append(f"#{rank}  {gt_name}  {args.metric}={metric_val}")

    with open(os.path.join(args.out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f"\n完成。结果保存至: {args.out_dir}")