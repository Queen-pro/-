"""
eval_zeroshot.py
Zero-shot 评测脚本：不依赖 trimap，对全图所有像素计算指标。
专用于对比 ZIM 与 MattingAnything 等无需 trimap 输入的方法。

使用方式：
1) 单方法评测：
   python eval_zeroshot.py --gt-dir GT_ALPHA --pred-dir PRED_DIR

2) 多方法对比（推荐）：
   python eval_zeroshot.py \
       --gt-dir /path/to/alpha \
       --methods ZIM=/path/to/zim_out \
                 MattingAnything=/path/to/ma_out \
       --out-dir results_zeroshot

输出：
- results_zeroshot/total.csv              汇总均值指标（与实验一 total.csv 格式完全一致）
- results_zeroshot/per_image/{model}.csv  逐图指标，用于 failure case 分析

注意：
- 全图评估 = 构造全 128 trimap，让所有像素参与计算
- 与实验一（trimap-based）的 total.csv 格式相同，可直接合并报告
- MSE / MAD 列已 ×1e3，与实验一量纲一致
"""

import os
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import scipy.ndimage
from skimage.measure import label

def gauss(x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def dgauss(x, sigma):
    return -x * gauss(x, sigma) / (sigma ** 2)

def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(
        sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))
    ).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)
    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()
    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')
    return gx, gy


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_gradient_loss(pred, target, trimap):
    pred   = pred   / 255.0
    target = target / 255.0
    pred_x,   pred_y   = gaussgradient(pred,   1.4)
    target_x, target_y = gaussgradient(target, 1.4)
    pred_amp   = np.sqrt(pred_x   ** 2 + pred_y   ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)
    error_map  = (pred_amp - target_amp) ** 2
    return np.sum(error_map[trimap == 128]) / 1000.


def compute_connectivity_error(pred, target, trimap, step=0.1):
    pred   = pred   / 255.0
    target = target / 255.0
    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float64) * -1
    for i in range(1, len(thresh_steps)):
        pred_thresh   = (pred   >= thresh_steps[i]).astype(np.int32)
        target_thresh = (target >= thresh_steps[i]).astype(np.int32)
        omega = getLargestCC(pred_thresh * target_thresh).astype(np.int32)
        flag  = ((l_map == -1) & (omega == 0)).astype(np.int32)
        l_map[flag == 1] = thresh_steps[i - 1]
    l_map[l_map == -1] = 1
    pred_d    = pred   - l_map
    target_d  = target - l_map
    pred_phi   = 1 - pred_d   * (pred_d   >= 0.15).astype(np.int32)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int32)
    return np.sum(np.abs(pred_phi - target_phi)[trimap == 128]) / 1000.


def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    return np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))
    return loss / 1000, np.sum(trimap == 128) / 1000


def compute_mad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    return np.sum(error_map * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)


# ─────────────────────────────────────────────────────────────
# 文件查找（同实验一）
# ─────────────────────────────────────────────────────────────

def find_file(name_base, directory, exts=('.png', '.jpg', '.jpeg'), suffixes=('', '_alpha')):
    for suffix in suffixes:
        for ext in exts:
            p = os.path.join(directory, name_base + suffix + ext)
            if os.path.exists(p):
                return p
    return None


# ─────────────────────────────────────────────────────────────
# 核心评测：Zero-shot 全图模式
# ─────────────────────────────────────────────────────────────

def evaluate_method_zeroshot(gt_dir, pred_dir, per_image_csv_path=None):
    """
    Zero-shot 全图评测：
    - 不依赖 trimap，构造全 128 trimap 让所有像素参与计算
    - 适用于 ZIM / MattingAnything 等无需 trimap 的方法
    - 指标定义与实验一完全一致，量纲相同

    返回:
        summary: dict，汇总均值指标
        per_image_rows: list[dict]，逐图指标
    """
    gt_files = sorted([
        f for f in os.listdir(gt_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    sad_list, mse_list, mad_list, grad_list, conn_list = [], [], [], [], []
    per_image_rows = []
    n_skip = 0

    for gt_name in tqdm(gt_files, desc=f"[zeroshot] {os.path.basename(pred_dir.rstrip('/'))}"):
        base     = os.path.splitext(gt_name)[0]
        gt_path  = os.path.join(gt_dir, gt_name)
        pred_path = find_file(base, pred_dir)

        if pred_path is None:
            print(f"  ⚠ 找不到预测结果: {base}，跳过")
            n_skip += 1
            continue

        gt   = cv2.imread(gt_path,   0).astype(np.float64)
        pred = cv2.imread(pred_path, 0).astype(np.float64)

        if pred is None:
            print(f"  ⚠ 无法读取预测文件: {pred_path}，跳过")
            n_skip += 1
            continue

        # 尺寸不一致时对齐到 GT
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        # ── 关键：全图 trimap，所有像素值设为 128 ──────────────
        trimap_full = np.full_like(gt, fill_value=128, dtype=np.float64)
        # ────────────────────────────────────────────────────────

        sad,  _ = compute_sad_loss(pred, gt, trimap_full)
        mse     = compute_mse_loss(pred, gt, trimap_full)
        mad     = compute_mad_loss(pred, gt, trimap_full)
        grad    = compute_gradient_loss(pred, gt, trimap_full)
        conn    = compute_connectivity_error(pred, gt, trimap_full)

        sad_list.append(sad)
        mse_list.append(mse)
        mad_list.append(mad)
        grad_list.append(grad)
        conn_list.append(conn)

        per_image_rows.append({
            'image': gt_name,
            'SAD':   sad,
            'MSE':   mse * 1e3,   # ×1e3，与实验一量纲一致
            'MAD':   mad * 1e3,
            'Grad':  grad,
            'Conn':  conn,
        })

    n_valid = len(sad_list)
    if n_skip:
        print(f"  共跳过 {n_skip} 张，有效 {n_valid} 张")

    summary = {
        'n':    n_valid,
        'SAD':  float(np.mean(sad_list))        if n_valid else float('nan'),
        'MSE':  float(np.mean(mse_list) * 1e3)  if n_valid else float('nan'),
        'MAD':  float(np.mean(mad_list) * 1e3)  if n_valid else float('nan'),
        'Grad': float(np.mean(grad_list))       if n_valid else float('nan'),
        'Conn': float(np.mean(conn_list))       if n_valid else float('nan'),
    }

    if per_image_csv_path is not None:
        _save_per_image_csv(per_image_rows, per_image_csv_path)

    return summary, per_image_rows


# ─────────────────────────────────────────────────────────────
# CSV 输出（格式与实验一完全一致）
# ─────────────────────────────────────────────────────────────

def _save_per_image_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ['image', 'SAD', 'MSE', 'MAD', 'Grad', 'Conn']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'image': row['image'],
                'SAD':   f"{row['SAD']:.6f}",
                'MSE':   f"{row['MSE']:.6f}",
                'MAD':   f"{row['MAD']:.6f}",
                'Grad':  f"{row['Grad']:.6f}",
                'Conn':  f"{row['Conn']:.6f}",
            })


def update_total_csv(csv_path, model_name, summary):
    """
    写入/更新 total.csv，格式与实验一完全一致。
    - 已存在的 model 行：覆盖更新
    - 新 model：追加
    """
    fieldnames = ['model', 'n', 'SAD', 'MSE', 'MAD', 'Grad', 'Conn']
    new_row = {
        'model': model_name,
        'n':     summary['n'],
        'SAD':   f"{summary['SAD']:.6f}",
        'MSE':   f"{summary['MSE']:.6f}",
        'MAD':   f"{summary['MAD']:.6f}",
        'Grad':  f"{summary['Grad']:.6f}",
        'Conn':  f"{summary['Conn']:.6f}",
    }

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))

    found = False
    for row in rows:
        if row['model'] == model_name:
            row.update(new_row)
            found = True
            break
    if not found:
        rows.append(new_row)

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


# ─────────────────────────────────────────────────────────────
# 打印汇总表
# ─────────────────────────────────────────────────────────────

def print_summary_table(results):
    headers = ['Method', 'n', 'SAD↓', 'MSE↓', 'MAD↓', 'Grad↓', 'Conn↓']
    col_w   = [max(len(h), 14) for h in headers]
    col_w[0] = max(col_w[0], max(len(k) for k in results) + 2)

    def fmt_row(cells):
        return ' | '.join(str(c).ljust(w) for c, w in zip(cells, col_w))

    sep = '-' * (sum(col_w) + 3 * (len(col_w) - 1))
    print("\n========== Zero-shot 评测结果（全图，均值）==========")
    print("注：MSE / MAD 已 ×1e3，与 trimap-based 实验一量纲一致")
    print(fmt_row(headers))
    print(sep)
    for name, s in results.items():
        print(fmt_row([
            name,
            s['n'],
            f"{s['SAD']:.4f}",
            f"{s['MSE']:.4f}",
            f"{s['MAD']:.4f}",
            f"{s['Grad']:.4f}",
            f"{s['Conn']:.4f}",
        ]))
    print(sep)


# ─────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────

def parse_methods(method_args):
    methods = {}
    for item in method_args:
        if '=' not in item:
            raise ValueError(f"--methods 格式错误，应为 NAME=PATH，实际为: {item}")
        name, path = item.split('=', 1)
        methods[name] = path
    return methods


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Zero-shot matting eval：ZIM vs MattingAnything'
    )
    parser.add_argument('--gt-dir', type=str, default="/home/liangyin/zbk/rzkeshe/hardcase/alpha", help='GT alpha 目录')
    parser.add_argument('--pred-dir', type=str, default=None,
                        help='单方法预测目录')
    parser.add_argument('--methods',  type=str, nargs='+', default=None,
                        help='多方法对比，格式: NAME=PATH，例如:\n'
                             '  ZIM=/out/zim  MattingAnything=/out/ma')
    parser.add_argument('--out-dir',  type=str, default='./results_hard',
                        help='结果输出目录（默认: results_zeroshot）')
    args = parser.parse_args()

    if args.pred_dir is None and args.methods is None:
        parser.error("必须指定 --pred-dir 或 --methods 之一")

    os.makedirs(args.out_dir, exist_ok=True)
    per_image_dir  = os.path.join(args.out_dir, 'per_image')
    total_csv_path = os.path.join(args.out_dir, 'total.csv')

    if args.methods is not None:
        methods = parse_methods(args.methods)
    else:
        name    = os.path.basename(args.pred_dir.rstrip('/')) or 'pred'
        methods = {name: args.pred_dir}

    results = {}
    for name, pred_dir in methods.items():
        per_image_csv = os.path.join(per_image_dir, f'{name}.csv')
        summary, _    = evaluate_method_zeroshot(args.gt_dir, pred_dir, per_image_csv)
        results[name] = summary
        update_total_csv(total_csv_path, name, summary)

    print_summary_table(results)
    print(f"\n汇总表: {total_csv_path}")
    print(f"逐图指标: {per_image_dir}/<model>.csv")