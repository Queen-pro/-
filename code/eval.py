"""
eval.py
统一评测脚本：根据 GT alpha、固定 trimap 与一个或多个方法的预测 alpha
计算 SAD / MSE / MAD / Grad / Conn，用于
ViTMatte / ZIM / MattingAnything / Matte-Anything / SMat 等方法的统一对比。

使用方式：
1) 单方法评测：
   python eval.py --gt-dir GT_ALPHA --trimap-dir TRIMAP --pred-dir PRED_DIR

2) 多方法对比（推荐，保证所有方法用同一份 trimap）：
   python eval.py --gt-dir GT_ALPHA --trimap-dir TRIMAP \
       --methods ViTMatte=path/to/vitmatte_out \
                 ZIM=path/to/zim_out \
                 MattingAnything=path/to/ma_out \
                 MatteAnything=path/to/matte_anything_out \
                 SMat=path/to/smat_out \
       --out-dir results

输出：
- results/total.csv              所有方法的汇总指标（均值），按 model 名追加/更新行
- results/per_image/{model}.csv  每个方法的逐图指标，用于 failure case 分析

trimap 由 gen_trimap.py 预先从 GT alpha 生成，所有方法共用，
避免各方法因 trimap 不同导致指标不可比。
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
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy


def compute_gradient_loss(pred, target, trimap):
    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    loss = np.sum(error_map[trimap == 128])

    return loss / 1000.


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_connectivity_error(pred, target, trimap, step=0.1):
    pred = pred / 255.0
    target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float64) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int32)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int32)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.int32)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int32)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int32)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int32)
    loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])

    return loss / 1000.


def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))

    return loss / 1000, np.sum(trimap == 128) / 1000


def compute_mad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    return loss


def find_file(name_base, directory, exts=('.png', '.jpg', '.jpeg'), suffixes=('', '_alpha')):
    """
    name_base: 不含扩展名的文件名
    suffixes: 依次尝试的后缀，'' 表示同名，'_alpha' 表示 name_base + '_alpha' + ext
    """
    for suffix in suffixes:
        for ext in exts:
            p = os.path.join(directory, name_base + suffix + ext)
            if os.path.exists(p):
                return p
    return None


def evaluate_method(gt_dir, trimap_dir, pred_dir, per_image_csv_path=None):
    """
    对单个方法的预测结果在 gt_dir/trimap_dir 上计算指标。

    返回:
        summary: dict，汇总均值指标
        per_image_rows: list[dict]，每张图的逐图指标

    若 per_image_csv_path 不为 None，会将逐图指标写入该 csv 文件。
    """
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    sad_list, mse_list, mad_list, grad_list, conn_list = [], [], [], [], []
    per_image_rows = []
    n_valid = 0

    for gt_name in tqdm(gt_files, desc=os.path.basename(pred_dir.rstrip('/'))):
        base = os.path.splitext(gt_name)[0]

        gt_path = os.path.join(gt_dir, gt_name)
        pred_path = find_file(base, pred_dir)
        trimap_path = find_file(base, trimap_dir)

        if pred_path is None:
            print(f"  ⚠ [{pred_dir}] 找不到预测结果: {base}, 跳过")
            continue
        if trimap_path is None:
            print(f"  ⚠ 找不到 trimap: {base}, 跳过")
            continue

        gt = cv2.imread(gt_path, 0).astype(np.float64)
        pred = cv2.imread(pred_path, 0).astype(np.float64)
        trimap = cv2.imread(trimap_path, 0)

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        if trimap.shape != gt.shape:
            trimap = cv2.resize(trimap, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        n_unknown = np.sum(trimap == 128)
        if n_unknown == 0:
            print(f"  ⚠ {base}: trimap 未知区域为空，跳过")
            continue

        sad, _ = compute_sad_loss(pred, gt, trimap)
        mse = compute_mse_loss(pred, gt, trimap)
        mad = compute_mad_loss(pred, gt, trimap)
        grad = compute_gradient_loss(pred, gt, trimap)
        conn = compute_connectivity_error(pred, gt, trimap)

        # 与汇总表保持同一量纲（MSE/MAD ×1e3）
        mse_scaled = mse * 1e3
        mad_scaled = mad * 1e3

        sad_list.append(sad)
        mse_list.append(mse)
        mad_list.append(mad)
        grad_list.append(grad)
        conn_list.append(conn)
        n_valid += 1

        per_image_rows.append({
            'image': gt_name,
            'SAD': sad,
            'MSE': mse_scaled,
            'MAD': mad_scaled,
            'Grad': grad,
            'Conn': conn,
        })

    summary = {
        'n': n_valid,
        'SAD': float(np.mean(sad_list)) if n_valid else float('nan'),
        'MSE': float(np.mean(mse_list) * 1e3) if n_valid else float('nan'),
        'MAD': float(np.mean(mad_list) * 1e3) if n_valid else float('nan'),
        'Grad': float(np.mean(grad_list)) if n_valid else float('nan'),
        'Conn': float(np.mean(conn_list)) if n_valid else float('nan'),
    }

    if per_image_csv_path is not None:
        save_per_image_csv(per_image_rows, per_image_csv_path)

    return summary, per_image_rows


def save_per_image_csv(per_image_rows, csv_path):
    """
    将单个方法的逐图指标写入 csv 文件（每个方法一个文件，覆盖写）。
    列: image, SAD, MSE, MAD, Grad, Conn
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ['image', 'SAD', 'MSE', 'MAD', 'Grad', 'Conn']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow({
                'image': row['image'],
                'SAD': f"{row['SAD']:.6f}",
                'MSE': f"{row['MSE']:.6f}",
                'MAD': f"{row['MAD']:.6f}",
                'Grad': f"{row['Grad']:.6f}",
                'Conn': f"{row['Conn']:.6f}",
            })


def update_total_csv(csv_path, model_name, summary):
    """
    将单个模型的汇总指标写入/更新 total.csv。
    - 若文件不存在，新建并写入表头 + 该行
    - 若文件存在但 model_name 已有记录，更新该行（覆盖旧结果）
    - 若文件存在但 model_name 不存在，追加新行

    列: model, n, SAD, MSE, MAD, Grad, Conn
    """
    fieldnames = ['model', 'n', 'SAD', 'MSE', 'MAD', 'Grad', 'Conn']
    new_row = {
        'model': model_name,
        'n': summary['n'],
        'SAD': f"{summary['SAD']:.6f}",
        'MSE': f"{summary['MSE']:.6f}",
        'MAD': f"{summary['MAD']:.6f}",
        'Grad': f"{summary['Grad']:.6f}",
        'Conn': f"{summary['Conn']:.6f}",
    }

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

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


def parse_methods(method_args):
    """
    解析 --methods 参数，形如 ["ViTMatte=path1", "ZIM=path2", ...]
    返回 {name: path} 字典
    """
    methods = {}
    for item in method_args:
        if '=' not in item:
            raise ValueError(f"--methods 参数格式错误，应为 NAME=PATH，实际为: {item}")
        name, path = item.split('=', 1)
        methods[name] = path
    return methods


def print_summary_table(results):
    """
    results: {method_name: summary_dict}
    """
    headers = ['Method', 'n', 'SAD', 'MSE', 'MAD', 'Grad', 'Conn']
    col_w = [max(len(h), 12) for h in headers]
    col_w[0] = max(col_w[0], max(len(k) for k in results.keys()) + 2)

    def fmt_row(cells):
        return ' | '.join(str(c).ljust(w) for c, w in zip(cells, col_w))

    print("\n========== 评测结果 (均值) ==========")
    print(fmt_row(headers))
    print('-' * (sum(col_w) + 3 * (len(col_w) - 1)))
    for name, s in results.items():
        row = [
            name,
            s['n'],
            f"{s['SAD']:.4f}",
            f"{s['MSE']:.4f}",
            f"{s['MAD']:.4f}",
            f"{s['Grad']:.4f}",
            f"{s['Conn']:.4f}",
        ]
        print(fmt_row(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', type=str, default="/home/liangyin/zbk/rzkeshe/hardcase/alpha", help='GT alpha 目录')
    parser.add_argument('--trimap-dir', type=str, default="/home/liangyin/zbk/rzkeshe/hardcase/trimap",
                         help='trimap 目录')
    parser.add_argument('--pred-dir', type=str, default=None,
                         help='单方法预测 alpha 目录')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                         help='多方法对比，格式: NAME1=PATH1 NAME2=PATH2 ... '
                              '例如: ViTMatte=out_vitmatte ZIM=out_zim MattingAnything=out_ma '
                              'MatteAnything=out_matte_anything SMat=out_smat')
    parser.add_argument('--out-dir', type=str, default='results_hard',
                         help='结果输出目录，包含 total.csv 与 per_image/{model}.csv')
    args = parser.parse_args()

    if args.pred_dir is None and args.methods is None:
        parser.error("必须指定 --pred-dir 或 --methods 之一")

    os.makedirs(args.out_dir, exist_ok=True)
    per_image_dir = os.path.join(args.out_dir, 'per_image')
    total_csv_path = os.path.join(args.out_dir, 'total.csv')

    results = {}

    if args.methods is not None:
        methods = parse_methods(args.methods)
    else:
        name = os.path.basename(args.pred_dir.rstrip('/')) or 'pred'
        methods = {name: args.pred_dir}

    for name, pred_dir in methods.items():
        per_image_csv_path = os.path.join(per_image_dir, f'{name}.csv')
        summary, _ = evaluate_method(args.gt_dir, args.trimap_dir, pred_dir, per_image_csv_path)
        results[name] = summary
        update_total_csv(total_csv_path, name, summary)

    print_summary_table(results)
    print(f"\n汇总表已写入: {total_csv_path}")
    print(f"逐图指标已写入: {per_image_dir}/<model>.csv")