"""
vis_prompts.py
可视化 gen_prompts.py 生成的 bbox / point 线索，
将其叠加到原图与 GT alpha 上，检查 prompt 是否准确覆盖前景目标。

输出：每张图一张对比图 (原图+bbox+point | GT alpha+bbox+point)，
保存到 ./prompt/vis 目录。

用法：
python vis_prompts.py \
    --image-dir PM/images \
    --alpha-dir PM/alpha \
    --prompts prompts.json \
    --vis-dir ./prompt/vis
"""
import os
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def find_file(directory, base, exts=('.jpg', '.jpeg', '.png')):
    for ext in exts:
        p = os.path.join(directory, base + ext)
        if os.path.exists(p):
            return p
    return None


def draw_prompt(img, bbox, points, box_color=(0, 255, 255), point_color_pos=(0, 255, 0), point_color_neg=(0, 0, 255)):
    """
    在 img (BGR) 上画 bbox 矩形和多个 point 圆点
    bbox: [x1, y1, x2, y2]
    points: [[x, y, label], ...]  label=1 正样本(绿)，label=0 负样本(红)
    """
    out = img.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

    for px, py, label in points:
        pcolor = point_color_pos if label == 1 else point_color_neg
        cv2.circle(out, (px, py), 6, pcolor, -1)
        cv2.circle(out, (px, py), 6, (255, 255, 255), 1)

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default="/home/liangyin/zbk/rzkeshe/hardcase/image", help='原图目录')
    parser.add_argument('--alpha-dir', type=str, default="/home/liangyin/zbk/rzkeshe/hardcase/alpha", help='GT alpha 目录')
    parser.add_argument('--prompts', type=str, default="./prompts_hard.json", help='gen_prompts.py 生成的 prompts.json')
    parser.add_argument('--vis-dir', type=str, default='./prompt/vis_hard', help='可视化输出目录')
    parser.add_argument('--image-ext', type=str, nargs='+', default=['.jpg', '.jpeg', '.png'],
                         help='原图扩展名尝试顺序')
    parser.add_argument('--alpha-ext', type=str, nargs='+', default=['.png', '.jpg', '.jpeg'],
                         help='alpha 扩展名尝试顺序')
    parser.add_argument('--max-num', type=int, default=None, help='最多可视化多少张（默认全部）')
    args = parser.parse_args()

    os.makedirs(args.vis_dir, exist_ok=True)

    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    items = list(prompts.items())
    if args.max_num is not None:
        items = items[:args.max_num]

    skipped = []

    for base, p in tqdm(items):
        image_path = find_file(args.image_dir, base, tuple(args.image_ext))
        alpha_path = find_file(args.alpha_dir, base, tuple(args.alpha_ext))

        if image_path is None or alpha_path is None:
            print(f"  ⚠ 找不到图像或alpha: {base}, 跳过")
            skipped.append(base)
            continue

        image = cv2.imread(image_path)
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)

        if image is None or alpha is None:
            print(f"  ⚠ 读取失败: {base}, 跳过")
            skipped.append(base)
            continue

        alpha_bgr = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)

        bbox = p['bbox']
        points = p['point']  # [[x, y, label], ...]

        img_vis = draw_prompt(image, bbox, points)
        alpha_vis = draw_prompt(alpha_bgr, bbox, points)

        # 尺寸对齐（防止图像与alpha分辨率不一致）
        h = max(img_vis.shape[0], alpha_vis.shape[0])
        w_img, w_alpha = img_vis.shape[1], alpha_vis.shape[1]

        if img_vis.shape[0] != h:
            img_vis = cv2.resize(img_vis, (w_img, h))
        if alpha_vis.shape[0] != h:
            alpha_vis = cv2.resize(alpha_vis, (w_alpha, h))

        # 分隔条
        sep = np.full((h, 4, 3), 255, dtype=np.uint8)
        combined = np.concatenate([img_vis, sep, alpha_vis], axis=1)

        # 顶部标题条
        title_h = 30
        title = np.full((title_h, combined.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(title, base, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        combined = np.concatenate([title, combined], axis=0)

        out_path = os.path.join(args.vis_dir, base + '_prompt.png')
        cv2.imwrite(out_path, combined)

    print(f"\n完成。共可视化 {len(items) - len(skipped)} 张，跳过 {len(skipped)} 张。")
    print(f"结果保存至: {args.vis_dir}")