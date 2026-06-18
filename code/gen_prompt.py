"""
gen_prompts.py
从 GT alpha 批量提取统一的 prompt 线索（bbox / point），
供 ZIM / Matting Anything 等 prompt-based 方法共用，
保证三者在同一套先验信息下对比。

bbox: alpha 前景的最小外接矩形 [x1, y1, x2, y2]
point: 手动点击选取的若干正/负点 [[x, y, label], ...]
       label=1 表示前景(positive)点，label=0 表示背景(negative)点

交互方式：
- 显示原图（alpha 叠加在底图上，或直接显示 alpha）
- 鼠标左键点击 = 正样本点 (label=1)
- 鼠标右键点击 = 负样本点 (label=0)
- 按 'z' 撤销上一次点击
- 按 's' 或 Enter 保存当前图的标注并进入下一张
- 按 'n' 跳过当前图（不写入 prompts）
- 按 'q' 提前退出整个程序（已标注的会保存）

输出 prompts.json，格式：
{
  "photo1": {
    "bbox": [x1, y1, x2, y2],
    "point": [[x1, y1, 1], [x2, y2, 0], ...]
  },
  ...
}
"""
import os
import json
import argparse
import numpy as np
import cv2


def get_bbox_from_alpha(alpha, fg_thresh=127):
    """
    alpha: H×W uint8 灰度图
    返回: bbox [x1, y1, x2, y2]
    若 alpha 全为背景，返回 None
    """
    fg_mask = alpha > fg_thresh
    fg_set = np.where(fg_mask)
    if fg_set[0].size == 0:
        return None

    y1, y2 = int(fg_set[0].min()), int(fg_set[0].max())
    x1, x2 = int(fg_set[1].min()), int(fg_set[1].max())

    return [x1, y1, x2, y2]


class PointSelector:
    """交互式选点工具：左键正样本(1)，右键负样本(0)"""

    def __init__(self, image, window_name='select points (left=pos, right=neg, z=undo, s=save, n=skip, q=quit)'):
        self.image = image
        self.display = image.copy()
        self.window_name = window_name
        self.points = []  # [[x, y, label], ...]
        self.action = None  # 'save' | 'skip' | 'quit'

    def _redraw(self):
        self.display = self.image.copy()
        for x, y, label in self.points:
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # 绿=正, 红=负
            cv2.circle(self.display, (x, y), 6, color, -1)
            cv2.circle(self.display, (x, y), 6, (255, 255, 255), 1)
        cv2.imshow(self.window_name, self.display)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y, 1])
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y, 0])
            self._redraw()

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._redraw()

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('s'), 13):  # 's' 或 Enter
                self.action = 'save'
                break
            elif key == ord('z'):  # 撤销
                if self.points:
                    self.points.pop()
                    self._redraw()
            elif key == ord('n'):  # 跳过
                self.action = 'skip'
                break
            elif key == ord('q'):  # 退出
                self.action = 'quit'
                break

        cv2.destroyWindow(self.window_name)
        return self.points, self.action


def alpha_to_display(alpha):
    """将单通道 alpha 转为 3 通道图像用于显示"""
    return cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha-dir', type=str,
                        default="/home/liangyin/zbk/rzkeshe/hardcase/alpha",
                        help='GT alpha 目录')
    parser.add_argument('--image-dir', type=str, default="/home/liangyin/zbk/rzkeshe/hardcase/image",
                        help='原图目录（可选，若提供则在原图上标点，否则在 alpha 上标点）')
    parser.add_argument('--out-json', type=str, default='prompts_hard.json', help='输出 json 路径')
    parser.add_argument('--fg-thresh', type=int, default=127, help='前景判定阈值 (alpha > thresh 视为前景)')
    parser.add_argument('--resume', action='store_true', help='从已有 out-json 继续标注，跳过已存在的条目')
    args = parser.parse_args()

    alpha_files = sorted([f for f in os.listdir(args.alpha_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    prompts = {}
    if args.resume and os.path.exists(args.out_json):
        with open(args.out_json, 'r', encoding='utf-8') as f:
            prompts = json.load(f)

    skipped = []

    for fname in alpha_files:
        base = os.path.splitext(fname)[0]
        if args.resume and base in prompts:
            continue

        alpha_path = os.path.join(args.alpha_dir, fname)
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            print(f"  ⚠ 无法读取: {alpha_path}, 跳过")
            skipped.append(base)
            continue

        bbox = get_bbox_from_alpha(alpha, fg_thresh=args.fg_thresh)
        if bbox is None:
            print(f"  ⚠ {base}: alpha 全为背景，跳过")
            skipped.append(base)
            continue

        # 选择用于显示的底图
        if args.image_dir is not None:
            display_img = None
            for ext in ('.jpg', '.jpeg', '.png'):
                candidate = os.path.join(args.image_dir, base + ext)
                if os.path.exists(candidate):
                    display_img = cv2.imread(candidate)
                    break
            if display_img is None:
                display_img = alpha_to_display(alpha)
        else:
            display_img = alpha_to_display(alpha)

        # 画出 bbox 作为参考
        cv2.rectangle(display_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)

        print(f"\n[{base}] 请点击 3 个正样本点(左键)和 3 个负样本点(右键)，按 's'/Enter 保存，'n' 跳过，'q' 退出")

        selector = PointSelector(display_img)
        points, action = selector.run()

        if action == 'quit':
            print("用户退出，保存已标注内容。")
            break
        if action == 'skip':
            print(f"  跳过 {base}")
            skipped.append(base)
            continue

        if len(points) == 0:
            print(f"  ⚠ {base}: 未选取任何点，跳过")
            skipped.append(base)
            continue

        n_pos = sum(1 for p in points if p[2] == 1)
        n_neg = sum(1 for p in points if p[2] == 0)
        print(f"  已记录 {n_pos} 个正样本点, {n_neg} 个负样本点")

        prompts[base] = {
            'bbox': bbox,
            'point': points,
        }

        # 每张图标注完即时保存，防止意外中断丢失数据
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

    cv2.destroyAllWindows()

    print(f"\n完成。共生成 {len(prompts)} 条 prompt，跳过 {len(skipped)} 条。")
    print(f"已保存至: {args.out_json}")