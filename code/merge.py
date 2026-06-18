import os
import csv
import cv2
import argparse
import numpy as np
from tqdm import tqdm


# ============================================================
# 支持 Windows 中文路径的 imread / imwrite
# ============================================================

def cv_imread(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def cv_imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False

    buf.tofile(path)
    return True


# ============================================================
# 文件匹配
# ============================================================

def find_file_by_stem(folder, stem, exts):
    """
    优先精确匹配：
        0001.png

    如果找不到，再宽松匹配：
        0001_醒图.png
        0001_cutout.png
        醒图_0001.png
    """
    for ext in exts:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p

    candidates = []
    for name in os.listdir(folder):
        file_stem, file_ext = os.path.splitext(name)
        if file_ext.lower() not in exts:
            continue
        if stem in file_stem:
            candidates.append(os.path.join(folder, name))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print(f"[WARN] {stem}: 找到多个候选抠图文件，使用第一个: {candidates[0]}")
        return candidates[0]

    return None


def list_images(folder, exts):
    files = []
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if ext.lower() in exts:
            files.append((stem, os.path.join(folder, name)))
    return sorted(files)


# ============================================================
# Alpha 裁剪、缩放、bbox
# ============================================================

def alpha_bbox(alpha, alpha_thr=10):
    ys, xs = np.where(alpha > alpha_thr)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    return int(x1), int(y1), int(x2), int(y2)


def tight_crop_rgba(bgra, alpha_thr=10):
    alpha = bgra[:, :, 3]
    box = alpha_bbox(alpha, alpha_thr=alpha_thr)

    if box is None:
        raise ValueError("cutout PNG 中没有有效 alpha 区域")

    x1, y1, x2, y2 = box
    cropped = bgra[y1:y2, x1:x2].copy()

    return cropped, box


def resize_bgra(bgra, scale):
    h, w = bgra.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(bgra, (new_w, new_h), interpolation=interp)


# ============================================================
# 候选生成：每个 scale 取 top-k，而不是只取最高点
# ============================================================

def nms_topk_from_response(response, top_k=10, suppress_radius=30):
    """
    从 matchTemplate 的 response map 中取 top-k 个局部最大位置。
    """
    response = response.copy()
    candidates = []

    h, w = response.shape[:2]

    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        x, y = max_loc

        if max_val < -0.5:
            break

        candidates.append((float(max_val), int(x), int(y)))

        x1 = max(0, x - suppress_radius)
        x2 = min(w, x + suppress_radius + 1)
        y1 = max(0, y - suppress_radius)
        y2 = min(h, y + suppress_radius + 1)

        response[y1:y2, x1:x2] = -1.0

    return candidates


def generate_candidates_one_scale(
    original_bgr,
    cutout_bgra,
    scale,
    top_k=10,
    alpha_thr=10,
):
    """
    在单个 scale 下生成多个候选匹配位置。
    """
    oh, ow = original_bgr.shape[:2]
    th, tw = cutout_bgra.shape[:2]

    if th > oh or tw > ow:
        return []

    template_bgr = cutout_bgra[:, :, :3]
    template_alpha = cutout_bgra[:, :, 3]

    mask = (template_alpha > alpha_thr).astype(np.uint8) * 255
    if mask.sum() == 0:
        return []

    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(
        original_gray,
        template_gray,
        cv2.TM_CCORR_NORMED,
        mask=mask
    )

    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)

    suppress_radius = max(10, min(th, tw) // 8)
    top_locs = nms_topk_from_response(
        response=result,
        top_k=top_k,
        suppress_radius=suppress_radius,
    )

    candidates = []

    for corr_score, x, y in top_locs:
        candidates.append({
            "x": int(x),
            "y": int(y),
            "w": int(tw),
            "h": int(th),
            "scale": float(scale),
            "corr_score": float(corr_score),
            "alpha": template_alpha,
            "cutout_bgra": cutout_bgra,
        })

    return candidates


# ============================================================
# 候选重排序：边缘 + 颜色 + 位置先验
# ============================================================

def compute_edge_score(original_bgr, cutout_bgra, x, y, alpha_thr=10, edge_sigma=4.0):
    """
    边缘对齐分数。

    思路：
      1. 对模板前景区域提取 Canny 边缘；
      2. 对原图对应 ROI 提取 Canny 边缘；
      3. 使用 distance transform 计算模板边缘到原图边缘的平均距离；
      4. 距离越小，score 越高。

    返回范围大致为 [0, 1]。
    """
    h, w = cutout_bgra.shape[:2]
    roi = original_bgr[y:y + h, x:x + w]

    if roi.shape[:2] != (h, w):
        return 0.0

    template_bgr = cutout_bgra[:, :, :3]
    template_alpha = cutout_bgra[:, :, 3]
    mask = template_alpha > alpha_thr

    if mask.sum() == 0:
        return 0.0

    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    template_edge = cv2.Canny(template_gray, 50, 150)
    roi_edge = cv2.Canny(roi_gray, 50, 150)

    template_edge[~mask] = 0

    edge_pts = template_edge > 0
    if edge_pts.sum() < 20:
        return 0.0

    # distanceTransform 要求前景为 0，背景为非 0
    inv_roi_edge = np.where(roi_edge > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv_roi_edge, cv2.DIST_L2, 3)

    mean_dist = float(dist[edge_pts].mean())

    edge_score = float(np.exp(-mean_dist / edge_sigma))
    return edge_score


def compute_color_score(original_bgr, cutout_bgra, x, y, alpha_thr=10):
    """
    前景区域颜色一致性分数。
    醒图如果没有明显调色，该分数能帮助区分相似叶片位置。
    """
    h, w = cutout_bgra.shape[:2]
    roi = original_bgr[y:y + h, x:x + w]

    if roi.shape[:2] != (h, w):
        return 0.0

    template_bgr = cutout_bgra[:, :, :3].astype(np.float32)
    template_alpha = cutout_bgra[:, :, 3]
    mask = template_alpha > alpha_thr

    if mask.sum() == 0:
        return 0.0

    roi_bgr = roi.astype(np.float32)

    diff = np.abs(template_bgr[mask] - roi_bgr[mask]).mean()
    score = 1.0 - diff / 255.0

    return float(np.clip(score, 0.0, 1.0))


def compute_prior_score(candidate, prior_bbox, prior_sigma=200.0):
    """
    可选位置先验分数。

    prior_bbox 来自醒图同画布 PNG 的原始 alpha bbox。
    如果醒图只是轻微平移或缩放，用这个先验可以避免匹配到远处相似植物。
    """
    if prior_bbox is None:
        return 0.0

    px1, py1, px2, py2 = prior_bbox
    prior_cx = (px1 + px2) / 2.0
    prior_cy = (py1 + py2) / 2.0

    cand_cx = candidate["x"] + candidate["w"] / 2.0
    cand_cy = candidate["y"] + candidate["h"] / 2.0

    dist = np.sqrt((cand_cx - prior_cx) ** 2 + (cand_cy - prior_cy) ** 2)

    score = np.exp(-(dist ** 2) / (2.0 * prior_sigma ** 2))
    return float(score)


def rerank_candidates(
    original_bgr,
    candidates,
    prior_bbox=None,
    use_prior=False,
    alpha_thr=10,
    w_corr=0.50,
    w_edge=0.25,
    w_color=0.15,
    w_prior=0.10,
    prior_sigma=200.0,
):
    """
    对所有 scale 的候选位置进行重排序。
    """
    if not candidates:
        return None

    best = None

    for cand in candidates:
        x = cand["x"]
        y = cand["y"]
        cutout_bgra = cand["cutout_bgra"]

        corr_score = cand["corr_score"]

        edge_score = compute_edge_score(
            original_bgr=original_bgr,
            cutout_bgra=cutout_bgra,
            x=x,
            y=y,
            alpha_thr=alpha_thr,
        )

        color_score = compute_color_score(
            original_bgr=original_bgr,
            cutout_bgra=cutout_bgra,
            x=x,
            y=y,
            alpha_thr=alpha_thr,
        )

        prior_score = 0.0
        if use_prior:
            prior_score = compute_prior_score(
                candidate=cand,
                prior_bbox=prior_bbox,
                prior_sigma=prior_sigma,
            )

        final_score = (
            w_corr * corr_score +
            w_edge * edge_score +
            w_color * color_score +
            (w_prior * prior_score if use_prior else 0.0)
        )

        cand["edge_score"] = float(edge_score)
        cand["color_score"] = float(color_score)
        cand["prior_score"] = float(prior_score)
        cand["final_score"] = float(final_score)

        if best is None or cand["final_score"] > best["final_score"]:
            best = cand

    return best


def match_cutout_to_original_v2(
    original_bgr,
    cutout_bgra,
    scales,
    alpha_thr=10,
    top_k=10,
    use_prior=False,
    prior_sigma=200.0,
    w_corr=0.50,
    w_edge=0.25,
    w_color=0.15,
    w_prior=0.10,
):
    """
    改进版匹配：
      1. 先裁剪 alpha 有效区域；
      2. 多 scale；
      3. 每个 scale 取 top-k 候选；
      4. 使用 corr + edge + color + prior 重排序。
    """
    raw_alpha = cutout_bgra[:, :, 3]
    prior_bbox = alpha_bbox(raw_alpha, alpha_thr=alpha_thr)

    cutout_crop, crop_bbox = tight_crop_rgba(
        cutout_bgra,
        alpha_thr=alpha_thr,
    )

    all_candidates = []

    for scale in scales:
        scaled = resize_bgra(cutout_crop, scale)

        candidates = generate_candidates_one_scale(
            original_bgr=original_bgr,
            cutout_bgra=scaled,
            scale=scale,
            top_k=top_k,
            alpha_thr=alpha_thr,
        )

        all_candidates.extend(candidates)

    if not all_candidates:
        raise ValueError("所有 scale 均匹配失败")

    best = rerank_candidates(
        original_bgr=original_bgr,
        candidates=all_candidates,
        prior_bbox=prior_bbox,
        use_prior=use_prior,
        alpha_thr=alpha_thr,
        w_corr=w_corr,
        w_edge=w_edge,
        w_color=w_color,
        w_prior=w_prior,
        prior_sigma=prior_sigma,
    )

    if best is None:
        raise ValueError("候选重排序失败")

    # 为了兼容旧日志字段
    best["score"] = best["final_score"]

    return best


def make_full_alpha(original_bgr, match_result):
    """
    根据匹配结果，把缩放后的 alpha 贴回原图大小画布。
    """
    oh, ow = original_bgr.shape[:2]

    x = match_result["x"]
    y = match_result["y"]
    w = match_result["w"]
    h = match_result["h"]
    alpha = match_result["alpha"]

    full_alpha = np.zeros((oh, ow), dtype=np.uint8)

    x2 = min(ow, x + w)
    y2 = min(oh, y + h)

    paste_w = x2 - x
    paste_h = y2 - y

    if paste_w <= 0 or paste_h <= 0:
        raise ValueError("匹配位置超出原图范围")

    full_alpha[y:y2, x:x2] = alpha[:paste_h, :paste_w]

    return full_alpha


# ============================================================
# Debug 可视化
# ============================================================

def save_debug_image(original_bgr, full_alpha, output_path):
    """
    保存叠加图：
        原图 + 红色 alpha 区域 + 红色轮廓
    """
    vis = original_bgr.copy()

    overlay = np.zeros_like(vis)
    overlay[:, :, 2] = full_alpha

    vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)

    contours, _ = cv2.findContours(
        (full_alpha > 10).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv_imwrite(output_path, vis)


# ============================================================
# 单张处理
# ============================================================

def process_one(
    stem,
    original_path,
    cutout_path,
    alpha_out_dir,
    debug_out_dir,
    scales,
    save_debug=False,
    min_score=0.35,
    force_match=False,
    alpha_thr=10,
    top_k=10,
    use_prior=False,
    prior_sigma=200.0,
    w_corr=0.50,
    w_edge=0.25,
    w_color=0.15,
    w_prior=0.10,
):
    original_bgr = cv_imread(original_path, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"无法读取原图: {original_path}")

    cutout_bgra = cv_imread(cutout_path, cv2.IMREAD_UNCHANGED)
    if cutout_bgra is None:
        raise FileNotFoundError(f"无法读取醒图 PNG: {cutout_path}")

    if cutout_bgra.ndim != 3 or cutout_bgra.shape[2] != 4:
        raise ValueError(f"醒图抠图结果必须是带透明通道的 PNG: {cutout_path}")

    oh, ow = original_bgr.shape[:2]
    ch, cw = cutout_bgra.shape[:2]

    if oh == ch and ow == cw and not force_match:
        full_alpha = cutout_bgra[:, :, 3]

        match_result = {
            "x": 0,
            "y": 0,
            "w": ow,
            "h": oh,
            "scale": 1.0,
            "corr_score": 1.0,
            "edge_score": 1.0,
            "color_score": 1.0,
            "prior_score": 1.0,
            "final_score": 1.0,
            "score": 1.0,
        }

        mode = "same_canvas"

    else:
        match_result = match_cutout_to_original_v2(
            original_bgr=original_bgr,
            cutout_bgra=cutout_bgra,
            scales=scales,
            alpha_thr=alpha_thr,
            top_k=top_k,
            use_prior=use_prior,
            prior_sigma=prior_sigma,
            w_corr=w_corr,
            w_edge=w_edge,
            w_color=w_color,
            w_prior=w_prior,
        )

        full_alpha = make_full_alpha(
            original_bgr=original_bgr,
            match_result=match_result,
        )

        mode = "force_matched_v2" if force_match else "matched_v2"

    os.makedirs(alpha_out_dir, exist_ok=True)
    alpha_path = os.path.join(alpha_out_dir, stem + ".png")
    cv_imwrite(alpha_path, full_alpha)

    if save_debug:
        debug_path = os.path.join(debug_out_dir, stem + "_debug.jpg")
        save_debug_image(original_bgr, full_alpha, debug_path)

    status = "ok"
    if match_result["score"] < min_score:
        status = "low_score"

    return {
        "name": stem,
        "status": status,
        "mode": mode,
        "x": match_result["x"],
        "y": match_result["y"],
        "w": match_result["w"],
        "h": match_result["h"],
        "scale": f"{match_result['scale']:.4f}",
        "score": f"{match_result['score']:.6f}",
        "corr_score": f"{match_result.get('corr_score', 0.0):.6f}",
        "edge_score": f"{match_result.get('edge_score', 0.0):.6f}",
        "color_score": f"{match_result.get('color_score', 0.0):.6f}",
        "prior_score": f"{match_result.get('prior_score', 0.0):.6f}",
        "final_score": f"{match_result.get('final_score', match_result['score']):.6f}",
        "original_path": original_path,
        "cutout_path": cutout_path,
        "alpha_path": alpha_path,
    }


# ============================================================
# 参数
# ============================================================

def parse_scales(scale_min, scale_max, scale_step):
    scales = []
    s = scale_min
    while s <= scale_max + 1e-9:
        scales.append(round(s, 4))
        s += scale_step
    return scales


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", type=str, required=True, help="原图目录")
    parser.add_argument("--cutout-dir", type=str, required=True, help="醒图透明 PNG 目录")
    parser.add_argument("--out-dir", type=str, required=True, help="输出根目录")

    parser.add_argument("--image-exts", type=str, nargs="+",
                        default=[".jpg", ".jpeg", ".png", ".bmp"])
    parser.add_argument("--cutout-exts", type=str, nargs="+",
                        default=[".png"])

    parser.add_argument("--scale-min", type=float, default=0.60,
                        help="模板匹配最小缩放比例")
    parser.add_argument("--scale-max", type=float, default=1.40,
                        help="模板匹配最大缩放比例")
    parser.add_argument("--scale-step", type=float, default=0.01,
                        help="缩放搜索步长")

    parser.add_argument("--alpha-thr", type=int, default=10,
                        help="裁剪 cutout 时使用的 alpha 阈值")

    parser.add_argument("--top-k", type=int, default=12,
                        help="每个 scale 保留的候选匹配数量")

    parser.add_argument("--save-debug", action="store_true",
                        help="保存叠加轮廓检查图")
    parser.add_argument("--min-score", type=float, default=0.35,
                        help="低于该匹配分数会标记为 low_score")

    parser.add_argument("--force-match", action="store_true",
                        help="即使 cutout 与原图同尺寸，也强制裁剪 alpha 后模板匹配回原图")

    parser.add_argument("--use-prior", action="store_true",
                        help="使用 cutout 原始 alpha bbox 的位置作为弱先验，适合轻微平移/缩放错误")

    parser.add_argument("--prior-sigma", type=float, default=200.0,
                        help="位置先验的容忍范围，越大越弱")

    parser.add_argument("--w-corr", type=float, default=0.50,
                        help="模板匹配分数权重")
    parser.add_argument("--w-edge", type=float, default=0.25,
                        help="边缘对齐分数权重")
    parser.add_argument("--w-color", type=float, default=0.15,
                        help="颜色一致性分数权重")
    parser.add_argument("--w-prior", type=float, default=0.10,
                        help="位置先验分数权重，仅 use-prior 时生效")

    args = parser.parse_args()

    image_exts = tuple(
        e.lower() if e.startswith(".") else "." + e.lower()
        for e in args.image_exts
    )

    cutout_exts = tuple(
        e.lower() if e.startswith(".") else "." + e.lower()
        for e in args.cutout_exts
    )

    alpha_out_dir = os.path.join(args.out_dir, "alpha")
    debug_out_dir = os.path.join(args.out_dir, "debug")
    csv_path = os.path.join(args.out_dir, "restore_coords.csv")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(alpha_out_dir, exist_ok=True)

    if args.save_debug:
        os.makedirs(debug_out_dir, exist_ok=True)

    scales = parse_scales(
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_step=args.scale_step,
    )

    images = list_images(args.image_dir, image_exts)

    rows = []
    skipped = []

    print(f"共找到原图 {len(images)} 张")
    print(f"scale 搜索范围: {scales[0]} ~ {scales[-1]}, 共 {len(scales)} 个 scale")
    print(f"force_match: {args.force_match}")
    print(f"use_prior: {args.use_prior}")
    print(f"top_k: {args.top_k}")

    for stem, original_path in tqdm(images):
        cutout_path = find_file_by_stem(
            folder=args.cutout_dir,
            stem=stem,
            exts=cutout_exts,
        )

        if cutout_path is None:
            skipped.append((stem, "missing_cutout"))
            rows.append({
                "name": stem,
                "status": "missing_cutout",
                "mode": "",
                "x": "",
                "y": "",
                "w": "",
                "h": "",
                "scale": "",
                "score": "",
                "corr_score": "",
                "edge_score": "",
                "color_score": "",
                "prior_score": "",
                "final_score": "",
                "original_path": original_path,
                "cutout_path": "",
                "alpha_path": "",
                "error": "",
            })
            continue

        try:
            row = process_one(
                stem=stem,
                original_path=original_path,
                cutout_path=cutout_path,
                alpha_out_dir=alpha_out_dir,
                debug_out_dir=debug_out_dir,
                scales=scales,
                save_debug=args.save_debug,
                min_score=args.min_score,
                force_match=args.force_match,
                alpha_thr=args.alpha_thr,
                top_k=args.top_k,
                use_prior=args.use_prior,
                prior_sigma=args.prior_sigma,
                w_corr=args.w_corr,
                w_edge=args.w_edge,
                w_color=args.w_color,
                w_prior=args.w_prior,
            )

            rows.append(row)

        except Exception as e:
            skipped.append((stem, str(e)))
            rows.append({
                "name": stem,
                "status": "error",
                "mode": "",
                "x": "",
                "y": "",
                "w": "",
                "h": "",
                "scale": "",
                "score": "",
                "corr_score": "",
                "edge_score": "",
                "color_score": "",
                "prior_score": "",
                "final_score": "",
                "original_path": original_path,
                "cutout_path": cutout_path,
                "alpha_path": "",
                "error": str(e),
            })

    fieldnames = [
        "name",
        "status",
        "mode",
        "x",
        "y",
        "w",
        "h",
        "scale",
        "score",
        "corr_score",
        "edge_score",
        "color_score",
        "prior_score",
        "final_score",
        "original_path",
        "cutout_path",
        "alpha_path",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            for key in fieldnames:
                row.setdefault(key, "")
            writer.writerow(row)

    print("\n完成。")
    print(f"alpha 输出目录: {alpha_out_dir}")
    print(f"坐标 CSV: {csv_path}")

    if args.save_debug:
        print(f"debug 检查图目录: {debug_out_dir}")

    print(f"跳过/失败数量: {len(skipped)}")

    if skipped:
        print("前 20 个问题样本:")
        for item in skipped[:20]:
            print("  ", item)


if __name__ == "__main__":
    main()