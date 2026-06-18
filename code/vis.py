import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 模型文件分组配置
# =========================

GROUP_CONFIG = {
    "ViTMatte": {
        "files": {
            "B-DIS": "ViTMatte_B_DIS",
            "B-Com": "ViTMatte_B_Com",
            "S-DIS": "ViTMatte_S_DIS",
            "S-Com": "ViTMatte_S_Com",
        }
    },
    "ZIM": {
        "files": {
            "B-box": "ZIM_B_box",
            "B-point": "ZIM_B_point",
            "L-box": "ZIM_L_box",
            "L-point": "ZIM_L_point",
        }
    },
    "Matting Anything": {
        "files": {
            "Box": "MAM_box",
            "Point": "MAM_point",
            "Text": "MAM_text",
        }
    },
}


METRICS = ["SAD", "MSE", "MAD"]

def find_metric_file(data_dir, stem):
    """
    根据文件名 stem 查找csv文件
    """
    candidates = []

    for ext in [".xlsx", ".xls", ".csv"]:
        path = os.path.join(data_dir, stem + ext)
        if os.path.exists(path):
            candidates.append(path)

    if len(candidates) == 0:
        # 宽松匹配
        for path in glob.glob(os.path.join(data_dir, "*")):
            name = os.path.splitext(os.path.basename(path))[0]
            ext = os.path.splitext(path)[1].lower()
            if stem.lower() == name.lower() and ext in [".xlsx", ".xls", ".csv"]:
                candidates.append(path)

    if len(candidates) == 0:
        raise FileNotFoundError(f"找不到指标文件: {stem}.xlsx / .xls / .csv")

    return candidates[0]


def read_metric_file(path):
    """
    读取单个指标文件
    要求至少包含列：
        image, SAD, MSE, MAD, Grad, Conn
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"不支持的文件类型: {path}")

    # 清理列名
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["image", "SAD", "MSE", "MAD"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{path} 缺少必要列: {col}")

    # 去掉空行
    df = df.dropna(subset=["image"]).copy()

    # 统一 image 名称为字符串
    df["image"] = df["image"].astype(str)

    # 指标转数值
    for metric in ["SAD", "MSE", "MAD", "Grad", "Conn"]:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df


def load_all_results(data_dir):
    """
    返回：
        results[group_name][method_name] = dataframe
    """
    results = {}

    for group_name, group_info in GROUP_CONFIG.items():
        results[group_name] = {}

        for method_name, file_stem in group_info["files"].items():
            path = find_metric_file(data_dir, file_stem)
            df = read_metric_file(path)
            results[group_name][method_name] = df

            print(f"[OK] {group_name} / {method_name}: {path}, n={len(df)}")

    return results


# =========================
# 3. Top-20 样本选择
# =========================

def get_top_images_for_group_metric(group_dfs, metric, top_k=20):
    """
    对某一模型组和某一指标，计算该组所有方法在每张图上的平均误差，
    选出平均误差最高的 Top-K 样本

    这样每个子图内部所有折线使用相同的 x 轴样本
    """
    merged = None

    for method_name, df in group_dfs.items():
        sub = df[["image", metric]].copy()
        sub = sub.rename(columns={metric: method_name})

        if merged is None:
            merged = sub
        else:
            merged = pd.merge(merged, sub, on="image", how="outer")

    method_cols = [c for c in merged.columns if c != "image"]
    merged["mean_metric"] = merged[method_cols].mean(axis=1, skipna=True)

    top_df = merged.sort_values("mean_metric", ascending=False).head(top_k)

    return top_df["image"].tolist()


def extract_values(df, images, metric):
    """
    按照指定 image 顺序提取某个指标值
    """
    temp = df.set_index("image")
    values = []

    for img in images:
        if img in temp.index:
            values.append(float(temp.loc[img, metric]))
        else:
            values.append(np.nan)

    return values

def setup_paper_style():
    """
    论文风格绘图参数。
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 9

    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["lines.linewidth"] = 1.6
    plt.rcParams["lines.markersize"] = 3.5

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def plot_top20_3x3(
    results,
    out_path,
    top_k=20,
    dpi=300,
):
    setup_paper_style()

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(14.5, 9.2),
        constrained_layout=False,
    )

    group_names = ["ViTMatte", "ZIM", "Matting Anything"]

    markers = ["o", "s", "^", "D", "v", "P"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    for row_idx, metric in enumerate(METRICS):
        for col_idx, group_name in enumerate(group_names):
            ax = axes[row_idx, col_idx]
            group_dfs = results[group_name]

            top_images = get_top_images_for_group_metric(
                group_dfs=group_dfs,
                metric=metric,
                top_k=top_k,
            )

            x = np.arange(len(top_images))

            for i, (method_name, df) in enumerate(group_dfs.items()):
                y = extract_values(df, top_images, metric)

                ax.plot(
                    x,
                    y,
                    marker=markers[i % len(markers)],
                    linestyle=linestyles[i % len(linestyles)],
                    label=method_name,
                    alpha=0.95,
                )

            # 第一行显示模型组名
            if row_idx == 0:
                ax.set_title(group_name, fontweight="bold")

            # 每行显示指标名
            if col_idx == 0:
                ax.set_ylabel(metric)

            # 最后一行显示样本编号
            if row_idx == 2:
                ax.set_xlabel("Top-20 samples")
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [os.path.splitext(img)[0] for img in top_images],
                    rotation=60,
                    ha="right",
                )
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.5,
                alpha=0.45,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # 仅第一行显示图例；第二、三行默认与第一行相同，不重复显示
            if row_idx == 0:
                ax.legend(
                    frameon=False,
                    loc="upper right",
                    handlelength=2.0,
                    borderaxespad=0.2,
                    ncol=1,
                )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    # 保存 PDF，方便插图
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"\n[Saved] {out_path}")
    print(f"[Saved] {pdf_path}")

    plt.close(fig)


# =========================
# 5. 主函数
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="存放各模型 Excel/CSV 指标文件的文件夹",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="./figures_metric",
        help="输出图片文件夹",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="每个子图展示误差最高的 Top-K 样本",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=1200,
        help="输出 PNG 分辨率",
    )

    args = parser.parse_args()

    results = load_all_results(args.data_dir)

    out_path = os.path.join(
        args.out_dir,
        f"top{args.top_k}_metrics_3x3.png",
    )

    plot_top20_3x3(
        results=results,
        out_path=out_path,
        top_k=args.top_k,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()