import os
from PIL import Image

# 路径配置
img_src_dir = r"D:\error\大三下2026春季学期\认知工程\data\hardcase\xingtu"

TARGET_W, TARGET_H = 1024, 1024
img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def batch_force_resize(folder_path):
    """
    与 AIM 处理脚本保持一致的策略：
    - 灰度图 (L/1模式) 用 NEAREST
    - 彩色图用 LANCZOS
    强制 resize 到 512x512，允许长宽比改变。
    原地覆盖保存。
    """
    if not os.path.isdir(folder_path):
        print(f"目录不存在：{folder_path}")
        return

    cnt = 0
    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in img_exts:
            continue
        full_path = os.path.join(folder_path, fname)
        try:
            with Image.open(full_path) as im:
                if im.mode in ("L", "1"):
                    im_out = im.resize((TARGET_W, TARGET_H), Image.Resampling.NEAREST)
                else:
                    im_out = im.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
                im_out.save(full_path)
            cnt += 1
        except Exception as e:
            print(f"处理失败 {fname}: {str(e)}")

    print(f"resize完成：{folder_path}，共处理 {cnt} 张")


if __name__ == "__main__":
    batch_force_resize(img_src_dir)
    print("\n全部任务完成")