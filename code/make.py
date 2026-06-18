import os
import shutil
from PIL import Image

# 路径配置
img_src_dir = r"D:\error\大三下2026春季学期\认知工程\data\for_plant\image"
mask_src_dir = r"D:\error\大三下2026春季学期\认知工程\data\AIM-500\mask"
trimap_src_dir = r"D:\error\大三下2026春季学期\认知工程\data\AIM-500\trimap"
alpha_dst_dir = r"D:\error\大三下2026春季学期\认知工程\data\for_plant\alpha"
trimap_dst_dir = r"D:\error\大三下2026春季学期\认知工程\data\for_plant\trimap"

TARGET_W, TARGET_H = 1024, 1024
img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"创建目录：{folder_path}")

def copy_match_files():
    create_folder(alpha_dst_dir)
    create_folder(trimap_dst_dir)
    if not os.path.isdir(img_src_dir):
        print(f"源image目录不存在：{img_src_dir}")
        return False

    # 预先把mask、trimap目录构建
    mask_name_map = {}
    for f in os.listdir(mask_src_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in img_exts:
            mask_name_map[stem] = os.path.join(mask_src_dir, f)

    trimap_name_map = {}
    for f in os.listdir(trimap_src_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in img_exts:
            trimap_name_map[stem] = os.path.join(trimap_src_dir, f)

    cnt = 0
    for fname in os.listdir(img_src_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in img_exts:
            continue
        stem, _ = os.path.splitext(fname)

        # 按主干名查找，不再比对后缀
        if stem in mask_name_map:
            shutil.copy2(mask_name_map[stem], alpha_dst_dir)
            cnt += 1
        else:
            print(f"缺失mask主干：{stem}")

        if stem in trimap_name_map:
            shutil.copy2(trimap_name_map[stem], trimap_dst_dir)
        else:
            print(f"缺失trimap主干：{stem}")

    print(f"成功复制匹配文件 {cnt} 组\n")
    return True

# 第二步：统一resize
def batch_force_resize(folder_path):
    if not os.path.isdir(folder_path):
        print(f"跳过不存在目录：{folder_path}")
        return
    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in img_exts:
            continue
        full_path = os.path.join(folder_path, fname)
        try:
            with Image.open(full_path) as im:
                # 灰度掩码、trimap用最近邻；彩色图用LANCZOS
                if im.mode in ("L", "1"):
                    im_out = im.resize((TARGET_W, TARGET_H), Image.Resampling.NEAREST)
                else:
                    im_out = im.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
                im_out.save(full_path)
        except Exception as e:
            print(f"处理失败 {fname}: {str(e)}")
    print(f"目录强制resize完成：{folder_path}")

if __name__ == "__main__":
    copy_ok = copy_match_files()
    if not copy_ok:
        exit(1)
    # 三个目录统一执行resize策略
    process_dirs = [img_src_dir, alpha_dst_dir, trimap_dst_dir]
    for d in process_dirs:
        batch_force_resize(d)
    print("\n全部任务完成")