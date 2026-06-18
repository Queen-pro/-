"""
Batch inference for ViTMatte model.

Run:
python run_multi_image.py \
    --model vitmatte-s \
    --checkpoint-dir ./checkpoints/ViTMatte_S_Com.pth \
    --image-dir ./data/images \
    --trimap-dir ./data/trimaps \
    --output-dir ./data/output \
    --device cuda
"""

import os
import glob
import time
from PIL import Image
from os.path import join as opj
from torchvision.transforms import functional as F
from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

def init_model(model_name, checkpoint, device):
    assert model_name in ['vitmatte-s', 'vitmatte-b'], \
        f"model 必须是 vitmatte-s 或 vitmatte-b，当前: {model_name}"

    config = 'configs/common/model.py'
    cfg = LazyConfig.load(config)

    if model_name == 'vitmatte-b':
        cfg.model.backbone.embed_dim = 768
        cfg.model.backbone.num_heads = 12
        cfg.model.decoder.in_chans   = 768

    model = instantiate(cfg.model)
    model.to(device)
    model.eval()
    DetectionCheckpointer(model).load(checkpoint)
    return model

def get_data(image_path, trimap_path):
    image  = Image.open(image_path).convert('RGB')
    image  = F.to_tensor(image).unsqueeze(0)
    trimap = Image.open(trimap_path).convert('L')
    trimap = F.to_tensor(trimap).unsqueeze(0)
    return {'image': image, 'trimap': trimap}

def infer_one_image(model, input_data, save_path):
    output = model(input_data)['phas'].flatten(0, 2)
    output = F.to_pil_image(output)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output.save(save_path)

def get_image_paths(image_dir):
    """获取目录下所有图片，按文件名排序"""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(opj(image_dir, ext)))
    return sorted(paths)


def find_trimap(trimap_dir, image_basename):
    """
    根据原图文件名（无扩展名）匹配 trimap。
    支持两种命名规则：
      - 与原图同名:        1.png
      - 加 _trimap 后缀:   1_trimap.png
    """
    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        # 规则1：同名
        candidate = opj(trimap_dir, image_basename + ext)
        if os.path.exists(candidate):
            return candidate
        # 规则2：_trimap 后缀
        candidate = opj(trimap_dir, image_basename + '_trimap' + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def run_multi_image(model, image_dir, trimap_dir, output_dir, device):
    image_paths = get_image_paths(image_dir)
    if not image_paths:
        print(f"[警告] 在 {image_dir} 下未找到任何图片！")
        return

    os.makedirs(output_dir, exist_ok=True)

    total     = len(image_paths)
    success   = 0
    fail_list = []

    print(f"\n共找到 {total} 张图片，开始批量推理...\n")
    t_start = time.time()

    for idx, image_path in enumerate(image_paths, 1):
        basename = os.path.splitext(os.path.basename(image_path))[0]

        # 匹配 trimap
        trimap_path = find_trimap(trimap_dir, basename)
        if trimap_path is None:
            print(f"[{idx}/{total}] ✗ 跳过 {basename}（找不到对应 trimap）")
            fail_list.append(basename)
            continue

        # 推理
        try:
            input_data  = get_data(image_path, trimap_path)

            # 将 tensor 移到对应 device
            input_data = {k: v.to(device) for k, v in input_data.items()}

            save_path   = opj(output_dir, basename + '_alpha.png')
            infer_one_image(model, input_data, save_path)

            elapsed = time.time() - t_start
            avg     = elapsed / idx
            eta     = avg * (total - idx)
            print(f"[{idx}/{total}] ✓ {basename}  "
                  f"| 已用 {elapsed:.1f}s  ETA {eta:.1f}s")
            success += 1

        except Exception as e:
            print(f"[{idx}/{total}] ✗ {basename} 推理出错: {e}")
            fail_list.append(basename)

    # 汇总
    print(f"\n{'='*50}")
    print(f"完成！成功 {success}/{total} 张，失败 {len(fail_list)} 张")
    print(f"总耗时: {time.time() - t_start:.1f}s")
    if fail_list:
        print(f"失败文件: {fail_list}")
    print(f"结果保存在: {output_dir}")


# ------------------------------------------------------------------ #
#  主入口
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--model',          type=str, default='vitmatte-s')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/ViTMatte_S_DIS.pth')
    parser.add_argument('--image-dir',      type=str, default='/home/liangyin/zbk/rzkeshe/hardcase/image')
    parser.add_argument('--trimap-dir',     type=str, default='/home/liangyin/zbk/rzkeshe/hardcase/trimap')
    # parser.add_argument('--image-dir',      type=str, default='/home/liangyin/zbk/rzkeshe/for_plant/image')
    # parser.add_argument('--trimap-dir',     type=str, default='/home/liangyin/zbk/rzkeshe/for_plant/trimap')
    parser.add_argument('--output-dir',     type=str, default='./data/result_hard/output_S_DIS')
    parser.add_argument('--device',         type=str, default='cuda')
    args = parser.parse_args()

    print('Initializing model... Please wait...')
    model = init_model(args.model, args.checkpoint_dir, args.device)
    print('Model initialized.\n')

    run_multi_image(
        model      = model,
        image_dir  = args.image_dir,
        trimap_dir = args.trimap_dir,
        output_dir = args.output_dir,
        device     = args.device,
    )