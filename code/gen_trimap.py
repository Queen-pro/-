import os
import cv2
import numpy as np
import argparse


def alpha2trimap(pha, fg_width=10, bg_width=10, fg_thresh=0.95, bg_thresh=0.05):
    erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]
    alpha = pha

    fg_mask = (alpha >= fg_thresh).astype(np.uint8)
    bg_mask = (alpha <= bg_thresh).astype(np.uint8)

    if fg_width > 0:
        fg_mask = cv2.erode(fg_mask, erosion_kernels[fg_width])
    if bg_width > 0:
        bg_mask = cv2.erode(bg_mask, erosion_kernels[bg_width])

    trimap = np.ones_like(alpha) * 128
    trimap[fg_mask == 1] = 255
    trimap[bg_mask == 1] = 0

    return trimap


def parse_args():
    parser = argparse.ArgumentParser(description='Convert alpha matte to trimap (inference)')
    # parser.add_argument('--alpha_path', type=str, default="/home/liangyin/zbk/rzkeshe/ViTMatte/data/keshe/alpha",
    #                   help='Path to the alpha matte directory')
    # parser.add_argument('--trimap_path', type=str, default="/home/liangyin/zbk/rzkeshe/ViTMatte/data/keshe/trimap",
    #                   help='Path to save the generated trimaps')
    # parser.add_argument('--alpha_path', type=str, default="/home/liangyin/zbk/rzkeshe/for_plant/alpha",
    #                   help='Path to the alpha matte directory')
    # parser.add_argument('--trimap_path', type=str, default="/home/liangyin/zbk/rzkeshe/trimap_new_for_eval",
    #                   help='Path to save the generated trimaps')
    parser.add_argument('--alpha_path', type=str, default="/home/liangyin/zbk/rzkeshe/hardcase/alpha",
                      help='Path to the alpha matte directory')
    parser.add_argument('--trimap_path', type=str, default="/home/liangyin/zbk/rzkeshe/hardcase/trimap",
                      help='Path to save the generated trimaps')
    parser.add_argument('--fg_width', type=int, default=15,#4foreval,15forinference
                      help='Erosion kernel size for foreground')
    parser.add_argument('--bg_width', type=int, default=15,
                      help='Erosion kernel size for background')
    parser.add_argument('--fg_thresh', type=float, default=0.95,
                      help='Alpha threshold above which pixel is considered foreground')
    parser.add_argument('--bg_thresh', type=float, default=0.05,
                      help='Alpha threshold below which pixel is considered background')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.trimap_path, exist_ok=True)

    for image_name in os.listdir(args.alpha_path):
        print('Processing ' + image_name)
        image_path = os.path.join(args.alpha_path, image_name)
        img = cv2.imread(image_path, 0) / 255.
        trimap = alpha2trimap(img, fg_width=args.fg_width, bg_width=args.bg_width,
                               fg_thresh=args.fg_thresh, bg_thresh=args.bg_thresh)
        trimap_path = os.path.join(args.trimap_path, image_name)
        cv2.imwrite(trimap_path, trimap)