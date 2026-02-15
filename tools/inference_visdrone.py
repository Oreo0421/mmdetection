#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VisDrone Faster R-CNN 推理脚本
支持单张图片或整个文件夹输入，输出带检测框的可视化图片

Usage:
    # 单张图片
    python tools/inference_visdrone.py path/to/image.jpg \
        configs/visdrone/faster-rcnn_r50_fpn_1x_visdrone.py \
        work_dirs/visdrone_faster_rcnn/epoch_100.pth \
        --out-dir results/

    # 整个文件夹
    python tools/inference_visdrone.py /path/to/images/ \
        configs/visdrone/faster-rcnn_r50_fpn_1x_visdrone.py \
        work_dirs/visdrone_faster_rcnn/epoch_100.pth \
        --out-dir results/
"""

import argparse
from pathlib import Path

import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer

# VisDrone 10 类
VISDRONE_CLASSES = (
    'car', 'people', 'van', 'truck', 'motor', 'bicycle',
    'tricycle', 'awning-tricycle', 'pedestrian', 'bus',
)
VISDRONE_PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
]
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def parse_args():
    parser = argparse.ArgumentParser(description='VisDrone Faster R-CNN 推理')
    parser.add_argument('input', type=str, help='输入图片路径或文件夹路径')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('checkpoint', type=str, help='checkpoint 权重路径')
    parser.add_argument('--out-dir', type=str, default='results/visdrone_inference',
                        help='输出目录')
    parser.add_argument('--score-thr', type=float, default=0.02,
                        help='检测置信度阈值')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def get_image_list(input_path):
    """获取图片列表：单张或文件夹内所有图片"""
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() in IMG_EXTENSIONS:
            return [str(input_path)]
        else:
            raise ValueError(f'不支持的文件格式: {input_path.suffix}')
    elif input_path.is_dir():
        images = []
        for ext in IMG_EXTENSIONS:
            images.extend(input_path.glob(f'*{ext}'))
        images = sorted([str(p) for p in images])
        if not images:
            raise ValueError(f'文件夹内未找到图片: {input_path}')
        return images
    else:
        raise ValueError(f'路径不存在: {input_path}')


def main():
    args = parse_args()

    # 加载模型
    print_log(f'加载配置: {args.config}')
    print_log(f'加载权重: {args.checkpoint}')
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # 确保使用 VisDrone 类别（checkpoint 可能保存了 COCO 类别）
    model.dataset_meta = {
        'classes': list(VISDRONE_CLASSES),
        'palette': VISDRONE_PALETTE,
    }

    # 获取输入图片列表
    image_list = get_image_list(args.input)
    print_log(f'共 {len(image_list)} 张图片待推理')

    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 可视化器
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta

    # 逐张推理
    for i, img_path in enumerate(image_list):
        print_log(f'[{i + 1}/{len(image_list)}] 推理: {img_path}')
        result = inference_detector(model, img_path)

        # 加载图片 (BGR -> RGB)
        img_bytes = mmengine.fileio.get(img_path)
        img = mmcv.imfrombytes(img_bytes)
        img = img[:, :, ::-1]

        # 绘制并保存
        img_name = Path(img_path).stem
        out_file = out_dir / f'{img_name}.jpg'
        visualizer.add_datasample(
            name='result',
            image=img,
            data_sample=result,
            draw_gt=False,
            draw_pred=True,
            show=False,
            wait_time=0,
            out_file=str(out_file),
            pred_score_thr=args.score_thr,
        )

    print_log(f'推理完成，结果已保存到: {out_dir}')


if __name__ == '__main__':
    main()
