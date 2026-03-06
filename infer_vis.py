"""Inference and visualization for both Faster R-CNN models on VisDrone."""
import os
import time
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv

# Paths
DATA_ROOT = '/mnt/data_hdd/fzhi/track_data/VisDrone/image_detection/data/images/'
OUT_DIR = '/home/fzhi/fzt/intern_pipline/detect/mmdetection/work_dirs/infer_results'
os.makedirs(OUT_DIR, exist_ok=True)

# Models config
models = {
    'resnet50': {
        'config': 'work_dirs/visdrone_faster_rcnn/faster-rcnn_r50_fpn_1x_visdrone.py',
        'ckpt': 'work_dirs/visdrone_faster_rcnn/best_coco_bbox_mAP_epoch_85.pth',
    },
    'mobilenetv2': {
        'config': 'work_dirs/visdrone_faster_rcnn_mobilenetv2/faster-rcnn_mobilenetv2_fpn_visdrone.py',
        'ckpt': 'work_dirs/visdrone_faster_rcnn_mobilenetv2/best_coco_bbox_mAP_epoch_30.pth',
    },
}

# Select diverse images for visualization
test_images = [
    '0000001_02999_d_0000005.jpg',
    '0000006_00159_d_0000001.jpg',
    '0000048_01989_d_0000098.jpg',
    '0000344_01569_d_0000301.jpg',
    '9999945_00000_d_0000080.jpg',
    '9999994_00000_d_0000054.jpg',
]

for model_name, model_info in models.items():
    print(f'\n{"="*60}')
    print(f'Loading model: {model_name}')
    print(f'{"="*60}')

    model = init_detector(model_info['config'], model_info['ckpt'], device='cuda:0')
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    model_out_dir = os.path.join(OUT_DIR, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    # Warmup
    dummy_img = os.path.join(DATA_ROOT, test_images[0])
    _ = inference_detector(model, dummy_img)

    total_time = 0
    for img_name in test_images:
        img_path = os.path.join(DATA_ROOT, img_name)
        if not os.path.exists(img_path):
            print(f'  Skip {img_name} (not found)')
            continue

        # Inference with timing
        t0 = time.time()
        result = inference_detector(model, img_path)
        t1 = time.time()
        infer_time = (t1 - t0) * 1000
        total_time += infer_time

        # Visualize
        img = mmcv.imread(img_path, channel_order='rgb')
        visualizer.add_datasample(
            name=img_name,
            image=img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=0.3,
            show=False,
        )
        vis_img = visualizer.get_image()

        out_path = os.path.join(model_out_dir, img_name)
        mmcv.imwrite(mmcv.rgb2bgr(vis_img), out_path)

        # Count detections
        n_dets = (result.pred_instances.scores > 0.3).sum().item()
        print(f'  {img_name}: {n_dets} dets, {infer_time:.1f}ms')

    avg_time = total_time / len(test_images)
    print(f'\n  Average inference time: {avg_time:.1f}ms')

    del model
    torch.cuda.empty_cache()

print(f'\nResults saved to {OUT_DIR}')
