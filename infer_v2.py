"""Inference for ResNet50-v2 model."""
import os, time, torch
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv

DATA_ROOT = '/mnt/data_hdd/fzhi/track_data/VisDrone/image_detection/data/images/'
OUT_DIR = '/home/fzhi/fzt/intern_pipline/detect/mmdetection/work_dirs/infer_results/resnet50_v2'
os.makedirs(OUT_DIR, exist_ok=True)

config = 'work_dirs/visdrone_faster_rcnn_v2/faster-rcnn_r50_fpn_1x_visdrone_v2.py'
ckpt = 'work_dirs/visdrone_faster_rcnn_v2/best_coco_bbox_mAP_epoch_30.pth'

test_images = [
    '0000001_02999_d_0000005.jpg',
    '0000006_00159_d_0000001.jpg',
    '0000048_01989_d_0000098.jpg',
    '0000344_01569_d_0000301.jpg',
    '9999945_00000_d_0000080.jpg',
    '9999994_00000_d_0000054.jpg',
]

model = init_detector(config, ckpt, device='cuda:0')
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# Warmup
_ = inference_detector(model, os.path.join(DATA_ROOT, test_images[0]))

total_time = 0
for img_name in test_images:
    img_path = os.path.join(DATA_ROOT, img_name)
    t0 = time.time()
    result = inference_detector(model, img_path)
    t1 = time.time()
    infer_time = (t1 - t0) * 1000
    total_time += infer_time

    img = mmcv.imread(img_path, channel_order='rgb')
    visualizer.add_datasample(
        name=img_name, image=img, data_sample=result,
        draw_gt=False, pred_score_thr=0.3, show=False)
    vis_img = visualizer.get_image()
    mmcv.imwrite(mmcv.rgb2bgr(vis_img), os.path.join(OUT_DIR, img_name))

    n_dets = (result.pred_instances.scores > 0.3).sum().item()
    print(f'  {img_name}: {n_dets} dets, {infer_time:.1f}ms')

print(f'\n  Average: {total_time/len(test_images):.1f}ms')
