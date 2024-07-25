import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from IPython.display import Image, display
import tempfile
import os.path as osp
import cv2


try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

local_runtime = True


video_path = './marche_Miqus_5_26587.avi'

img = '/home/kahina/mmpose/tests/data/coco/000000196141.jpg'
# pose_config = '/home/kahina/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
pose_config = '/home/kahina/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
det_config = '/home/kahina/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

device = 'cuda:0'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))


# build detector
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)


# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# init visualizer
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

def process_video(video_path, detector, pose_estimator, visualizer, show_interval=1):
    """Process a video frame by frame and visualize predicted keypoints."""
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Predict bbox
        scope = detector.cfg.get('default_scope', 'mmdet')
        if scope is not None:
            init_default_scope(scope)
        detect_result = inference_detector(detector, frame_rgb)
        pred_instance = detect_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                       pred_instance.scores > 0.3)]
        bboxes = bboxes[nms(bboxes, 0.3)][:, :4]
        
        # Predict keypoints
        pose_results = inference_topdown(pose_estimator, frame_rgb, bboxes)
        data_samples = merge_data_samples(pose_results)
        
        # Show the results
        visualizer.add_datasample(
            'result',
            frame_rgb,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=True,
            draw_bbox=True,
            show=False,
            wait_time=show_interval,
            out_file=None,
            kpt_thr=0.3)
        
        # Retrieve the visualized image
        vis_result = visualizer.get_image()
        
        # Convert image from RGB to BGR for OpenCV
        vis_result_bgr = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
        
        # Display the frame using OpenCV
        cv2.imshow('Visualization Result', vis_result_bgr)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()



process_video(
    video_path,
    detector,
    pose_estimator,
    visualizer,
    show_interval=1
)
