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

det_config = 'projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
pose_config = 'projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/1/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth'


device = 'cuda:0'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=False)))

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

def process_webcam(detector, pose_estimator, visualizer, show_interval=1):
    """Process webcam frames and visualize predicted keypoints."""
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set desired width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)  # Set desired heigh
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    saving = False
    pred_instances_list = []
    frame_idx = 0
    output_root = './output'
    mmengine.mkdir_or_exist(output_root)
    pred_save_path = f'{output_root}/results_webcam.json'
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
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
            draw_heatmap=False,
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
        
        # Press 'q' to exit the loop, 's' to start/stop saving
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            saving = not saving
            if saving:
                print("Started saving keypoints.")
            else:
                print("Stopped saving keypoints.")
        
    
    cap.release()
    cv2.destroyAllWindows()
    
    

process_webcam(
    detector,
    pose_estimator,
    visualizer,
    show_interval=1
)
