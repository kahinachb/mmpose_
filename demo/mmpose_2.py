
import numpy as np
import cv2
from sort import Sort  # Ensure sort.py is in the same directory
from mmpose.apis import MMPoseInferencer
import os
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import MMPoseInferencer

no_sujet = 1
no_cam = 26587
task = 'marche'

video_path = '/home/kahina/Documents/challenge_markerless/Data/Data/les-deux/danse/videos/' + str(no_cam) + '/' + str(no_cam) + '.avi'
output_txt_path_1 = '/home/kahina/Documents/challenge_markerless/Data/Data/les-deux/danse/videos/' + str(no_cam)+ '_bleu.txt'
output_txt_path_2 = '/home/kahina/Documents/challenge_markerless/Data/Data/les-deux/danse/videos/' + str(no_cam)+ '_vert.txt'

output_vid = f"/home/kahina/Documents/challenge_markerless/Data/Data/les-deux/danse/videos/result_{str(no_cam)}_les_deux.avi"



# Specify the path to model config and checkpoint file
# config_file = '/home/kahina/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
# checkpoint_file = '/home/kahina/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
# det_model = init_detector(config_file, checkpoint_file)

def which_bbox(bbox_current,bbox_previous_list):
    if np.linalg.norm(bbox_current - bbox_previous_list[0]) > np.linalg.norm(bbox_current - bbox_previous_list[1]): 
        return 1
    else : 
        return 0
    

def process_video_save(video_path, inferencer, output_txt_path_1, output_vid_path, output_txt_path_2):
    """Process a video frame by frame, visualize predicted keypoints, and save keypoints to a .txt file."""

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    print("Video size:", size)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_vid_path, fourcc, 30.0, size)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Open the output .txt file for writing keypoints
    with open(output_txt_path_1, 'w') as f1, open(output_txt_path_2, 'w') as f2:
    # with open(output_txt_path, 'w') as f:
        i = 0
        bbox_previous = []
        nums_bbox = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB as inferencer expects RGB input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            result_generator = inferencer(frame_rgb, return_vis=True, kpt_thr = 0.7, bbox_thr = 0.7,   show=False)
            result = next(result_generator)
            
            # Extract the visualization result
            vis_frame = result['visualization'][0]

            # Convert visualization result from RGB back to BGR for OpenCV
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

            # Retrieve keypoints and bounding boxes
            predictions = result['predictions']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

            null_combined_str = f"{i},{0},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
            combined_str_list = [null_combined_str, null_combined_str]
            # print(combined_str_list)

            for person_idx, prediction in enumerate(predictions):
                num_bbox = 0
                # Each prediction is a list of dictionaries; access each person's data
                for idx, person in enumerate(prediction):
                    # print(person)
                    keypoints = np.asarray(person['keypoints'])
                    bbox = person['bbox'][0]  # This is already a list, no need to access [0]
                    # print("bbox", bbox) []
                    bbox_score = person['bbox_score']

                    if keypoints is not None:
                        keypoints_2d = keypoints[:, :2]  # We need only the x, y coordinates

                        # Draw bounding box with different colors

                        (x, y, w, h) = map(int, bbox)  # Ensure coordinates are integers
                        if i == 0:
                            bbox_previous.append(np.array([x,y]).reshape(2,1))
                            nums_bbox = [0, 1]
                            
                        
                        else:
                            num_bbox = which_bbox(np.array([x,y]).reshape(2,1),bbox_previous)
                            # print(num_bbox)
                            nums_bbox[num_bbox] = num_bbox

                            bbox_previous[num_bbox] =  np.array([x,y]).reshape(2,1)

                        color = colors[num_bbox]
                        cv2.rectangle(vis_frame_bgr, (x, y), (x+w, y+h), color, 2)

                        # Draw keypoints with different colors
                        for (kx, ky) in keypoints_2d:
                            cv2.circle(vis_frame_bgr, (int(kx), int(ky)), 3, color, -1)

                        # Flatten the keypoints array and convert to string with comma separator
                        keypoints_str = ','.join(map(str, keypoints_2d.flatten()))
                        #bbox_str = ','.join(map(str, bbox))
                        # print(bbox_str)
                        score_str = str(bbox_score)
                        
                        # Combine keypoints and scores into one string
                        combined_str = f"{i},{score_str},{keypoints_str}"

                        combined_str_list[num_bbox] = combined_str
                        # Write the keypoints to the corresponding .txt file
            # print("i",i)
            # print(nums_bbox)
            # print(combined_str_list)
            if nums_bbox[0] == 0 and nums_bbox[1] == 1:
                f1.write(f"{combined_str_list[0]}\n")
                f2.write(f"{combined_str_list[1]}\n")


            elif nums_bbox[1] == 0 and nums_bbox[0] == 1:
                f1.write(f"{combined_str_list[1]}\n")
                f2.write(f"{combined_str_list[0]}\n")


            # if idx == 0:
            #     f1.write(f"{combined_str}\n")
            # elif idx == 1:
            #     f2.write(f"{combined_str}\n")   
                
            # Write the frame with inferred keypoints and bounding box to output video
            out.write(vis_frame_bgr)

            # Display the frame with inferred keypoints and bounding box
            cv2.imshow('Video with Keypoints and Bounding Box', vis_frame_bgr)
            
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1

    # Release the video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()



inferencer = MMPoseInferencer('rtmo')  # or use the appropriate model alias

process_video_save(video_path, inferencer, output_txt_path_1, output_vid, output_txt_path_2)

