import cv2
from mmpose.apis import MMPoseInferencer
import numpy as np

no_sujet = 1
no_cam = 2
task = 'marche'

video_path = './vids_challenge/' + task + '_Miqus_' + str(no_cam) + '_sujet' + str(no_sujet) + '.avi'
out_file = './vids_challenge/result_' + task + '_' + str(no_cam) + '_sujet' + str(no_sujet) + '.txt'


def process_video(video_path, inferencer):
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
        
        # Convert frame to RGB as inferencer expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        result_generator = inferencer(frame_rgb, return_vis=True, show=False)
        result = next(result_generator)

        # Extract the visualization result
        vis_frame = result['visualization'][0]

        # Convert visualization result from RGB back to BGR for OpenCV
        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

        # Display the frame with inferred keypoints
        cv2.imshow('Video with Keypoints', vis_frame_bgr)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def process_video_save(video_path, inferencer, output_txt_path):
    """Process a video frame by frame, visualize predicted keypoints, and save keypoints to a .txt file."""

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Open the output .txt file for writing keypoints
    with open(output_txt_path, 'w') as f:
        i=0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB as inferencer expects RGB input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            result_generator = inferencer(frame_rgb, return_vis=True, show=False)
            result = next(result_generator)

            # Extract the visualization result
            vis_frame = result['visualization'][0]

            # Convert visualization result from RGB back to BGR for OpenCV
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

            # Display the frame with inferred keypoints
            cv2.imshow('Video with Keypoints', vis_frame_bgr)
            
            # Extract keypoints
            predictions = result['predictions']
            for prediction in predictions:
                # print(prediction[0]['keypoints'])
                keypoints = np.asarray(prediction[0]['keypoints'])
                print(np.shape(keypoints))
                if keypoints is not None:
                    keypoints_2d = keypoints[:, :2]  # We need only the x, y coordinates

                    # Flatten the keypoints array and convert to string with comma separator
                    keypoints_str = ','.join(map(str, keypoints_2d.flatten()))
                    keypoints_str = str(i) + "," + keypoints_str
                    
                    # Write the keypoints to the .txt file
                    f.write(f"{keypoints_str}\n")
            
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
# Example usage
inferencer = MMPoseInferencer('body26')  # or use appropriate model alias

process_video_save(video_path, inferencer, out_file)