import cv2
from mmpose.apis import MMPoseInferencer
import numpy as np

no_sujet = 2
no_cam = 26579
task = 'exotique'

vi = 'Test_6.MOV'

# video_path = "/home/kahina/Downloads/wetransfer_test_mmpose_2024-06-11_1149/Test_mmpose/" + vi 
# out_file ="/home/kahina/Downloads/wetransfer_test_mmpose_2024-06-11_1149/Test_mmpose/" + vi + 'res.txt'
# file = "/home/kahina/Downloads/wetransfer_test_mmpose_2024-06-11_1149/Test_mmpose/" + vi + 'res.avi'

# video_path = '/home/kahina/Documents/challenge_markerless/Data/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/' + str(no_cam) + '/' + str(no_cam) + '.avi'
# out_file = '/home/kahina/Documents/challenge_markerless/Data/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/result_' + task + '_' + str(no_cam) + '_sujet' + str(no_sujet) + '.txt'
# file = f"/home/kahina/Documents/challenge_markerless/Data/Data/sujet_0{str(no_sujet)}/{task}/videos/result_{task }_{str(no_cam)}_sujet{str(no_sujet)}_video_res.avi"

video_path = '/home/kahina/Documents/challenge_markerless/Data/Data/les-deux/danse/videos/' + str(no_cam) + '/' + str(no_cam) + '.avi'
out_file = '/home/kahina/Documents/challenge_markerless/Data/Data/les-deux/danse/videos/' + str(no_cam) + '/' + str(no_cam) + '.txt'
# out_file2 = '/home/kahina/Documents/challenge_markerless/Data/Data/les-deux/danse/out2.txt'

file = f"/home/kahina/Documents/challenge_markerless/Data/Data/les-deux/danse/videos/{str(no_cam)}/result_{str(no_cam)}_les_deux.avi"



def process_video_save(video_path, inferencer, output_txt_path):
    """Process a video frame by frame, visualize predicted keypoints, and save keypoints to a .txt file."""

    # Initialize video capture

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    
    size = (frame_width, frame_height)
    print("size")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file, fourcc, 30.0, size)


    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Open the output .txt file for writing keypoints
    with open(output_txt_path, 'w') as f:
        i=0
        while True:
            ret, frame = cap.read()
            # print("ret", ret)
            if not ret:
                
                break
            
            # Convert frame to RGB as inferencer expects RGB input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            result_generator = inferencer(frame_rgb, return_vis=True,draw_bbox = True, show=False, bbox_thr=0.5)
            result = next(result_generator)
            # print("kkkkkkkkkkk")
            # print(predictions[0][2])
            
            # Extract the visualization result
            vis_frame = result['visualization'][0]


            # Convert visualization result from RGB back to BGR for OpenCV
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

            out.write(vis_frame_bgr)

            # Display the frame with inferred keypoints
            cv2.imshow('Video with Keypoints', vis_frame_bgr)
            
            # Extract keypoints
            predictions = result['predictions']
            for prediction in predictions:
                # print("prediction", prediction)
                
                # bbox1 = np.array(prediction[0]['bbox'])
                # bbox2 = np.array(prediction[1]['bbox'])

                keypoints1 = np.asarray(prediction[0]['keypoints'])
                # print(keypoints1)
                
                # keypoints2 = np.asarray(prediction[1]['keypoints'])

                # print(np.shape(keypoints1))
                if keypoints1 is not None:
                    # print(keypoints)
                    keypoints1_2d = keypoints1[:, :2]  # We need only the x, y coordinates
                    # print("keypoints", keypoints1_2d)
                    score = prediction[0]['bbox_score']
                    # print("score", prediction[0]['bbox_score'])
                    # Flatten the keypoints array and convert to string with comma separator

                    keypoints1_str = ','.join(map(str, keypoints1_2d.flatten()))
                    # keypoints1_str = str(i) + "," + keypoints1_str
                    score_str = ','.join(map(str, score.flatten()))
                    # score_str = "," + score_str
                    
                    
                    # Combine keypoints and scores into one string
                    combined_str = str(i) + "," + score_str + "," + keypoints1_str

                    # Write the keypoints to the .txt file
                    f.write(f"{combined_str}\n")



            
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1

    # Release the video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
# Example usage
inferencer = MMPoseInferencer('rtmo')  # or use appropriate model alias

process_video_save(video_path, inferencer, out_file)
