from mmpose.apis import MMPoseInferencer
import torch
img_path = '/home/kahina/mmpose/tests/data/coco/000000000785.jpg'   # replace this with your own image path

# instantiate the inferencer using the model alias
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('body26', device=device)

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
