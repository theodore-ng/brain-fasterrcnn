import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes


# Transformation just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Save model
def save_model(model, path):
    if not os.path.exists(path):
        dir, file = os.path.split(path)
        os.mkdir(dir) 
    torch.save(model.state_dict(), path)
    print(f"Model save to {path}")
    
# Remove the results with low confident rate 
def remove_under_confident(prediction: dict, score: float):
    """prediction is a dictionary with keys:
    boxes: tensor(n, 4)
    labels: tensor(n, 1)
    scores: tensor(n, 1)
    we need to remove all the results with scores under confident score

    Args:
        predictions (dict): _description_
    """
    results = prediction["scores"]
    criteria_meet = [results > score]
    for key, item in prediction.items():
        prediction[key] = item[criteria_meet]
    return prediction

# Draw bboxes and save to a directory
def test_visualization(image, pred_boxes, pred_labels, img_path, colors="blue"):
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors=colors)
    dir, file = os.path.split(img_path)
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except:
            pass
    # Save the output image
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.savefig(img_path)
    print(f"Results save completed at {img_path}.")