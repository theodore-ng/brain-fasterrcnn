from dataset_coco import BrainDataset
from config import (
    EVAL_DATA_DIR,
    EVAL_COCO,
    MODEL_PATH,
    TEST_IMG_PATH, 
    TRAIN_BATCH_SIZE,
    TRAIN_SHUFFLE_DL,
    NUM_WORKERS_DL,
    NUM_CLASSES,
    CONFIDENT_SCORE,
)
from model import get_model_instance_segmentation
from utils import collate_fn, get_transform, remove_under_confident, test_visualization

import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image

print("Torch version:", torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

eval_transform = get_transform()

# import data from eval dataset
val_dataset = BrainDataset(
    root=EVAL_DATA_DIR, 
    annotation=EVAL_COCO, 
    transforms=get_transform()
)

# to data loader
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=TRAIN_SHUFFLE_DL,
    num_workers=NUM_WORKERS_DL,
    collate_fn=collate_fn,
)

# retrieve model
model = get_model_instance_segmentation(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# take 1 example from test dataset 
# TODO: make function here
image = Image.open("data/test/31_jpg.rf.7ddd7d1b0964a258c819cfc9e721a854.jpg")
image = F.pil_to_tensor(image)  # convert to tensor shape (3,640,640)
image_tensor = F.convert_image_dtype(image)    # covert to type for the model

# Do the predict
with torch.no_grad():
    image_tensor.to("cpu")
    model.to("cpu")
    prediction = model([image_tensor, ])[0]
    prediction = remove_under_confident(prediction, CONFIDENT_SCORE)
    scores = prediction["scores"]
    pred_labels = [f"confident: {score:.3f}" for score in scores]
    pred_boxes = prediction["boxes"].long()
    

# Save the output image
test_visualization(image, pred_boxes, pred_labels, img_path=TEST_IMG_PATH)