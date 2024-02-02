import os

from dataset_coco import BrainDataset
from config import (
    EVAL_DATA_DIR,
    EVAL_COCO,
    MODEL_PATH,
    TEST_DATA_DIR,
    TEST_IMG_DIR, 
    BATCH_SIZE,
    SHUFFLE_DL,
    NUM_WORKERS_DL,
    NUM_CLASSES,
    CONFIDENT_SCORE,
)
from model import get_model_instance_segmentation
from utils import (
    collate_fn,
    get_transform, 
    remove_under_confident, 
    test_visualization, 
    pick_image_example, 
    predict_image
)


import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from PIL import Image

print("Torch version:", torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

eval_transform = get_transform()

# import data from eval dataset
val_dataset = BrainDataset(root=EVAL_DATA_DIR, annotation=EVAL_COCO, transforms=get_transform())

# to data loader
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DL, num_workers=NUM_WORKERS_DL, collate_fn=collate_fn)

# retrieve model
model = get_model_instance_segmentation(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# pick 1 random example from test dataset 
image, image_tensor, img_name = pick_image_example(TEST_DATA_DIR)

# Do the predict
pred_boxes, pred_labels = predict_image(model, image_tensor, CONFIDENT_SCORE) 

# Save the output image
result_path = os.path.join(TEST_IMG_DIR, img_name)
test_visualization(image, pred_boxes, pred_labels, img_path=result_path)