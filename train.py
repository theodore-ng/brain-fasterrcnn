from dataset_coco import BrainDataset
from config import (
    TRAIN_DATA_DIR,
    TRAIN_COCO,
    MODEL_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_SHUFFLE_DL,
    NUM_WORKERS_DL,
    NUM_CLASSES,
    NUM_EPOCHS,
    LR,
    MOMENTUM,
    WEIGHT_DECAY
)
from model import get_model_instance_segmentation
from utils import collate_fn, get_transform, save_model

import torch

print("Torch version:", torch.__version__)

# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create own Dataset
my_dataset = BrainDataset(
    root=TRAIN_DATA_DIR, annotation=TRAIN_COCO, transforms=get_transform()
)

# own DataLoader
data_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=TRAIN_SHUFFLE_DL,
    num_workers=NUM_WORKERS_DL,
    collate_fn=collate_fn,
)

# DataLoader is iterable over Dataset
for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
print("Dataloader complete!")


model = get_model_instance_segmentation(NUM_CLASSES)

# move model to the right device
model.to(device)

# parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)

len_dataloader = len(data_loader)

# Training
for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch}/{NUM_EPOCHS}")
    model.train()
    i = 0
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")
        
# Save model to directory
save_model(model, MODEL_PATH)
        
print(f"Train completed!")
