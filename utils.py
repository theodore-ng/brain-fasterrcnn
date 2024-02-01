import os
import shutil
import torchvision
import torch

# In my case, just added ToTensor
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