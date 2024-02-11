from model_test import predict_image
from config import CONFIDENT_SCORE, NUM_CLASSES, MODEL_PATH
from model import get_model_instance_segmentation
from PIL import Image
from fastapi import FastAPI, UploadFile
import torch
import torchvision.transforms.functional as F
 
# Creating FastAPI instance
app = FastAPI()
 
# @app.post("/files/")
# async def create_file(file: Annotated[bytes, File()]):
#     return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

@app.post("/predict/") 
async def predict_upload_file(file: UploadFile):
    # Get model
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # Convert data
    # contents = file.file.read()
    # print(type(contents))
    # image = Image.open(io.BytesIO(file))
    
    image = Image.open(file.file)
    
    image = F.pil_to_tensor(image)                  # convert to tensor shape (3,640,640)
    image_tensor = F.convert_image_dtype(image)     # covert to type for the model
    
    # prediction
    pred_boxes, pred_labels = predict_image(model, image_tensor, CONFIDENT_SCORE)
    return {
        "boxes": pred_boxes,
        "labels": pred_labels
    }