import streamlit as st
import requests
import base64
import io

from test import predict_image
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

##### Set up sidebar.

# Add in location to select image.
st.sidebar.write("### Select an image to upload.")
uploaded_file = st.sidebar.file_uploader(
    "", type=["png", "jpg", "jpeg"], accept_multiple_files=False
)

## Add in sliders.
confidence_threshold = st.sidebar.slider("Confidence threshold:", 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider("Overlap threshold:", 0.0, 1.0, 0.5, 0.01)

image = Image.open("./images/pytorch.jpg")
st.sidebar.image(image, use_column_width=True)

##### Set up main app.

## Title.
st.write("# Brain CT Object Detection")

input_col, result_col = st.columns(2)

# Pull in default image or user-selected image.
with input_col:
    # Subtitle
    st.header("Input image")
    if uploaded_file is None:
        # Default image.
        url = "https://theodore-ng.github.io/brain-fasterrcnn/images/samples/53-origin.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

    else:
        # User-selected image.
        image = Image.open(uploaded_file)

    # Display image.
    st.image(image, use_column_width=True)

with result_col:
    # Subtitle
    st.header("Result image")
    if uploaded_file is None:
        # Default image.
        url = "https://theodore-ng.github.io/brain-fasterrcnn/images/samples/53-result.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

    else:
        # User-selected image.
        image = Image.open(uploaded_file)

    # Display image.
    st.image(image, use_column_width=True)


# # Convert to JPEG Buffer.
# buffered = io.BytesIO()
# image.save(buffered, quality=90, format='JPEG')

# # Base 64 encode.
# img_str = base64.b64encode(buffered.getvalue())
# img_str = img_str.decode('ascii')

# ## Construct the URL to retrieve image.
# upload_url = ''.join([
#     'https://infer.roboflow.com/rf-bccd-bkpj9--1',
#     f'?access_token={st.secrets["access_token"]}',
#     '&format=image',
#     f'&overlap={overlap_threshold * 100}',
#     f'&confidence={confidence_threshold * 100}',
#     '&stroke=2',
#     '&labels=True'
# ])

# ## POST to the API.
# r = requests.post(upload_url,
#                   data=img_str,
#                   headers={
#     'Content-Type': 'application/x-www-form-urlencoded'
# })

# image = Image.open(BytesIO(r.content))

# # Convert to JPEG Buffer.
# buffered = io.BytesIO()
# image.save(buffered, quality=90, format='JPEG')

# ## Construct the URL to retrieve JSON.
# upload_url = ''.join([
#     'https://infer.roboflow.com/rf-bccd-bkpj9--1',
#     f'?access_token={st.secrets["access_token"]}'
# ])

# ## POST to the API.
# r = requests.post(upload_url,
#                   data=img_str,
#                   headers={
#     'Content-Type': 'application/x-www-form-urlencoded'
# })

# ## Save the JSON.
# output_dict = r.json()

# ## Generate list of confidences.
# confidences = [box['confidence'] for box in output_dict['predictions']]

# ## Summary statistics section in main app.
# st.write('### Summary Statistics')
# st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
# st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')

# ## Histogram in main app.
# st.write('### Histogram of Confidence Levels')
# fig, ax = plt.subplots()
# ax.hist(confidences, bins=10, range=(0.0,1.0))
# st.pyplot(fig)

# ## Display the JSON in main app.
# st.write('### JSON Output')
# st.write(r.json())
