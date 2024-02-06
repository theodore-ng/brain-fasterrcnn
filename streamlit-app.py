import streamlit as st
import requests
import io

from PIL import Image

# Url from API
url = "http://127.0.0.1:8000/predict/"

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
        # buffered = io.BytesIO()
        # image.save(buffered, quality=90, format="JPEG")
        image.save("image.jpg", "JPEG")
        with open("image.jpg", "rb") as f:
            resp = requests.post(url, files={"file":f})
        st.write(resp.json())

    # Display image.
    st.image(image, use_column_width=True)
