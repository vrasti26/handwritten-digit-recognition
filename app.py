import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# 1. Apne trained model ko load karein
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('handwritten_digit_model.h5') 

model = load_my_model()

# 2. Web App ki Webpage UI design karein
st.title("✍️ Handwritten Character Recognition")
st.write("Apni handwritten image upload karein aur AI use recognize karega!")

# Image upload karne ka button
uploaded_file = st.file_uploader("Image chuniye...", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    # Image ko screen par dikhane ke liye
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Predicting...")

    # 3. Image ko RGB me convert karein (Taaki channels wala error na aaye)
    rgb_image = image.convert('RGB')

    # 4. Ab converted image ko numpy array me badlein
    img_array = np.array(rgb_image)
    
    # 5. OpenCV se grayscale karein
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 6. Image ko resize aur normalize karein (Jaise training ke waqt kiya tha)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    
    # 7. Model se predict karwayein
    prediction = model.predict(reshaped)
    result = np.argmax(prediction) # Jo highest probability hai use chunna
    
    # Output ko screen par dikhayein
    st.success(f"🎯 AI Prediction: **{result}**")