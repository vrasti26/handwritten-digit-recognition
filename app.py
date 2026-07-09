import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 1. Model load karne ka function
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('handwritten_digit_model.h5')

model = load_my_model()

# 2. Web App ki Webpage UI design karein
st.title("✍️ Handwritten Character Recognition")
st.write("Apni handwritten image upload karein aur AI use recognize karega!")

# Image upload karne ka button
uploaded_file = st.file_uploader("Image chuniye...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Step A: Pehle original image ko open karo
    image = Image.open(uploaded_file)
    
    # Step B: Copy ki photo ka white background black karne ke liye preprocessing
    image_gray = image.convert('L')          # Grayscale mein convert kiya
    image_inverted = ImageOps.invert(image_gray) # Colors ulte kiye (White -> Black)
    
    # Step C: Screen par processed image dikhane ke liye (clean UI, no warnings)
    st.image(image_inverted, caption='Processed Image (MNIST Format)', use_container_width=True)
    st.write("Predicting...")
    
    # Step D: CNN Model ke liye image ko resize aur 4D shape (1, 28, 28, 1) mein convert karna
    img_resized = image_inverted.resize((28, 28))
    img_array = np.array(img_resized) / 255.0  # Normalization
    
    img_reshaped = np.expand_dims(img_array, axis=0)     # Batch dimension add kiya (1, 28, 28)
    img_reshaped = np.expand_dims(img_reshaped, axis=-1) # Channel dimension add kiya (1, 28, 28, 1)
    
    # Step E: Model se prediction lena
    prediction = model.predict(img_reshaped)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Step F: Output screen par dikhana
    st.success(f"🤖 AI Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
