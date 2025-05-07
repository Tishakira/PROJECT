import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the PLD and PLQ models
pld_model = load_model("litter_classification_model.h5")
plq_model = load_model("plq_model.h5")

# Function to preprocess images for PLD
def preprocess_for_pld(image):
    image = cv2.resize(image, (100, 100))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to preprocess images for PLQ
def preprocess_for_plq(image):
    image = cv2.resize(image, (50, 50))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app

# Sidebar
st.sidebar.title("📋 Project Overview")
st.sidebar.info(
    """
    ### 🌍 Importance of the Project
    - **Environmental Impact**: Understanding pollution levels helps address environmental concerns effectively.
    - **Purpose**: This app classifies tile images into pollution levels to aid in monitoring and cleanup efforts.

    ### 🔍 Instructions
    1. Upload an image of the tile.
    2. The app will process the image and classify it as:
       - **Litter - High**: Highly polluted.
       - **Litter - Low**: Mild pollution.
       - **Litter - No**: Clean tile.
    3. View results and download processed images if needed.

    ### ⚙️ Theme Options
    Use the dropdown below to switch between light and dark themes!
    """
)
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #2D2D2D;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
st.title("🌍 Marine Pollution Detection System")
st.write("Welcome! This app analyzes tile images to determine pollution levels and identifies the type of litter for polluted areas.")

# File uploader
uploaded_file = st.file_uploader("Upload a tile image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for PLD model
    preprocessed_pld = preprocess_for_pld(image)

    # Predict using the PLD model
    pld_prediction = pld_model.predict(preprocessed_pld)
    pld_class = np.argmax(pld_prediction)  # Get class with highest probability

    if pld_class == 2:  # Assuming class 2 = "Litter-free"
        st.success("✨ The tile is clean and **litter-free**!")
    else:
        pollution_level = "highly polluted" if pld_class == 0 else "low pollution"
        st.warning(f"⚠️ The tile is **{pollution_level}**.")

        # Preprocess the image for PLQ model
        preprocessed_plq = preprocess_for_plq(image)
        
        # Predict using the PLQ model
        plq_prediction = plq_model.predict(preprocessed_plq)
        plq_class = np.argmax(plq_prediction)  # Get class with highest probability
        
        # Add a more detailed message
        litter_types = {
            litter_types = {
    0: 'Cans',
    1: 'Carton',
    2: 'Other',
    3: 'Plastic bag - large',
    4: 'Plastic bag - small',
    5: 'Plastic bottles',
    6: 'Plastic bowls',
    7: 'Plastic canister',
    8: 'Plastic cups',
    9: 'Plastic other',
    10: 'Polystyrene packaging',
    11: 'Shoes',
    12: 'String and cord',
    13: 'Styrofoam',
    14: 'Textiles'
}
litter_type = litter_types.get(plq_class, "Unknown Litter Type")

        litter_type = litter_types.get(plq_class, "Unknown Litter Type")
        
        st.write(f"🔍 The litter type is classified as: **{litter_type}**.")
