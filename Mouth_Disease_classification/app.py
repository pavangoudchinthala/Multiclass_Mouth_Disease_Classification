import os
import gdown

MODEL_PATH = "oral_efficientnet_b3_V2.keras"

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1yMmRHXSETa8ATVOBzbnJfooyc8XSv30Y"
    gdown.download(url, MODEL_PATH, quiet=False)


import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Oral Disease Classifier",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-box {
        padding: 1rem;
        border-radius: 5px;
        background: #f0f2f6;
        margin: 0.5rem 0;
        color: #000;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# Class names from training
CLASS_NAMES = ['Calculus', 'Caries', 'Gingivitis', 'Healthy', 'Hypodontia', 'Mouth_Ulcer', 'Tooth_Discoloration']

# Disease descriptions
DISEASE_INFO = {
    'Calculus': {
        'description': 'Dental calculus (tartar) is hardened dental plaque that forms on teeth.',
        'icon': '',
        'color': '#FF6B6B'
    },
    'Caries': {
        'description': 'Dental caries (cavities) are permanently damaged areas in teeth.',
        'icon': '',
        'color': '#4ECDC4'
    },
    'Gingivitis': {
        'description': 'Gingivitis is inflammation of the gums caused by bacterial infection.',
        'icon': '',
        'color': '#95E1D3'
    },
    'Healthy': {
        'description': 'Healthy teeth and gums with no visible diseases or abnormalities. Good oral hygiene maintained.',
        'icon': '',
        'color': '#4CAF50'
    },
    'Hypodontia': {
        'description': 'Hypodontia is the developmental absence of one or more teeth.',
        'icon': '',
        'color': '#F38181'
    },
    'Mouth_Ulcer': {
        'description': 'Mouth ulcers are painful sores that appear in the mouth.',
        'icon': '',
        'color': '#AA96DA'
    },
    'Tooth_Discoloration': {
        'description': 'Tooth discoloration is abnormal tooth color, hue or translucency.',
        'icon': '',
        'color': '#FCBAD3'
    }
}

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Resize to model input size
    img = image.resize((300, 300))
    # Convert to array
    img_array = np.array(img)
    # Ensure RGB
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)
    # Normalize (EfficientNet preprocessing)
    img_array = img_array.astype('float32')
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image):
    """Make prediction on image"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    return predictions[0]

# Header
st.markdown('<h1 class="main-header">Oral Health Classification System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-Powered Dental Diagnosis | EfficientNet-B3 V2 | 85.73% Accuracy</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/tooth.png", width=100)
    st.markdown("### About")
    st.info("""
    This AI system uses EfficientNet-B3 deep learning architecture to classify oral conditions from images.
    
    **Model Performance (V2):**
    - Test Accuracy: 85.73%
    - Classes: 7 oral conditions
    - Dataset: 4,963 images
    - Total Parameters: 11.18M
    
    **Top Performing Classes:**
    - Healthy Teeth: 98% accuracy
    - Mouth Ulcer: 95% precision
    - Hypodontia: 96% precision
    """)
    
    st.markdown("### Detected Conditions")
    for disease in CLASS_NAMES:
        info = DISEASE_INFO[disease]
        st.markdown(f"**{disease}**")
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload an oral/dental image
    2. View the prediction results
    3. Check confidence scores
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an oral/dental image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the oral condition"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Analyze button with custom color
        analyze_button = st.button("Click to Analyze", key="analyze_btn", use_container_width=True)
        
        # Display image info
        st.markdown("**Image Details:**")
        st.text(f"Format: {image.format}")
        st.text(f"Size: {image.size}")
        st.text(f"Mode: {image.mode}")
    else:
        analyze_button = False

with col2:
    st.markdown("### Analysis Results")
    
    if uploaded_file is not None and analyze_button:
        # Load model
        model = load_model()
        
        if model is not None:
            with st.spinner('Analyzing image...'):
                # Make prediction
                predictions = predict(model, image)
                predicted_class_idx = np.argmax(predictions)
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = predictions[predicted_class_idx] * 100
                
                # Display prediction
                info = DISEASE_INFO[predicted_class]
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{predicted_class}</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                    <p>{info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display all class probabilities
                st.markdown("#### Confidence Scores", unsafe_allow_html=True)
                
                # Sort predictions by confidence
                sorted_indices = np.argsort(predictions)[::-1]
                
                for idx in sorted_indices:
                    class_name = CLASS_NAMES[idx]
                    class_confidence = predictions[idx] * 100
                    info = DISEASE_INFO[class_name]
                    
                    st.markdown(f"""
                    <div class="confidence-box">
                        <strong style="color: #000;">{class_name}</strong>
                        <div style="background: #ddd; border-radius: 5px; margin-top: 0.5rem;">
                            <div style="background: {info['color']}; width: {class_confidence}%; 
                                        padding: 0.3rem; border-radius: 5px; text-align: center; color: #000; font-weight: bold;">
                                {class_confidence:.2f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Warning message
                st.warning("**Disclaimer:** This is an AI-based diagnostic tool and should not replace professional medical advice. Please consult a dentist for proper diagnosis and treatment.")
        else:
            st.error("Failed to load the model. Please ensure 'oral_efficientnet_b3_V2.keras' exists in the project directory.")
    elif uploaded_file is not None:
        st.info("Click the 'Click to Analyze' button to analyze the image")
    else:
        st.info("Upload an image to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Powered by EfficientNet-B3 V2 | Built with Streamlit & TensorFlow</p>
    <p>Model trained on 4,963 images across 7 oral classifications | Accuracy: 85.73%</p>
    <p style="font-size: 0.9rem; color: #999; margin-top: 1rem;\">Disclaimer: For educational and screening purposes. Always consult a dentist for professional diagnosis.</p>
</div>
""", unsafe_allow_html=True)
