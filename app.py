import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Page config
st.set_page_config(
    page_title="Oral Disease Classifier",
    page_icon="🦷",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Main Header */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sub Header */
.sub-header {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 20px;
}

/* Upload Box */
.stFileUploader {
    border: 2px dashed #4facfe;
    padding: 20px;
    border-radius: 10px;
    background-color: #1e293b;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #00f2fe, #4facfe);
}

/* Prediction Box */
.prediction-box {
    padding: 2rem;
    border-radius: 15px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

/* Confidence Box */
.confidence-box {
    padding: 10px;
    border-radius: 8px;
    background: #1e293b;
    margin: 8px 0;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.3);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #4facfe;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Classes
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

TREATMENT_TIPS = {
    "Calculus": "Maintain oral hygiene, regular brushing & professional cleaning.",
    "Caries": "Reduce sugar intake, use fluoride toothpaste, visit dentist.",
    "Gingivitis": "Brush twice daily, floss regularly, use mouthwash.",
    "Healthy": "Maintain good oral hygiene and regular dental checkups.",
    "Hypodontia": "Consult dentist for orthodontic or prosthetic solutions.",
    "Mouth_Ulcer": "Avoid spicy food, use antiseptic gels, stay hydrated.",
    "Tooth_Discoloration": "Avoid staining foods, try whitening treatments."
}

# MODEL LOADING
@st.cache_resource
def load_model():
    import os
    import gdown

    MODEL_PATH = "oral_efficientnet_b3_V2.keras"

    try:
        if not os.path.exists(MODEL_PATH):
            url = "https://drive.google.com/uc?id=10IbkzHOwowedSzu_UNOUZqejneQefMcV"
            gdown.download(url, MODEL_PATH, quiet=False)

        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load once
model = load_model()

# Preprocess
def preprocess_image(image):
    img = image.resize((300, 300))
    img = np.array(img)

    if img.shape[-1] != 3:
        img = np.stack([img]*3, axis=-1)

    img = preprocess_input(img.astype('float32'))
    return np.expand_dims(img, axis=0)

# Predict
def predict(image):
    img = preprocess_image(image)
    preds = model.predict(img, verbose=0)
    return preds[0]


def create_pdf(label, conf, tip, now):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4

    pdf_path = "report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)

    styles = getSampleStyleSheet()
    content = []

    # Custom styles
    title_style = ParagraphStyle(
        'title',
        parent=styles['Title'],
        textColor=colors.darkblue,
        fontSize=18,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        'heading',
        parent=styles['Heading2'],
        textColor=colors.darkgreen,
        spaceAfter=10
    )

    normal_style = styles["Normal"]

    # 🏥 Logo (optional)
    try:
        logo = Image("https://img.icons8.com/fluency/96/tooth.png", width=50, height=50)
        content.append(logo)
    except:
        pass

    # Title
    content.append(Paragraph("ORAL HEALTH REPORT", title_style))
    content.append(Spacer(1, 10))

    # Date
    content.append(Paragraph(f"<b>Date:</b> {now}", normal_style))
    content.append(Spacer(1, 15))

    # Prediction Section
    content.append(Paragraph("PREDICTION RESULT", heading_style))
    content.append(Paragraph(f"Disease: <b>{label}</b>", normal_style))
    content.append(Paragraph(f"Confidence: <b>{conf:.2f}%</b>", normal_style))
    content.append(Spacer(1, 15))

    # Model Section
    content.append(Paragraph("MODEL DETAILS", heading_style))
    content.append(Paragraph("Model: EfficientNet-B3", normal_style))
    content.append(Paragraph("Accuracy: 85.73%", normal_style))
    content.append(Spacer(1, 15))

    # Recommendation Section
    content.append(Paragraph("RECOMMENDED ACTION", heading_style))
    content.append(Paragraph(tip, normal_style))
    content.append(Spacer(1, 15))

    # Disclaimer
    content.append(Paragraph("DISCLAIMER", heading_style))
    content.append(Paragraph("This is for educational purposes only.", normal_style))
    content.append(Paragraph("Consult a dentist for professional advice.", normal_style))
    content.append(Spacer(1, 20))

    # Footer
    content.append(Paragraph("<i>Generated by Oral Health AI System</i>", normal_style))

    doc.build(content)
    return pdf_path

# Header
st.markdown('<h1 class="main-header">Multiclass Mouth Disease Classification System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">EfficientNet-B3 | 85.73% Accuracy</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/tooth.png", width=100)
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload an oral/dental image
    2. View the prediction results
    3. Check confidence scores
    """)
    st.markdown("---")

    st.markdown("### Detected Conditions")
    for disease in CLASS_NAMES:
        st.markdown(f"**{disease}**")

    st.markdown("---")

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


# ---------------- MAIN LAYOUT ----------------

analyze = False

col_upload, col_preview = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])

with col_preview:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=250)

        analyze = st.button("Analyze Image")

# ---------------- PREDICTION ----------------

if uploaded_file and analyze:

    if model:
        with st.spinner("🔍 AI is analyzing the image..."):
            preds = predict(image)

            idx = np.argmax(preds)
            label = CLASS_NAMES[idx]
            conf = preds[idx]*100

            st.subheader("💊 Recommended Action")
            st.info(TREATMENT_TIPS[label])

            info = DISEASE_INFO[label]

            st.markdown(f"""
            <div class="prediction-box">
                <h2>{label}</h2>
                <h3>{conf:.2f}%</h3>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Sort predictions by confidence
            sorted_indices = np.argsort(preds)[::-1]

            import pandas as pd
            import matplotlib.pyplot as plt

            data = {
                "Disease": CLASS_NAMES,
                "Confidence": preds * 100
            }

            df = pd.DataFrame(data)

            st.subheader("📊 Prediction Insights")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("### 📊 Bar Chart")
                fig_bar, ax_bar = plt.subplots()

                df_sorted = df.sort_values(by="Confidence", ascending=True)

                colors = plt.cm.viridis(df_sorted["Confidence"] / df_sorted["Confidence"].max())

                ax_bar.barh(df_sorted["Disease"], df_sorted["Confidence"], color=colors)

                ax_bar.set_xlabel("Confidence (%)")
                ax_bar.set_title("Confidence Distribution")

                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)

                st.pyplot(fig_bar)

            with colB:
                st.markdown("### 🥧 Pie Chart")
                fig, ax = plt.subplots()

                filtered_preds = [p for p in preds if p > 0.01]
                filtered_labels = [CLASS_NAMES[i] for i, p in enumerate(preds) if p > 0.01]

                ax.pie(filtered_preds, labels=filtered_labels, autopct='%1.1f%%')
                st.pyplot(fig)

            # -------- PDF --------
            st.subheader("📄 Download Report")

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            pdf_file = create_pdf(label, conf, TREATMENT_TIPS[label], now)

            with open(pdf_file, "rb") as f:
                st.download_button(
                    "📥 Download PDF Report",
                    f,
                    file_name="Oral_Report.pdf"
                )

    else:
        st.error("Model not loaded")

# Footer (UNCHANGED)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Powered by EfficientNet-B3 V2 | Built with Streamlit & TensorFlow</p>
    <p>Model trained on 4,963 images across 7 oral classifications | Accuracy: 85.73%</p>
    <p style="font-size: 0.9rem; color: #999; margin-top: 1rem;">Disclaimer: For educational and screening purposes. Always consult a dentist for professional diagnosis.</p>
</div>
""", unsafe_allow_html=True)