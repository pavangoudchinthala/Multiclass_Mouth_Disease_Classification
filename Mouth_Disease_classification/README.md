# Run these commands in the terminal
pip install streamlit
pip install numpy
pip install tensorflow 
pip install PIL
streamlit run app.py

# Oral Health Classification System - V2

Advanced AI-powered dental disease classification and healthy teeth detection using EfficientNet-B3 deep learning model.

## Overview

This system uses a trained EfficientNet-B3 neural network to automatically classify oral conditions from dental images. It can identify 7 different oral classifications including diseases and healthy teeth, with **85.73% accuracy** on the test dataset.

## 🎯 Features

✅ **7 Classification Classes:**
- Calculus (Tartar)
- Caries (Cavities)
- Gingivitis (Gum Inflammation)
- **Healthy Teeth** (NEW - V2 Model)
- Hypodontia (Missing Teeth)
- Mouth Ulcer
- Tooth Discoloration

✅ **Model Performance (V2):**
- **Overall Accuracy: 85.73%**
- **Healthy Teeth Detection: 98% accuracy**
- **Mouth Ulcer: 95% precision**
- **Hypodontia: 96% precision**
- Total Parameters: 11.18M
- Input Size: 300×300 RGB images

✅ **Web Interface:**
- Interactive Streamlit application
- Real-time image upload and analysis
- Confidence scores for all predictions
- Professional disease information
- Medical disclaimer

✅ **Testing & Organization:**
- Comprehensive test suite on 5,963 images
- Automatic organization of correct/incorrect predictions
- Detailed classification reports
- Accuracy metrics

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- 500+ MB disk space (model is 177 MB)
- 1+ GB RAM for running predictions

### Step 1: Clone or Download the Project

**Option A: Using Git**
```powershell
git clone <repository-url>
cd oral-health-classification
```

**Option B: Download ZIP**
1. Download project ZIP from GitHub
2. Extract to your desired location
3. Open terminal in the extracted folder

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**This installs:**
- streamlit - Web interface
- tensorflow & keras - Model framework
- numpy & Pillow - Image processing
- scikit-learn - Metrics calculation
- matplotlib & seaborn - Visualizations

### Step 4: Verify Installation

```powershell
pip list
```

### Step 5: Verify Model File

Ensure `oral_efficientnet_b3_V2.keras` is in the project root directory (177 MB)

## 🎮 Quick Start

### Option 1: Run the Web Application

#### **Step 1: Activate Virtual Environment**

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

#### **Step 2: Run Streamlit App**

```powershell
streamlit run app.py
```

**Expected Output:**
```
Collecting usage statistics...
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

✅ **Access at: http://localhost:8501**

Then:
1. Upload a dental image (JPG, PNG, JPEG)
2. Click "Click to Analyze" button
3. View predictions with confidence scores
4. Read disease information and medical disclaimer

---

### Option 2: Test All Images (Comprehensive)

#### **Prerequisites:**
- Close any running Streamlit app (`Ctrl+C` in terminal)
- Ensure 10+ GB free disk space (for copying 5,963 images)
- Ensure virtual environment is activated

#### **Run Test:**

```powershell
python test_organize.py
```

#### **Expected Output:**
```
======================================================================
TEST ALL IMAGES - V2 MODEL
======================================================================

[*] Loading model...
[+] Model loaded!

Testing...
[+] Image 1/5963 - Processing...
[+] Image 2/5963 - Processing...
...
[+] Test Complete!
[+] Accuracy: 85.73%
```

#### **Results Location:**

The script creates: `test_results_organized/`

```
test_results_organized/
├── correct_predictions/
│   ├── Calculus/
│   ├── Caries/
│   ├── Gingivitis/
│   ├── Healthy_Teeth/
│   ├── Hypodontia/
│   ├── Mouth_Ulcer/
│   └── Tooth_Discoloration/
├── wrong_predictions/
│   ├── Calculus/
│   ├── Caries/
│   ├── Gingivitis/
│   ├── Healthy_Teeth/
│   ├── Hypodontia/
│   ├── Mouth_Ulcer/
│   └── Tooth_Discoloration/
└── test_summary.txt              # Summary with accuracy percentage
```

#### **Check Results:**

```powershell
# View summary
type test_results_organized\test_summary.txt

# View correct predictions count
dir test_results_organized\correct_predictions -Recurse | Measure-Object | Select-Object Count

# View wrong predictions count
dir test_results_organized\wrong_predictions -Recurse | Measure-Object | Select-Object Count
```

#### **Expected Runtime:** 15-20 minutes for all 5,963 images

---

### Test All Images (Quick)

## 📖 Usage Guide

### Web Application

1. **Upload Image:** Click "Browse files" and select a dental image (JPG, PNG, JPEG)
2. **Click Analyze:** Press the blue "Click to Analyze" button
3. **View Results:**
   - Predicted disease/condition with confidence
   - Disease description and information
   - Confidence scores for all 7 classes
   - Visual confidence bars
4. **Medical Reminder:** Read the disclaimer about consulting a dentist

## 📁 Project Structure

```
cnn/
├── app.py                              # Streamlit web interface
├── test_organize.py                    # Comprehensive test script
├── oral_efficientnet_b3_V2.keras       # Trained V2 model (177 MB)
├── requirements.txt                    # Python dependencies
├── README.md                           # Documentation
├── prepared_data/
│   ├── train/                          # Training images (4,172)
│   ├── val/                            # Validation images (894)
│   └── test/                           # Test images (897)
├── test_results_organized/             # Test output folder
│   ├── correct_predictions/            # Correct classifications
│   ├── wrong_predictions/              # Misclassifications
│   └── test_summary.txt                # Detailed test report
└── .venv/                              # Python virtual environment
```

## 🔧 Model Information

### V2 Model Specifications

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B3 |
| Input Size | 300×300 RGB |
| Total Parameters | 11,178,806 |
| Model File Size | 177 MB |
| Test Accuracy | 85.73% |
| Classes | 7 (with Healthy Teeth) |
| Framework | TensorFlow 2.15.0 |

### Performance by Class

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Calculus | 70% | 75% | 73% |
| Caries | 93% | 76% | 83% |
| Gingivitis | 85% | 84% | 84% |
| **Healthy_Teeth** | **98%** | **98%** | **98%** |
| Hypodontia | 96% | 98% | 97% |
| Mouth_Ulcer | 95% | 100% | 98% |
| Tooth_Discoloration | 95% | 64% | 77% |

### Training Data

- **Total Images:** 4,963
- **Train Split:** 4,172 images (69.96%)
- **Validation Split:** 894 images (14.99%)
- **Test Split:** 897 images (15.05%)
- **Image Format:** JPG, PNG
- **Preprocessing:** Resize to 300×300, RGB conversion, normalization

## 📦 Requirements

```
streamlit==1.28.0
tensorflow==2.15.0
keras==2.15.0
numpy==1.24.3
Pillow==10.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
reportlab>=4.0.0
```

## ⚙️ Python Integration

Use the model programmatically:

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model('oral_efficientnet_b3_V2.keras', compile=False)

# Prepare image
img = Image.open('image.jpg').convert('RGB')
img = img.resize((300, 300))
img_array = np.expand_dims(np.array(img, dtype='float32'), 0)

# Predict
predictions = model.predict(img_array, verbose=0)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx] * 100

CLASS_NAMES = ['Calculus', 'Caries', 'Gingivitis', 'Healthy_Teeth', 
               'Hypodontia', 'Mouth_Ulcer', 'Tooth_Discoloration']

print(f"Prediction: {CLASS_NAMES[class_idx]}")
print(f"Confidence: {confidence:.2f}%")
```

## ⚠️ Important Disclaimers

**MEDICAL DISCLAIMER:**

This AI system is designed for **educational and preliminary screening purposes only** and should **NOT** be used as a substitute for professional medical diagnosis or treatment.

- ⚠️ Always consult a qualified dentist or oral healthcare professional
- ⚠️ Results should be verified by licensed medical professionals
- ⚠️ Do not make treatment decisions based solely on this system's output
- ⚠️ In case of dental emergencies, seek immediate professional help

**Use at Your Own Risk** - The creators are not liable for any medical decisions based on this system's output.

## 🛠️ Troubleshooting

### Model Loading Error
```
Error: Failed to load model
Solution: Ensure oral_efficientnet_b3_V2.keras exists in project root
```

### Prediction Issues
```
Solution: Ensure sufficient RAM (500 MB+ available)
         Close other applications
         Check image format (RGB, 300×300 after resize)
```

### Test Script Issues
```
Solution: Close Streamlit app before running tests
         Ensure 10+ GB free disk space for image copies
         Reduce batch size if memory errors occur
```

## 📊 Performance Benchmarks

- **Single Image Prediction:** 0.3-0.5 seconds (CPU)
- **Batch Processing:** 15-20 minutes for 5,963 images
- **Model Load Time:** 10-15 seconds
- **RAM Usage:** ~500 MB during operation

## 📝 File Information

| File | Size | Description |
|------|------|-------------|
| app.py | 8 KB | Streamlit web interface |
| test_organize.py | 4 KB | Comprehensive test script |
| oral_efficientnet_b3_V2.keras | 177 MB | Trained model |
| requirements.txt | <1 KB | Dependencies |
| README.md | 15 KB | Documentation |

## 🚀 Future Improvements

- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Mobile app version
- [ ] Real-time camera feed
- [ ] API endpoint
- [ ] Patient record management
- [ ] Attention visualization (Grad-CAM)
- [ ] Multi-language support

## 📞 Support

For issues or questions:
1. Review the troubleshooting section
2. Check that all files are in correct locations
3. Verify system meets requirements
4. Check terminal for detailed error messages

---

**Version:** 2.0 (Includes Healthy Teeth Detection)  
**Last Updated:** December 5, 2025  
**Model Accuracy:** 85.73%  
**Status:** Production Ready (with medical oversight)  
**License:** Educational Use
- [ ] Create mobile app version

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Add appropriate documentation
- Test changes before submitting

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ using TensorFlow, Streamlit, and EfficientNet-B3**
