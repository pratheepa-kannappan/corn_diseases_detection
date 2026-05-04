# 🌽 CornCare AI — Corn Disease Detector

An AI-powered Streamlit web app to detect corn plant diseases from leaf images.

## Features
- 🔬 Deep learning model inference (ResNet / EfficientNet)
- 📥 PDF report download with full diagnosis details
- 📚 Built-in disease library for 4 corn conditions
- 🎨 Beautiful custom UI matching your design mockups

## Supported Diseases
| Disease | Description |
|---|---|
| Northern Corn Leaf Blight | Fungal, large tan/gray lesions |
| Gray Leaf Spot | Rectangular brown-gray lesions |
| Common Rust | Rust-colored oval pustules |
| Healthy Corn | No disease detected |

## Setup

### 1. Install Dependencies
-crate a virtual envirement and install the requirements
```bash
python -m venv .venv
python --version
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. dowmload the dataset
-Download the dataset from kaggle
-Create model from model folder
-select best model
-Download the best model


### 2. Place Your Model
Copy your `best_model.pth` file into the app folder:
```
corn_disease_app/
├── app.py
├── best_model.pth   ← place here
├── requirements.txt
└── README.md
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Add Your Anthropic API Key
- Open the sidebar (left arrow on screen)
- Paste your Anthropic API key (`sk-ant-...`)
- The AI will now explain WHY each prediction was made

## Getting an Anthropic API Key
1. Go to required api platform
2. Sign up / Log in
3. Go to API Keys → Create Key
4. Copy and paste into the sidebar or add in the code in app.py

## Model Architecture
The app auto-detects your model architecture (ResNet18/34/50 or EfficientNet-B0) from the saved weights. It supports:
- Full model objects
- State dicts (`state_dict` or `model_state_dict` keys)
- Raw state dicts

## File Structure
```
corn_disease_app/
├── app.py              # Main Streamlit application
├── best_model.pth      # Your trained PyTorch model
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
