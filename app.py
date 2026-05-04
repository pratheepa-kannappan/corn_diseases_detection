import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import anthropic
import io
import base64
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CornCare AI – Plant Disease Detector",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_NAMES = [
    "Northern Corn Leaf Blight",
    "Gray Leaf Spot",
    "Common Rust",
    "Healthy Corn",
]

CLASS_INFO = {
    "Northern Corn Leaf Blight": {
        "icon": "🍂",
        "color": "#e07b39",
        "description": "A fungal disease favored by high humidity and moderate temperatures, which can lead to significant yield loss if not managed.",
        "symptoms": [
            "Long (1–6 inches), elliptical, grayish-green or tan lesions.",
            "Lesions typically appear on lower leaves first.",
            "In severe cases, lesions can merge, blighting the entire leaf.",
        ],
        "causes": [
            "Caused by the fungus Exserohilum turcicum.",
            "Thrives in cool (65–80°F), moist weather with heavy dews.",
            "Spreads via airborne spores from infected plant debris.",
        ],
        "treatment": [
            "Apply fungicides containing active ingredients like azoxystrobin or propiconazole.",
            "Remove and destroy infected plant debris after harvest.",
            "Improve air circulation through proper plant spacing.",
        ],
        "prevention": [
            "Plant resistant corn hybrids and varieties.",
            "Manage crop residue by tilling to reduce fungal survival.",
            "Apply fungicides when disease is first detected.",
        ],
    },
    "Gray Leaf Spot": {
        "icon": "🔘",
        "color": "#7a9e7e",
        "description": "Caused by the fungus Cercospora zeae-maydis, this disease thrives in warm, humid conditions with extended periods of leaf wetness.",
        "symptoms": [
            "Small, rectangular, brown to gray lesions restricted by leaf veins.",
            "Lesions start on lower leaves and move up the plant.",
            "Can cause premature drying of leaves.",
        ],
        "causes": [
            "Caused by Cercospora zeae-maydis fungus.",
            "Warm temperatures (75–95°F) and prolonged leaf wetness promote spread.",
            "Survives in corn crop residue from previous seasons.",
        ],
        "treatment": [
            "Apply strobilurin or triazole fungicides early in the season.",
            "Till soil to bury infected residue and reduce inoculum.",
            "Scout fields regularly and act at first sign of infection.",
        ],
        "prevention": [
            "Use resistant hybrids — this is the primary management tool.",
            "Practice crop rotation with non-host crops like soybeans.",
            "Tillage helps bury infected residue, reducing inoculum for the next season.",
        ],
    },
    "Common Rust": {
        "icon": "🟠",
        "color": "#c0392b",
        "description": "A fungal disease that produces characteristic rust-colored pustules. It develops in cool, moist weather (60–77°F).",
        "symptoms": [
            "Small, circular to oval, cinnamon-brown pustules on both leaf surfaces.",
            "Pustules rupture the leaf surface, giving it a rough feel.",
            "Yellowing of surrounding leaf tissue as disease progresses.",
        ],
        "causes": [
            "Caused by the fungus Puccinia sorghi.",
            "Presence of rust spores from alternate hosts like Oxalis (wood sorrel).",
            "Cool temperatures (60–77°F) with high humidity or dew.",
        ],
        "treatment": [
            "Apply systemic fungicides containing triazoles or strobilurins, especially early.",
            "Remove infected plant debris to reduce inoculum levels.",
            "Improve air circulation through adequate plant spacing.",
        ],
        "prevention": [
            "Select resistant corn hybrids for the most effective control.",
            "Apply foliar fungicides if severe on susceptible hybrids.",
            "Early planting can allow the crop to mature before rust becomes severe.",
        ],
    },
    "Healthy Corn": {
        "icon": "🌿",
        "color": "#27ae60",
        "description": "A healthy corn plant with no visible signs of disease, pests, or nutrient deficiencies, indicating optimal growing conditions.",
        "symptoms": [
            "Leaves are uniformly green and vibrant.",
            "No spots, lesions, or discoloration on leaves or stalk.",
            "Strong, upright stalk and healthy root system.",
        ],
        "causes": [
            "No pathogen detected.",
            "Plant is in optimal health.",
            "Growing conditions appear favorable.",
        ],
        "treatment": [
            "No treatment required at this time.",
            "Continue current crop management practices.",
            "Monitor regularly for any early signs of stress or disease.",
        ],
        "prevention": [
            "Maintain optimal soil fertility and pH.",
            "Ensure proper watering, avoiding both drought and waterlogging.",
            "Monitor regularly for early signs of pests or disease.",
        ],
    },
}

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f0faf2 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── HERO ── */
.hero {
    min-height: 88vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 80px 24px 60px;
    background: linear-gradient(160deg, #e8f8ee 0%, #f5fbf6 40%, #eaf6f0 100%);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    width: 600px; height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(39,174,96,0.07) 0%, transparent 70%);
    top: -100px; right: -100px;
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(39,174,96,0.1);
    color: #1a7a46;
    border: 1px solid rgba(39,174,96,0.25);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 28px;
}
.hero h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(42px, 6vw, 72px);
    font-weight: 800;
    color: #0d4a28;
    line-height: 1.1;
    margin-bottom: 20px;
    max-width: 800px;
}
.hero h1 span { color: #27ae60; }
.hero p {
    font-size: 18px;
    color: #4a6b58;
    max-width: 520px;
    line-height: 1.7;
    margin-bottom: 40px;
    font-weight: 400;
}
.btn-primary {
    display: inline-block;
    background: linear-gradient(135deg, #f39c12, #e67e22);
    color: white;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 16px;
    padding: 16px 40px;
    border-radius: 100px;
    border: none;
    cursor: pointer;
    text-decoration: none;
    box-shadow: 0 8px 24px rgba(230,126,34,0.35);
    transition: all 0.2s ease;
}
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 12px 32px rgba(230,126,34,0.45); }

/* ── SECTION ── */
.section {
    padding: 80px 24px;
    max-width: 900px;
    margin: 0 auto;
}
.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #0d4a28;
    text-align: center;
    margin-bottom: 10px;
}
.section-sub {
    text-align: center;
    color: #4a6b58;
    font-size: 16px;
    line-height: 1.7;
    margin-bottom: 48px;
}

/* ── UPLOAD AREA ── */
.upload-card {
    background: white;
    border-radius: 24px;
    padding: 48px 40px;
    box-shadow: 0 4px 40px rgba(13,74,40,0.08);
    border: 1px solid rgba(39,174,96,0.12);
}

/* Streamlit file uploader override */
[data-testid="stFileUploader"] {
    border: 2px dashed #a8d5b5 !important;
    border-radius: 16px !important;
    background: #f8fdf9 !important;
    padding: 32px !important;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"]:hover { border-color: #27ae60 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] svg { color: #7bbf94 !important; }

/* ── RESULT CARD ── */
.result-card {
    background: white;
    border-radius: 24px;
    padding: 40px;
    box-shadow: 0 4px 40px rgba(13,74,40,0.1);
    border: 1px solid rgba(39,174,96,0.15);
    margin-top: 32px;
}
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    padding-bottom: 20px;
    border-bottom: 1px solid #edf7f0;
    flex-wrap: wrap;
    gap: 16px;
}
.result-label {
    font-size: 12px;
    font-weight: 600;
    color: #7bbf94;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}
.result-name {
    font-size: 28px;
    font-weight: 800;
    color: #0d4a28;
}
.severity-ring {
    width: 80px; height: 80px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    font-weight: 800;
    font-size: 20px;
    color: #0d4a28;
    background: conic-gradient(var(--ring-color) var(--ring-pct), #edf7f0 0);
    position: relative;
}
.severity-ring::after {
    content: '';
    position: absolute;
    width: 60px; height: 60px;
    border-radius: 50%;
    background: white;
}
.severity-num { position: relative; z-index: 1; }
.severity-txt { position: relative; z-index: 1; font-size: 9px; font-weight: 600; color: #7bbf94; }

/* Info columns */
.info-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; margin: 24px 0; }
@media (max-width: 700px) { .info-grid { grid-template-columns: 1fr; } }
.info-col h4 {
    font-size: 13px;
    font-weight: 700;
    color: #0d4a28;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}
.info-col ul { list-style: none; padding: 0; }
.info-col ul li {
    font-size: 13.5px;
    color: #3d5a49;
    padding: 5px 0 5px 18px;
    position: relative;
    line-height: 1.5;
}
.info-col ul li::before {
    content: '▸';
    position: absolute;
    left: 0;
    color: #27ae60;
    font-size: 11px;
    top: 7px;
}

/* AI Explanation */
.ai-box {
    background: linear-gradient(135deg, #f0faf4, #e8f5ec);
    border-radius: 16px;
    padding: 28px;
    margin-top: 24px;
    border: 1px solid rgba(39,174,96,0.2);
}
.ai-box-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    color: #0d4a28;
    margin-bottom: 12px;
    font-size: 15px;
}
.ai-box p { color: #3d5a49; font-size: 14.5px; line-height: 1.75; }

/* ── DISEASE LIBRARY ── */
.lib-card {
    background: white;
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 20px;
    box-shadow: 0 2px 20px rgba(13,74,40,0.06);
    border: 1px solid rgba(39,174,96,0.1);
    border-left: 5px solid var(--accent);
}
.lib-card h3 { font-size: 20px; font-weight: 700; color: #0d4a28; margin-bottom: 6px; }
.lib-card .desc { color: #4a6b58; font-size: 14px; margin-bottom: 20px; }
.lib-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 600px) { .lib-grid { grid-template-columns: 1fr; } }
.lib-col h5 { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #27ae60; margin-bottom: 10px; }
.lib-col ul { list-style: none; padding: 0; }
.lib-col ul li { font-size: 13.5px; color: #3d5a49; padding: 4px 0 4px 16px; position: relative; line-height: 1.5; }
.lib-col ul li::before { content: '•'; position: absolute; left: 0; color: #7bbf94; }

/* ── NAV ── */
.topnav {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 1000;
    background: rgba(240,250,242,0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(39,174,96,0.12);
    padding: 14px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.nav-logo { font-size: 20px; font-weight: 800; color: #0d4a28; }
.nav-logo span { color: #27ae60; }
.nav-links { display: flex; gap: 32px; }
.nav-links a { text-decoration: none; color: #4a6b58; font-size: 14px; font-weight: 600; transition: color 0.2s; }
.nav-links a:hover { color: #0d4a28; }

/* ── FOOTER ── */
.footer {
    background: #0d4a28;
    color: rgba(255,255,255,0.5);
    text-align: center;
    padding: 32px;
    font-size: 13px;
}
.footer span { color: #7bbf94; font-weight: 600; }

/* Streamlit adjustments */
.stButton > button {
    background: linear-gradient(135deg, #27ae60, #1e8449) !important;
    color: white !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-size: 15px !important;
    box-shadow: 0 4px 16px rgba(39,174,96,0.3) !important;
    width: 100%;
}
.stButton > button:hover { transform: translateY(-1px) !important; }
.stSpinner > div { color: #27ae60 !important; }
.stImage { border-radius: 16px; overflow: hidden; }
div[data-testid="stImage"] img { border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
@st.cache_resource
def load_model():
    """Load the corn disease classification model."""
    try:
        model_data = torch.load('best_model.pth', map_location='cpu', weights_only=False)

        if isinstance(model_data, dict):
            if 'state_dict' in model_data or 'model_state_dict' in model_data:
                sd = model_data.get('state_dict', model_data.get('model_state_dict'))
                arch = detect_architecture(sd)
                model = build_model(arch, num_classes=4)
                model.load_state_dict(sd, strict=False)
            else:
                arch = detect_architecture(model_data)
                model = build_model(arch, num_classes=4)
                try:
                    model.load_state_dict(model_data, strict=False)
                except:
                    pass  # use random weights
        else:
            model = model_data

        model.eval()
        return model, None

    except Exception as e:
        # Corrupted model file — load a default ResNet50 with random weights
        st.warning("⚠️ Could not load best_model.pth (file may be corrupted). Running with default model weights — predictions will be inaccurate until you replace the model file.")
        model = build_model('resnet50', num_classes=4)
        model.eval()
        return model, None


def detect_architecture(state_dict):
    """Guess architecture from state dict keys."""
    keys = list(state_dict.keys())
    key_str = ' '.join(keys[:20])
    if 'features' in key_str and 'classifier' in key_str:
        return 'efficientnet_b0'
    elif 'layer4' in key_str:
        return 'resnet50'
    elif 'layer3' in key_str:
        return 'resnet34'
    elif 'layer1' in key_str:
        return 'resnet18'
    else:
        return 'resnet50'  # safe default


def build_model(arch, num_classes=4):
    """Build model with correct architecture."""
    if arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet34':
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # resnet50
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(model, image: Image.Image):
    """Run inference and return (class_name, confidence, all_probs)."""
    img_tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = probs.argmax().item()
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]) * 100, {CLASS_NAMES[i]: float(probs[i]) * 100 for i in range(4)}


# ─────────────────────────────────────────────
# ANTHROPIC AI EXPLANATION
# ─────────────────────────────────────────────
def get_ai_explanation(disease: str, confidence: float, all_probs: dict, image: Image.Image) -> str:
    """Get AI explanation via Anthropic API with vision."""
    try:
        client = anthropic.Anthropic(api_key="Your_API_KEY")

        # Convert image to base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

        probs_str = "\n".join([f"- {k}: {v:.1f}%" for k, v in all_probs.items()])

        prompt = f"""You are an expert plant pathologist AI. You analyzed a corn leaf image and the deep learning model predicted:

**Predicted Disease**: {disease}
**Confidence**: {confidence:.1f}%

All class probabilities:
{probs_str}

Look at the image carefully and explain IN 3-4 SENTENCES:
1. What visual features in the image led to this prediction (mention specific colors, patterns, shapes you see).
2. Why the model is {confidence:.0f}% confident (what makes the signs clear or ambiguous).
3. One actionable recommendation for the farmer.

Be specific, practical, and refer to actual things visible in the image. Do not use markdown formatting."""

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "⚠️ Invalid API key. Please check your Anthropic API key in the sidebar."
    except Exception as e:
        return f"⚠️ Could not fetch AI explanation: {str(e)}"


# ─────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────
def generate_pdf_report(disease, confidence, all_probs, ai_explanation, image):
    """Generate a downloadable PDF report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=60, bottomMargin=60)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'],
        fontSize=26, fontName='Helvetica-Bold', textColor=colors.HexColor('#0d4a28'),
        spaceAfter=6, alignment=TA_CENTER)
    story.append(Paragraph("🌽 CornCare AI — Disease Report", title_style))

    date_style = ParagraphStyle('Date', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor('#4a6b58'), alignment=TA_CENTER, spaceAfter=20)
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", date_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#a8d5b5'), spaceAfter=20))

    # Disease Result
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=18, fontName='Helvetica-Bold', textColor=colors.HexColor('#0d4a28'), spaceAfter=8)
    story.append(Paragraph("Diagnosis Result", h2))

    result_data = [
        ["Detected Disease", disease],
        ["Confidence", f"{confidence:.1f}%"],
        ["Status", "Healthy ✓" if disease == "Healthy Corn" else "Disease Detected ⚠️"],
    ]
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f5ec')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0d4a28')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#f0faf4'), colors.HexColor('#ffffff')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#a8d5b5')),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 20))

    # Class Probabilities
    story.append(Paragraph("Classification Probabilities", h2))
    prob_data = [["Disease", "Probability"]] + [[k, f"{v:.1f}%"] for k, v in sorted(all_probs.items(), key=lambda x: -x[1])]
    prob_table = Table(prob_data, colWidths=[4*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d4a28')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0faf4'), colors.HexColor('#ffffff')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#a8d5b5')),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 20))

    # AI Explanation
    if ai_explanation and "⚠️" not in ai_explanation:
        story.append(Paragraph("AI Expert Analysis", h2))
        body = ParagraphStyle('Body', parent=styles['Normal'],
            fontSize=11, textColor=colors.HexColor('#3d5a49'), leading=18, spaceAfter=16)
        story.append(Paragraph(ai_explanation.replace('\n', '<br/>'), body))

    # Disease Info
    info = CLASS_INFO.get(disease, {})
    for section, items in [("Symptoms", info.get("symptoms", [])),
                            ("Causes", info.get("causes", [])),
                            ("Treatment", info.get("treatment", [])),
                            ("Prevention", info.get("prevention", []))]:
        story.append(Paragraph(section, h2))
        for item in items:
            bullet = ParagraphStyle('Bullet', parent=styles['Normal'],
                fontSize=11, textColor=colors.HexColor('#3d5a49'), leading=18,
                leftIndent=16, spaceAfter=4)
            story.append(Paragraph(f"• {item}", bullet))
        story.append(Spacer(1, 10))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#a8d5b5'), spaceBefore=20, spaceAfter=10))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#7bbf94'), alignment=TA_CENTER)
    story.append(Paragraph("Generated by CornCare AI • Powered by Deep Learning + Claude AI", footer_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ─────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="topnav">
    <div class="nav-logo">🌽 Corn<span>Care</span> AI</div>
    <div class="nav-links">
        <a href="?page=home">Home</a>
        <a href="?page=diagnose">Diagnose</a>
        <a href="?page=library">Disease Library</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Handle nav from URL params
query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"]

# Spacer for fixed nav
st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — API KEY
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Anthropic API Key")
    st.markdown("Enter your key to enable AI-powered explanations.")
    api_key_input = st.text_input("API Key", value=st.session_state.api_key,
                                   type="password", placeholder="sk-ant-...")
    if api_key_input:
        st.session_state.api_key = api_key_input
    st.markdown("---")
    st.markdown("**📋 Navigation**")
    if st.button("🏠 Home"):
        st.session_state.page = "home"
        st.rerun()
    if st.button("🔬 Diagnose"):
        st.session_state.page = "diagnose"
        st.rerun()
    if st.button("📚 Disease Library"):
        st.session_state.page = "library"
        st.rerun()

# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🌿 AI-Powered Crop Health</div>
        <h1>Identify and Cure<br><span>Corn Plant Diseases</span></h1>
        <p>Is your corn crop looking sick? Try our AI-powered tool to identify the cause and get extensive disease and care info in a snap.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 Diagnose Now", key="hero_btn"):
            st.session_state.page = "diagnose"
            st.rerun()

    st.markdown("""
    <div style="padding: 80px 24px; max-width: 900px; margin: 0 auto; text-align: center;">
        <div class="section-title">Your Personal Plant Doctor</div>
        <p class="section-sub">Simply snap a photo of the issue to get a diagnosis. Our AI will give you detailed info
        on the disease, what caused it, how to treat it, and how to prevent it.</p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-top: 40px; text-align: left;">
            <div style="background: white; border-radius: 20px; padding: 32px; box-shadow: 0 4px 20px rgba(13,74,40,0.07); border-top: 4px solid #27ae60;">
                <div style="font-size: 36px; margin-bottom: 16px;">📸</div>
                <h3 style="font-size: 18px; font-weight: 700; color: #0d4a28; margin-bottom: 10px;">Upload a Photo</h3>
                <p style="color: #4a6b58; font-size: 14px; line-height: 1.6;">Take a clear photo of your corn leaf and upload it to our system.</p>
            </div>
            <div style="background: white; border-radius: 20px; padding: 32px; box-shadow: 0 4px 20px rgba(13,74,40,0.07); border-top: 4px solid #f39c12;">
                <div style="font-size: 36px; margin-bottom: 16px;">🤖</div>
                <h3 style="font-size: 18px; font-weight: 700; color: #0d4a28; margin-bottom: 10px;">AI Analysis</h3>
                <p style="color: #4a6b58; font-size: 14px; line-height: 1.6;">Our trained deep learning model instantly detects the disease with precision.</p>
            </div>
            <div style="background: white; border-radius: 20px; padding: 32px; box-shadow: 0 4px 20px rgba(13,74,40,0.07); border-top: 4px solid #e74c3c;">
                <div style="font-size: 36px; margin-bottom: 16px;">💊</div>
                <h3 style="font-size: 18px; font-weight: 700; color: #0d4a28; margin-bottom: 10px;">Get Treatment</h3>
                <p style="color: #4a6b58; font-size: 14px; line-height: 1.6;">Receive detailed causes, treatment steps, and prevention strategies.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: DIAGNOSE
# ─────────────────────────────────────────────
elif st.session_state.page == "diagnose":
    st.markdown("""
    <div style="padding: 60px 24px 0; text-align: center;">
        <div class="section-title">Your Personal Plant Doctor</div>
        <p class="section-sub">Simply snap a photo of the issue to get a diagnosis. Our AI will give you detailed info<br>
        on the disease, what caused it, how to treat it, and how to prevent it.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        with st.container():
            uploaded_file = st.file_uploader(
                "Click to upload an image or drag and drop",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="visible"
            )

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    diagnose_clicked = st.button("🔬 Diagnose Now")
                with col_b:
                    if st.button("✕ Clear"):
                        st.session_state.result = None
                        st.rerun()

                if diagnose_clicked:
                    with st.spinner("Analyzing your corn leaf..."):
                        model, err = load_model()
                        if err:
                            st.error(f"Model load error: {err}")
                        else:
                            disease, confidence, all_probs = predict(model, image)
                            st.session_state.result = {
                                "disease": disease,
                                "confidence": confidence,
                                "all_probs": all_probs,
                                "image": image,
                            }

        # Show result
        if st.session_state.result:
            r = st.session_state.result
            disease = r["disease"]
            confidence = r["confidence"]
            all_probs = r["all_probs"]
            img = r["image"]
            info = CLASS_INFO[disease]
            color = info["color"]
            severity = min(int(confidence), 99)

            # Get AI explanation if key provided
            ai_text = ""
            if st.session_state.api_key:
                with st.spinner("Getting AI expert analysis..."):
                    ai_text = get_ai_explanation(disease, confidence, all_probs, img)
            # Result card
            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <div>
                        <div class="result-label">🔬 Diagnosis Result</div>
                        <div class="result-name" style="color:{color};">{info['icon']} {disease}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:11px;color:#7bbf94;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Confidence</div>
                        <div style="font-size:36px;font-weight:800;color:{color};">{confidence:.0f}%</div>
                    </div>
                </div>
                <p style="color:#4a6b58;font-size:15px;line-height:1.7;margin-bottom:24px;">{info['description']}</p>

                <div class="info-grid">
                    <div class="info-col">
                        <h4>Possible Causes</h4>
                        <ul>{''.join(f"<li>{c}</li>" for c in info['causes'])}</ul>
                    </div>
                    <div class="info-col">
                        <h4>Treatment</h4>
                        <ul>{''.join(f"<li>{t}</li>" for t in info['treatment'])}</ul>
                    </div>
                    <div class="info-col">
                        <h4>Prevention</h4>
                        <ul>{''.join(f"<li>{p}</li>" for p in info['prevention'])}</ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if ai_text:
                st.markdown(f"""
                <div class="ai-box">
                    <div class="ai-box-title">🤖 Why did the AI predict this?</div>
                    <p>{ai_text}</p>
                </div>
                """, unsafe_allow_html=True)
            elif not st.session_state.api_key:
                st.markdown("""
                <div class="ai-box">
                    <div class="ai-box-title">💡 Enable AI Explanations</div>
                    <p>Add your Anthropic API key in the sidebar (⬅) to get expert visual analysis explaining exactly why this prediction was made.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Probability bar chart
            st.markdown("#### 📊 All Class Probabilities")
            sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
            for cls, prob in sorted_probs:
                bar_color = CLASS_INFO[cls]["color"]
                st.markdown(f"""
                <div style="margin: 8px 0;">
                    <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:600;color:#0d4a28;margin-bottom:4px;">
                        <span>{CLASS_INFO[cls]['icon']} {cls}</span><span>{prob:.1f}%</span>
                    </div>
                    <div style="background:#edf7f0;border-radius:100px;height:8px;">
                        <div style="background:{bar_color};width:{prob}%;height:8px;border-radius:100px;transition:width 0.5s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # PDF Download
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            pdf_bytes = generate_pdf_report(disease, confidence, all_probs, ai_text, img)
            st.download_button(
                label="📥 Download Full Report (PDF)",
                data=pdf_bytes,
                file_name=f"corncare_report_{disease.replace(' ', '_').lower()}.pdf",
                mime="application/pdf",
            )


# ─────────────────────────────────────────────
# PAGE: LIBRARY
# ─────────────────────────────────────────────
elif st.session_state.page == "library":
    st.markdown("""
    <div style="padding: 60px 24px 0; text-align: center;">
        <div class="section-title">Corn Disease Library</div>
        <p class="section-sub">Learn to identify common corn diseases with our detailed guide.<br>Each entry includes symptoms and preventative measures.</p>
    </div>
    <div style="max-width: 860px; margin: 0 auto; padding: 0 24px 80px;">
    """, unsafe_allow_html=True)

    for disease, info in CLASS_INFO.items():
        color = info["color"]
        syms = "".join(f"<li>{s}</li>" for s in info["symptoms"])
        prevs = "".join(f"<li>{p}</li>" for p in info["prevention"])
        st.markdown(f"""
        <div class="lib-card" style="--accent:{color};">
            <h3>{info['icon']} {disease}</h3>
            <div class="desc">{info['description']}</div>
            <div class="lib-grid">
                <div class="lib-col">
                    <h5>Common Symptoms</h5>
                    <ul>{syms}</ul>
                </div>
                <div class="lib-col">
                    <h5>Prevention Measures</h5>
                    <ul>{prevs}</ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>CornCare AI</span> • Built with Deep Learning •
    Helping farmers protect their crops
</div>
""", unsafe_allow_html=True)
