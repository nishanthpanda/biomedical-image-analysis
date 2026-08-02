import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import joblib
from google import genai
import os
import tempfile
import datetime

# --- NEW IMPORTS for Grad-CAM & PDF ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from fpdf import FPDF

# --- 1. Global Setup & Model Loading ---

# Device setup
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
               'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 
               'normal', 
               'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']

# Load PyTorch Image Model
def load_image_model():
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    try:
        model.load_state_dict(torch.load('best_chest_ct_model.pth', map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Warning: Could not load image model. {e}")
        return None

image_model = load_image_model()

# Image Preprocessing
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Clinical XGBoost Model & Preprocessors
try:
    clinical_model = joblib.load('clinical_rf_model.joblib')
    clinical_scaler = joblib.load('clinical_scaler.joblib')
    clinical_encoders = joblib.load('clinical_label_encoders.joblib')
except Exception as e:
    print(f"Warning: Could not load clinical model files. {e}")
    clinical_model, clinical_scaler, clinical_encoders = None, None, None

# Load LLM Configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    llm_client = genai.Client(api_key=API_KEY)
else:
    llm_client = None


# --- 2. Inference Functions ---

def analyze_patient(image, age, sex, bmi, sys_bp, dia_bp, glucose, chol, creat, diabetes, hyper, user_api_key):
    
    # Dynamically setup the LLM Client
    current_llm_client = llm_client
    if user_api_key and user_api_key.strip():
        try:
            current_llm_client = genai.Client(api_key=user_api_key.strip())
        except Exception as e:
            print(f"Failed to initialize client with provided key: {e}")
            
    # --- A. Image Inference & Grad-CAM ---
    image_result_text = "No image provided or model missing."
    heatmap_img = None
    
    if image is not None and image_model is not None:
        img_tensor = image_transforms(image).unsqueeze(0).to(DEVICE)
        
        # 1. Prediction
        with torch.no_grad():
            outputs = image_model(img_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[preds[0]]
            clean_class = predicted_class.replace('_', ' ').title()
            image_result_text = f"Diagnosis: **{clean_class}**"
            
        # 2. Grad-CAM Heatmap
        try:
            target_layers = [image_model.layer4[-1]]
            cam = GradCAM(model=image_model, target_layers=target_layers)
            
            # GradCAM requires gradient computation, so no torch.no_grad() here
            grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
            
            # Convert original PIL image to RGB float [0, 1] array for the overlay
            rgb_img = np.array(image.convert('RGB').resize((224, 224))) / 255.0
            heatmap_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
        except Exception as e:
            print(f"Grad-CAM Error: {e}")
            heatmap_img = np.array(image.convert('RGB').resize((224, 224)))

    
    # --- B. Clinical Inference ---
    clinical_result_text = "Clinical model not loaded."
    if clinical_model is not None:
        try:
            sex_encoded = clinical_encoders['sex'].transform([sex])[0]
            features = pd.DataFrame([{
                'age': age,
                'sex': sex_encoded,
                'bmi': bmi,
                'systolic_bp': sys_bp,
                'diastolic_bp': dia_bp,
                'glucose': glucose,
                'cholesterol': chol,
                'creatinine': creat,
                'diabetes': 1 if diabetes == 'Yes' else 0,
                'hypertension': 1 if hyper == 'Yes' else 0,
                'diagnosis': 0, # Dummy value
                'readmission_30d': 0 # Dummy value
            }])
            scaled_features = clinical_scaler.transform(features)
            risk_pred = clinical_model.predict(scaled_features)[0]
            clinical_result_text = "High Risk" if risk_pred == 1 else "Low Risk"
        except Exception as e:
            clinical_result_text = f"Error computing risk: {e}"

    # --- C. LLM Report Generation ---
    if current_llm_client:
        prompt = f"""
        You are an expert AI clinical assistant. Please write a comprehensive medical report for the following patient:
        
        Patient Vitals & Labs:
        - Age: {age}
        - Sex: {sex}
        - BMI: {bmi}
        - Blood Pressure: {sys_bp}/{dia_bp}
        - Glucose: {glucose}
        - Cholesterol: {chol}
        
        AI Image Analysis Result (Chest CT): {image_result_text}
        AI Clinical Risk Model Result: {clinical_result_text}
        
        Format your response with:
        1. Patient Summary
        2. Integrated Risk Assessment
        3. Recommendations for Follow-up
        """
        try:
            response = current_llm_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            llm_response = response.text
        except Exception as e:
            llm_response = f"LLM Error: {e}\n\nMake sure your API key is valid!"
    else:
        llm_response = f"""
        [Mock LLM Report]
        The patient is a {age}-year-old {sex}. 
        Image finding: {image_result_text}. 
        Clinical Risk: {clinical_result_text}. 
        
        (Please paste your GEMINI API KEY in the settings box to generate a full LLM report).
        """

    # --- D. PDF Generation ---
    pdf_path = None
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Helvetica", size=16, style='B')
        pdf.cell(0, 10, txt="Multimodal AI Medical Report", ln=True, align='C')
        pdf.ln(5)
        
        # Basic Vitals & Results
        pdf.set_font("Helvetica", size=12)
        vitals_txt = f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\nPatient: {age}yo {sex} | BMI: {bmi}\nBP: {sys_bp}/{dia_bp} | Glucose: {glucose}\n\n"
        results_txt = f"{image_result_text.replace('**', '')}\nClinical ML Risk Prediction: {clinical_result_text}\n"
        
        # write strings individually because fpdf multi_cell expects simple strings
        # We need to manually convert unicode if there are issues, but standard ASCII is fine.
        safe_vitals = vitals_txt.encode('latin-1', 'replace').decode('latin-1')
        safe_results = results_txt.encode('latin-1', 'replace').decode('latin-1')
        
        pdf.multi_cell(0, 8, txt=safe_vitals)
        pdf.multi_cell(0, 8, txt=safe_results)
        pdf.ln(5)
        
        # Images
        if image is not None and heatmap_img is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig:
                image.convert('RGB').resize((224,224)).save(tmp_orig.name)
                orig_path = tmp_orig.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_heat:
                Image.fromarray(heatmap_img).save(tmp_heat.name)
                heat_path = tmp_heat.name
                
            pdf.image(orig_path, x=20, w=70)
            pdf.image(heat_path, x=110, w=70)
            pdf.ln(80) # Move cursor down past the images
            
        # LLM Text
        pdf.set_font("Helvetica", size=10)
        safe_llm = llm_response.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, txt=safe_llm)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            pdf_path = tmp_pdf.name
            
    except Exception as e:
        print(f"PDF Error: {e}")

    return image_result_text, heatmap_img, clinical_result_text, llm_response, pdf_path

# --- 3. Gradio Interface Layout ---

with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Multimodal Chest CT & Clinical AI Assistant")
    gr.Markdown("Upload a Chest CT scan and input patient vitals to receive a unified AI analysis.")
    
    with gr.Row():
        gr.Markdown("### ⚙️ Settings")
        api_key_in = gr.Textbox(label="Gemini API Key (Optional if set in environment)", type="password", placeholder="Paste your API key here (AIza...)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload CT Scan")
            img_input = gr.Image(type="pil", label="Chest CT Scan")
            
            gr.Markdown("### 2. Patient Clinical Data")
            age_in = gr.Slider(0, 100, value=50, step=1, label="Age")
            sex_in = gr.Radio(["Male", "Female"], value="Male", label="Sex")
            bmi_in = gr.Number(value=25.0, label="BMI")
            sys_bp_in = gr.Number(value=120, label="Systolic BP")
            dia_bp_in = gr.Number(value=80, label="Diastolic BP")
            glucose_in = gr.Number(value=100, label="Glucose")
            chol_in = gr.Number(value=200, label="Cholesterol")
            creat_in = gr.Number(value=1.0, label="Creatinine")
            diab_in = gr.Radio(["Yes", "No"], value="No", label="Diabetes")
            hyper_in = gr.Radio(["Yes", "No"], value="No", label="Hypertension")
            
            analyze_btn = gr.Button("Generate AI Report & Heatmap", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### Analysis Results")
            img_res = gr.Textbox(label="CT Image Model Prediction", lines=2)
            heat_res = gr.Image(label="Grad-CAM Tumor Heatmap")
            clin_res = gr.Textbox(label="Clinical ML Risk Prediction", lines=2)
            
            gr.Markdown("### Comprehensive LLM Report")
            llm_res = gr.Textbox(label="AI Doctor's Assistant Report", lines=15)
            
            gr.Markdown("### Download Output")
            pdf_res = gr.File(label="Download Medical Report (PDF)")
            
    analyze_btn.click(
        fn=analyze_patient,
        inputs=[img_input, age_in, sex_in, bmi_in, sys_bp_in, dia_bp_in, glucose_in, chol_in, creat_in, diab_in, hyper_in, api_key_in],
        outputs=[img_res, heat_res, clin_res, llm_res, pdf_res]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", theme=gr.themes.Soft())
