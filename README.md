# Biomedical Image Analysis & Clinical Decision Support System

This repository contains a comprehensive suite of AI-powered tools for biomedical data analysis, chest CT scan classification, clinical risk prediction, and automated medical report generation.

## 🚀 Key Features

1. **Multimodal AI Medical Assistant (`app.py`)**:
   - **Gradio Web Interface**: User-friendly UI for uploading scans and entering patient data.
   - **ResNet50 Chest CT Classifier**: Deep learning model that classifies chest CT images into 4 distinct categories (Normal, Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma).
   - **Grad-CAM Explanations**: Visual activation heatmaps generated using PyTorch Grad-CAM to highlight suspicious features and regions of interest on the CT scans.
   - **XGBoost Clinical Risk Predictor**: Machine learning model analyzing 10 patient features (Age, Sex, BMI, Blood Pressure, Glucose, Cholesterol, Creatinine, Diabetes, Hypertension) to predict clinical risk.
   - **Automated Medical Reports**: Integrates predictions with the Google Gemini API (`gemini-2.5-flash`) to generate structured summaries, risk assessments, and recommendations.
   - **Downloadable PDF Reports**: Auto-generates a PDF report containing patient vitals, classification details, original CT scan, Grad-CAM activation heatmap, and Gemini's analysis.

2. **BioInsight AI Cohort Analytics (`main.py`)**:
   - **Streamlit Web Application**: An interactive dashboard for clinical registry/dataset cohort analysis.
   - **Natural Language Cohort Querying**: Users upload CSV or Excel registry datasets and query them using plain English.
   - **On-the-fly Data Visualization**: Automatically writes, compiles, and runs Python data visualization code (using matplotlib & seaborn) based on user queries, generating custom plots and insights dynamically.

3. **Machine Learning Pipelines**:
   - **Image Classifier Training (`train_image_model.py`)**: Training script for ResNet50 on chest CT scans.
   - **Clinical Model Training (`train_clinical_model.py`)**: XGBoost classifier training pipeline leveraging GPU acceleration (`cuda` tree method).

---

## 🛠️ Project Structure
- `app.py`: The Gradio-based multimodal assistant app.
- `main.py` & `main1.py`: Streamlit-based interactive cohort analytics apps.
- `train_image_model.py`: Script to train the chest CT classification ResNet50 model.
- `train_clinical_model.py`: Script to train the clinical risk XGBoost model.
- `llm_clinical_analysis.py`: Standalone script for LLM-based clinical report experiments.
- `best_chest_ct_model.pth`: Trained ResNet50 PyTorch weights.
- `clinical_rf_model.joblib`: Trained XGBoost classifier.
- `clinical_scaler.joblib`: StandardScaler for clinical inputs.
- `clinical_label_encoders.joblib`: LabelEncoder objects for categorical clinical features.

---

## ⚙️ Installation & Setup

1. **Set Up a Virtual Environment** (Recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy pandas joblib fpdf2 streamlit matplotlib seaborn xgboost scikit-learn gradio google-genai pytorch-grad-cam pillow
   ```
   *(Note: Adjust pytorch command if you need specific CUDA versions)*

3. **API Key Setup**:
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## 🏃 Running the Apps

### Launch the Multimodal Assistant (Gradio Web UI)
This application handles patient-level diagnostics:
```bash
python app.py
```
Open the local Gradio link (usually `http://127.0.0.1:7860`) in your browser.

### Launch BioInsight AI (Streamlit Cohort Analytics)
This application handles registry/cohort-level analysis:
```bash
streamlit run main.py
```
Open the local Streamlit link (usually `http://localhost:8501`) in your browser.

---

## 🔍 Model Information

### Chest CT Image Classifier (ResNet50)
- **Target classes**:
  1. `adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`
  2. `large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa`
  3. `normal`
  4. `squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa`

### Clinical Risk Model (XGBoost)
- **Input Features**: Age, Sex, BMI, Systolic BP, Diastolic BP, Glucose, Cholesterol, Creatinine, Diabetes, Hypertension.
- **Target**: Patient Mortality / Readmission Risk prediction.

---
_Disclaimer: This project is intended for research and educational purposes only. Always consult a qualified medical professional for clinical decisions._
