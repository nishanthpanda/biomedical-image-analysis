"""
Biomedical Data Analysis App (Production UX/UI Edition)
Built with Google GenAI SDK + Streamlit Premium UI
Author: Nishanth Panda
"""

import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, json, textwrap, os

# ─── 1. PAGE CONFIGURATION (MUST BE ABSOLUTE FIRST) ──────────────────────────
st.set_page_config(
    page_title="BioInsight AI | Clinical Analytics", 
    layout="wide", 
    page_icon="🧬"
)

# ─── 2. DIRECT API KEY CONFIGURATION ──────────────────────────────────────────
# Replace the string below with your actual Gemini API key from Google AI Studio
GEMINI_API_KEY = ""
GROQ_API_KEY = ""

# ─── 3. CUSTOM CSS THEMING ────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #ecf0f1; }
    .stHeading h1 { font-family: 'Inter', sans-serif; font-weight: 800; color: #00ffd1; }
    .stCaption { font-family: 'Inter', sans-serif; color: #8a9ba8; font-size: 14px; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00ffd1; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #a3b8cc; font-size: 12px; font-weight: 500; text-transform: uppercase; }
    .report-card { 
        background-color: #1a1f2c; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #2e374a;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Top Brand Header ─────────────────────────────────────────────────────────
st.title("🧬 BioInsight AI")
st.caption("Advanced Clinical Data Science & Natural Language Analytics Engine")
st.markdown("---")

# ─── Conversation state ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "df" not in st.session_state:
    st.session_state.df = None

# ─── Core Analytical Mechanics ────────────────────────────────────────────────
def build_system(df_info: str) -> str:
    return f"""You are an expert biomedical data scientist. The user has uploaded a clinical dataset:

{df_info}

When asked a question, respond ONLY with a JSON object:
{{
  "answer": "Provide a comprehensive, clinical, and structured summary explanation of the data results.",
  "python_code": "Valid Python code using the variable `df`. Use matplotlib/seaborn. CRITICAL: You must save the final image to exactly 'plot.png' using plt.savefig('plot.png', bbox_inches='tight'). Do not use any other filename. Do not call plt.show().",
  "plot": true or false
}}
Output ONLY the JSON."""

def get_df_info(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    desc = df.describe(include="all").to_string()
    sample = df.head(5).to_string() 
    return f"Columns: {list(df.columns)}\n\nInfo:\n{buf.getvalue()}\n\nDescribe:\n{desc}\n\nSample rows:\n{sample}"

def ask_gemini(question: str, df: pd.DataFrame, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)
    df_info = get_df_info(df)
    config = types.GenerateContentConfig(
        system_instruction=build_system(df_info),
        response_mime_type="application/json",
        temperature=0.2 
    )
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite", 
        contents=question,
        config=config
    )
    return json.loads(response.text)

def ask_groq(question: str, df: pd.DataFrame, api_key: str, model_name: str) -> dict:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Groq library is not installed. Please run `pip install groq`")
        
    client = Groq(api_key=api_key)
    df_info = get_df_info(df)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": build_system(df_info)},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def run_code(code: str, df: pd.DataFrame):
    local_ns = {"df": df, "pd": pd, "plt": plt, "sns": sns}
    plt.clf() 
    exec(textwrap.dedent(code), local_ns)

# ─── Sidebar – Interactive Pipeline Settings ──────────────────────────────────
with st.sidebar:
    st.header("🤖 AI Provider Selection")
    
    provider_tabs = st.tabs(["Gemini", "Groq"])
    
    with provider_tabs[0]:
        gemini_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password", key="gemini_key")
        if not gemini_key or gemini_key == "YOUR_GOOGLE_API_KEY_HERE":
            st.warning("Please enter a valid Gemini API Key.")
            
    with provider_tabs[1]:
        groq_key = st.text_input("Groq API Key", value=GROQ_API_KEY, type="password", key="groq_key")
        groq_model = st.selectbox("Groq Model", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"], key="groq_model")
        if not groq_key:
            st.warning("Please enter a valid Groq API Key.")
            
    active_provider = st.radio("Select Active Provider", ["Gemini", "Groq"])

    st.markdown("---")
    st.header("📥 Data Ingestion")
    file = st.file_uploader("Upload Clinical Registry (CSV or Excel)", type=["csv", "xlsx"])
    
    if file:
        try:
            if file.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(file)
            else:
                st.session_state.df = pd.read_excel(file)
            st.success("✅ Dataset Verified Successfully")
        except Exception as e:
            st.error(f"Ingestion Error: {e}")

    st.markdown("---")
    st.header("⚙️ Workspace Controls")
    if st.button("🗑️ Reset Application", use_container_width=True):
        st.session_state.history = []
        st.session_state.df = None
        if os.path.exists("plot.png"):
            os.remove("plot.png")
        st.rerun()

# ─── Main Display Layout ──────────────────────────────────────────────────────
if st.session_state.df is None:
    st.info("💡 To begin analysis, drop a clinical spreadsheet or registry into the sidebar data ingestion portal.")
else:
    df = st.session_state.df

    st.markdown("### 📊 Dataset Cohort Executive Summary")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.metric(label="Total Cohort Sample (N)", value=f"{df.shape[0]:,}")
    with m_col2:
        st.metric(label="Variables Tracked", value=f"{df.shape[1]}")
    with m_col3:
        missing_cells = df.isnull().sum().sum()
        st.metric(label="Missing Data Cells", value=f"{missing_cells:,}", delta="- Complete" if missing_cells==0 else "Action Required", delta_color="inverse")
    with m_col4:
        age_col = [c for c in df.columns if 'age' in c.lower()]
        if age_col:
            st.metric(label="Mean Patient Age", value=f"{df[age_col[0]].mean():.1f} yrs")
        else:
            st.metric(label="File Size", value=f"{file.size / 1024:.1f} KB")

    st.markdown("---")

    workspace_left, workspace_right = st.columns([1, 1], gap="large")

    with workspace_left:
        st.markdown("### 💬 Clinical Analytics Chat")
        
        for msg in st.session_state.history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])

        question = st.chat_input("Query your cohort dataset (e.g., 'What are the main predictors of patient mortality?')")

    if question:
        with workspace_left:
            with st.chat_message("user"):
                st.write(question)
        st.session_state.history.append({"role": "user", "content": question})
        
        with workspace_right:
            st.markdown("### 📈 Real-Time AI Generation Engine")
            with st.status("Initializing Gemini Inference Pipeline...", expanded=True) as status:
                try:
                    if active_provider == "Gemini":
                        if not st.session_state.gemini_key or st.session_state.gemini_key == "YOUR_GOOGLE_API_KEY_HERE":
                            raise ValueError("Gemini API key is missing or invalid.")
                        result = ask_gemini(question, df, st.session_state.gemini_key)
                    else:
                        if not st.session_state.groq_key:
                            raise ValueError("Groq API key is missing.")
                        result = ask_groq(question, df, st.session_state.groq_key, st.session_state.groq_model)
                        
                    answer = result.get("answer", "")
                    code   = result.get("python_code", "")
                    has_plot = result.get("plot", False)

                    if code:
                        status.update(label="Executing Automated Data Analysis Code...", state="running")
                        run_code(code, df.copy())

                    status.update(label="Analysis Finished!", state="complete", expanded=False)

                    st.markdown(f"""<div class="report-card">
                        <h4 style="color:#00ffd1;margin-top:0;">📋 Analytical Interpretation</h4>
                        <p style="font-size:15px;line-height:1.6;">{answer}</p>
                    </div>""", unsafe_allow_html=True)

                    if has_plot and os.path.exists("plot.png"):
                        st.image("plot.png", caption="AI-Generated Visual Insight Matrix", use_container_width=True)

                    if code:
                        with st.expander("Examine Server-Side Generation Logic (Python)"):
                            st.code(code, language="python")
                    
                    st.session_state.history.append({
                        "role": "assistant", 
                        "content": answer
                    })

                except Exception as e:
                    status.update(label="Execution Failed", state="error")
                    st.error(f"Inference Failure: {e}")