import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# --- LangChain & RAG Imports ---
langchain_available = True
try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
except ModuleNotFoundError:
    langchain_available = False
    FAISS = OpenAIEmbeddings = ChatOpenAI = RetrievalQA = Document = None

# Set page configuration - force sidebar collapsed by default
st.set_page_config(
    page_title="AI-Driven Smart Fuel Monitoring Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# HIGH-TECH CYBERPUNK TELEMETRY & NITRO FUEL MONITORING THEME
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syncopate:wght@400;700&display=swap');

    /* Hide default sidebar navigation elements */
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}

    /* Background: Deep Titanium Charcoal blending to True Carbon Black */
    .stApp {
        background: radial-gradient(circle at center, #111823 0%, #0b0f19 50%, #030508 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Headlines: Syncopate Font with Nitro Cyan Glow Accent */
    h1, h2, h3 {
        font-family: 'Syncopate', sans-serif !important;
        color: #00F0FF !important; /* Cyberpunk Cyan / Nitro Fuel */
        text-shadow: 0 0 20px rgba(0, 240, 255, 0.5), 
                     0 0 5px rgba(255, 255, 255, 0.3);
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Metric Cards: Transparent "Smoked Obsidian Glass" with Neon Teal Micro-borders */
    [data-testid="stMetric"] {
        background: rgba(11, 20, 32, 0.6) !important;
        border: 1px solid rgba(0, 240, 255, 0.4) !important; /* Nitro Teal Tint Border */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.1);
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border: 1px solid rgba(0, 240, 255, 0.8) !important;
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.25);
    }
    
    /* Metric Labels: Matte Silver / Tactical Gray */
    [data-testid="stMetricLabel"] {
        font-family: 'Syncopate', sans-serif !important;
        color: #9CA3AF !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.75rem !important;
    }
    
    /* Metric Values: Pure High-Contrast White with Digital Matrix Glow */
    [data-testid="stMetricValue"] {
        font-family: 'Share Tech Mono', monospace !important;
        color: #FFFFFF !important;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.6);
        font-size: 2.2rem !important;
    }

    /* Sidebar: Solid Dark Carbon with Matte Ingestion Border Line */
    [data-testid="stSidebar"] {
        background-color: #05080e !important;
        border-right: 1px solid rgba(0, 240, 255, 0.2);
    }

    /* Buttons: Kinetic Cyan Fill to High-contrast Text */
    .stButton>button {
        background-color: #00F0FF !important;
        color: #030508 !important; /* Dark text on bright button */
        font-family: 'Syncopate', sans-serif !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        border: 1px solid #FFFFFF !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
        transition: 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.7);
        transform: translateY(-2px);
        background-color: #FFFFFF !important;
        color: #010409 !important;
    }

    /* Custom Primary vs Secondary handling for safety */
    .stButton>button[kind="secondary"] {
        background-color: transparent !important;
        color: #00F0FF !important;
        border: 1px solid #00F0FF !important;
    }

    /* Slider Track: High-Octane Linear Influx (Teal to Warning Red Gradient) */
    div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #00F0FF 0%, #FF2E93 100%) !important;
    }
    div[role="slider"] {
        background-color: #00F0FF !important;
        box-shadow: 0 0 15px #00F0FF !important;
        border: 2px solid #FFFFFF !important;
    }

    /* Dataframe/Table: Deep Fleet Obsidian Grid matrix */
    .stDataFrame {
        background-color: rgba(7, 11, 19, 0.8);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 8px;
    }

    /* Footer Text Dashboard Signoff */
    .footer-text {
        text-align: center;
        color: #6B7280;
        padding: 25px;
        font-family: 'Syncopate', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 2px;
    }

    /* Ingestion Alert / St.info Box overrides */
    [data-testid="stAlert"] {
        background-color: rgba(9, 16, 26, 0.8) !important;
        color: #00F0FF !important;
        border: 1px solid rgba(0, 240, 255, 0.4) !important;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
        font-family: 'Share Tech Mono', monospace !important;
    }

    [data-testid="stAlert"] strong {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    [data-testid="stAlert"] div {
        color: #00F0FF !important;
    }
    
    [data-testid="stAlert"] svg {
        fill: #00F0FF !important;
    }

    /* Dynamic Captions / Metadata Frame */
    div[data-testid="stCaptionContainer"] {
        background-color: rgba(5, 8, 14, 0.6);
        padding: 8px;
        border-radius: 6px;
        border: 1px solid rgba(0, 240, 255, 0.2);
        margin-top: -10px;
    }
    
    div[data-testid="stCaptionContainer"] p {
        font-family: 'Share Tech Mono', monospace !important;
        color: #9CA3AF !important;
        letter-spacing: 1px;
    }

    /* Folium Map Matrix Skinning (Forces dark-grid aesthetics matching transport systems) */
    .folium-map {
        filter: invert(1) hue-rotate(180deg) brightness(0.9) contrast(1.1) !important;
        border: 1px solid #00F0FF !important;
        border-radius: 12px;
    }

    div[data-testid="stFolium"] {
        background-color: #070b13 !important;
        padding: 8px;
        border-radius: 12px;
    }

    /* Aligning logo groups nicely across centers */
    [data-testid="stHorizontalBlock"] div[style*="flex-direction: column"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .logo-cluster-container img {
        margin-top: auto;
        margin-bottom: auto;
        max-height: 110px;
        object-fit: contain;
    }

    /* --- SIDEBAR ACADEMIC & INDUSTRY MENTOR STYLING --- */
    .sidebar-mentor-container {
        padding: 4px;
        border-left: 2px solid #00F0FF; /* Bright cyan tracking indicator */
        padding-left: 12px;
        margin-bottom: 12px;
    }

    .s-label {
        font-family: 'Syncopate', sans-serif;
        color: #6B7280;
        font-size: 0.65rem !important;
        text-transform: uppercase;
        margin-bottom: 3px !important;
        letter-spacing: 1.5px;
    }

    .s-name {
        font-family: 'Share Tech Mono', monospace;
        color: #FFFFFF;
        font-size: 0.85rem !important;
        font-weight: bold;
        line-height: 1.2;
        margin-bottom: 0px !important;
    }

    /* Sidebar Navigation Links and Radio Inputs alignment */
    div[data-testid="stSidebarNav"] {
        padding-top: 0rem !important;
    }

    div[role="radiogroup"] label {
        color: #9CA3AF !important; 
        font-family: 'Syncopate', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 1px;
        transition: color 0.2s ease;
    }
    
    div[role="radiogroup"] label:hover {
        color: #00F0FF !important;
    }
    
    /* Ensure markdown body links or paragraphs stay safe against black backgrounds */
    .stMarkdown p, .stMarkdown span, label {
        color: #E5E7EB !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INITIALIZE SESSION STATE FOR NAVIGATION & CHAT
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def navigate_to(page_name):
    st.session_state.current_page = page_name
    st.rerun()

# -----------------------------------------------------------------------------
# 0. MOCK DATA GENERATOR & CACHING FOR SMOOTH USER EXPERIENCE
# -----------------------------------------------------------------------------
@st.cache_data
def generate_mock_telemetry_data():
    """Generates synthetic IoT telemetry data matching the project's exact specs."""
    np.random.seed(42)
    vehicles = [f"VEH{i:04d}" for i in range(10)]
    brands = ["Volvo", "Scania", "Mercedes-Benz", "Isuzu", "Hino"]
    
    base_time = datetime.datetime(2026, 6, 1, 0, 0)
    data_records = []
    
    for vehicle_id in vehicles:
        brand = np.random.choice(brands)
        current_fuel = 100.0
        
        # Generate 200 chronological data points per vehicle
        for t in range(200):
            timestamp = base_time + datetime.timedelta(minutes=t * 15)
            
            engine_rpm = float(np.random.normal(1500, 400))
            engine_rpm = max(0.0, min(8000.0, engine_rpm))
            
            engine_load = float(np.random.uniform(20, 85))
            throttle_pos = float(engine_load * np.random.uniform(0.9, 1.1))
            throttle_pos = max(0.0, min(100.0, throttle_pos))
            
            speed = float((engine_rpm / 1500) * 45 + np.random.normal(0, 5))
            speed = max(0.0, min(120.0, speed))
            
            air_flow = float((engine_rpm * 0.05) + (engine_load * 0.2) + np.random.normal(0, 2))
            temp = float(85.0 + np.random.normal(0, 3))
            
            fuel_consumption_lph = float((engine_rpm * 0.003) + (engine_load * 0.1) + (speed * 0.05))
            
            current_fuel -= (fuel_consumption_lph * 0.25) / 50
            if current_fuel < 5:
                current_fuel = 98.5
                
            data_records.append({
                "vehicle_id": vehicle_id,
                "brand": brand,
                "timestamp": timestamp,
                "engine_rpm": engine_rpm,
                "engine_load_percent": engine_load,
                "throttle_pos_percent": throttle_pos,
                "vehicle_speed_kph": speed,
                "air_flow_rate_gps": air_flow,
                "engine_temp_c": temp,
                "fuel_level_percent": max(0.0, current_fuel),
                "fuel_consumption_lph": fuel_consumption_lph
            })
            
    df = pd.DataFrame(data_records)
    raw_df = df.copy()
    raw_df.loc[10, 'fuel_level_percent'] = 999.0
    raw_df.loc[25, 'engine_rpm'] = -150
    return raw_df, df

raw_data, clean_data = generate_mock_telemetry_data()
features = ['engine_rpm', 'engine_load_percent', 'throttle_pos_percent', 'vehicle_speed_kph', 'air_flow_rate_gps', 'engine_temp_c']
target = 'fuel_consumption_lph'

# -----------------------------------------------------------------------------
# HIGH-LEVEL FEATURE: LANGCHAIN RAG INJECTION
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_rag_engine():
    """Compiles operational logistics guidelines and turns them into vector embeddings."""
    if not langchain_available:
        return None, False

    api_key = "PLACEHOLDER_KEY"
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "PLACEHOLDER_KEY")
    except Exception:
        api_key = "PLACEHOLDER_KEY"

    knowledge_base =
