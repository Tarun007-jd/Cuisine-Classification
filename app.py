import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(
    page_title="Cuisine Classification",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Enhanced Custom CSS
# ---------------------------------
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(180deg, #fafafa 0%, #f0f2f6 100%);
    }
    
    /* ===== HEADER STYLING ===== */
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #FEC89A 100%);
        border-radius: 24px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 15px 40px rgba(255, 107, 107, 0.35);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: shimmer 3s infinite linear;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.15);
        font-weight: 800;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.25rem;
        margin: 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* ===== TAB STYLING - FIXED VISIBILITY ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #ffffff;
        padding: 0.75rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.85rem 1.75rem;
        font-weight: 600;
        font-size: 0.95rem;
        color: #2d3436 !important;
        background: #f8f9fa;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #fff0f0;
        color: #FF6B6B !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }
    
    /* Tab text color fix */
    .stTabs button[role="tab"] p {
        color: #2d3436 !important;
        font-weight: 600;
    }
    
    .stTabs button[role="tab"][aria-selected="true"] p {
        color: white !important;
    }
    
    /* ===== CARD STYLING ===== */
    .info-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fc 100%);
        border-radius: 20px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 107, 107, 0.08);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
        border-color: rgba(255, 107, 107, 0.2);
    }
    
    .card-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a1a2e !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #fafbfc 100%);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        border-top: 5px solid #FF6B6B;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 50%, #FEC89A 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 50px rgba(255, 107, 107, 0.2);
    }
    
    .metric-value {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    
    .metric-label {
        color: #636e72 !important;
        font-size: 1rem;
        margin-top: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 4px solid transparent;
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(90deg, #FF6B6B, #FF8E53, #FEC89A) border-box;
        border-radius: 0 0 4px 4px;
    }
    
    .section-header h2 {
        color: #1a1a2e !important;
        font-size: 1.6rem;
        margin: 0;
        font-weight: 700;
    }
    
    /* ===== SUCCESS BADGE ===== */
    .success-badge {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white !important;
        padding: 0.85rem 1.75rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        box-shadow: 0 6px 20px rgba(0, 184, 148, 0.35);
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .success-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 184, 148, 0.45);
    }
    
    /* ===== PREDICTION RESULT ===== */
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.35);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-result::before {
        content: '';
        position: absolute;
        top: -100%;
        left: -100%;
        width: 300%;
        height: 300%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 50%);
    }
    
    .prediction-result h3 {
        margin: 0 0 1rem 0;
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 500;
        color: white !important;
        position: relative;
        z-index: 1;
    }
    
    .prediction-result .cuisine-name {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        color: white !important;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* ===== FEATURE IMPORTANCE ===== */
    .feature-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.25rem;
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 14px;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
        border: 1px solid #eee;
    }
    
    .feature-item:hover {
        background: linear-gradient(145deg, #fff5f5 0%, #ffffff 100%);
        transform: translateX(8px);
        border-color: rgba(255, 107, 107, 0.3);
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.1);
    }
    
    .feature-name {
        font-weight: 700;
        color: #2d3436 !important;
        font-size: 0.95rem;
    }
    
    .feature-score {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        box-shadow: 0 3px 10px rgba(255, 107, 107, 0.3);
    }
    
    /* ===== FORM STYLING ===== */
    .stForm {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fc 100%);
        padding: 2.5rem;
        border-radius: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 107, 107, 0.1);
    }
    
    /* ===== BUTTON STYLING ===== */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.9rem 2.5rem;
        border-radius: 50px;
        font-size: 1.15rem;
        font-weight: 700;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        text-transform: none;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) translateY(-2px);
        box-shadow: 0 12px 35px rgba(255, 107, 107, 0.5);
    }
    
    .stButton > button:active {
        transform: scale(0.98);
    }
    
    /* ===== DATAFRAME STYLING ===== */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border-radius: 16px;
        border: 1px solid #e0e0e0;
    }
    
    /* ===== REPORT CONTAINER ===== */
    .report-container {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        color: #dfe6e9 !important;
        padding: 2rem;
        border-radius: 18px;
        font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
        box-shadow: 0 10px 30px rgba(26, 26, 46, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .report-container pre {
        color: #dfe6e9 !important;
        margin: 0;
        white-space: pre-wrap;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 2.5rem;
        margin-top: 4rem;
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fc 100%);
        border-radius: 24px;
        color: #636e72 !important;
        box-shadow: 0 -5px 30px rgba(0, 0, 0, 0.05);
        border: 1px solid #eee;
    }
    
    .footer p {
        margin: 0;
        font-size: 1.05rem;
        color: #636e72 !important;
    }
    
    .footer .highlight {
        color: #FF6B6B !important;
        font-weight: 700;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff8f8 0%, #ffffff 100%);
        border-right: 1px solid rgba(255, 107, 107, 0.1);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1a1a2e !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #2d3436 !important;
    }
    
    [data-testid="stSidebar"] .stSlider label {
        color: #2d3436 !important;
    }
    
    /* ===== SLIDER STYLING ===== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%) !important;
    }
    
    .stSlider > div > div > div > div {
        background: white !important;
        border: 3px solid #FF6B6B !important;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3);
    }
    
    /* ===== INPUT STYLING ===== */
    .stNumberInput input {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus {
        border-color: #FF6B6B !important;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.1) !important;
    }
    
    .stNumberInput label {
        color: #2d3436 !important;
        font-weight: 600;
    }
    
    /* ===== SELECT BOX ===== */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    .stSelectbox label {
        color: #2d3436 !important;
        font-weight: 600;
    }
    
    /* ===== INFO BOXES ===== */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }
    
    div[data-testid="stAlert"] {
        background: linear-gradient(145deg, #e8f4fd 0%, #d4e9f7 100%) !important;
        border-left: 4px solid #3498db !important;
        border-radius: 12px !important;
    }
    
    /* ===== MARKDOWN TEXT ===== */
    .stMarkdown p {
        color: #2d3436 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a1a2e !important;
    }
    
    .stMarkdown strong {
        color: #1a1a2e !important;
    }
    
    /* ===== TABLE STYLING ===== */
    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .stMarkdown table th {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white !important;
        padding: 1rem;
        font-weight: 700;
        text-align: left;
    }
    
    .stMarkdown table td {
        padding: 0.85rem 1rem;
        border-bottom: 1px solid #eee;
        color: #2d3436 !important;
    }
    
    .stMarkdown table tr:hover td {
        background: #fff5f5;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2d3436 !important;
        border-radius: 12px;
    }
    
    /* ===== METRIC WIDGET ===== */
    [data-testid="stMetricValue"] {
        color: #FF6B6B !important;
        font-weight: 800;
    }
    
    [data-testid="stMetricLabel"] {
        color: #636e72 !important;
    }
    
    /* ===== HIDE DEFAULTS ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FF6B6B 0%, #FF8E53 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FF5252 0%, #FF7043 100%);
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .info-card, .metric-card {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Header Section
# ---------------------------------
st.markdown("""
<div class="main-header">
    <h1>🍽️ Cuisine Classification System</h1>
    <p>Predict restaurant cuisines using Machine Learning with Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset.csv")
    return df

df = load_data()

# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.markdown("## 🎛️ Model Settings")
    st.markdown("---")
    
    n_estimators = st.slider(
        "🌲 Number of Trees",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Number of trees in the Random Forest"
    )
    
    test_size = st.slider(
        "📊 Test Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Proportion of dataset for testing"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Quick Stats")
    st.info(f"📁 **Dataset Size:** {len(df):,} rows")
    st.info(f"📊 **Features:** {df.shape[1]} columns")

# ---------------------------------
# Main Content Tabs
# ---------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview", 
    "🧠 Model Training", 
    "📈 Performance", 
    "🔮 Predict"
])

# ---------------------------------
# Data Preprocessing (happens in background)
# ---------------------------------
df_processed = df.copy()
df_processed.fillna("Unknown", inplace=True)

binary_cols = ["Has Online delivery", "Has Table booking"]
for col in binary_cols:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].map({"Yes": 1, "No": 0})

target_column = "Cuisines"
le = LabelEncoder()
df_processed[target_column] = le.fit_transform(df_processed[target_column])

features = []
possible_features = [
    "Average Cost for two",
    "Price range",
    "Has Online delivery",
    "Has Table booking",
    "Votes"
]

for col in possible_features:
    if col in df_processed.columns:
        features.append(col)

X = df_processed[features]
y = df_processed[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# ---------------------------------
# Tab 1: Data Overview
# ---------------------------------
with tab1:
    st.markdown("""
    <div class="section-header">
        <h2>📊 Dataset Preview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Features Used</div>
        </div>
        """.format(len(features)), unsafe_allow_html=True)
    
    with col3:
        unique_cuisines = df["Cuisines"].nunique() if "Cuisines" in df.columns else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Unique Cuisines</div>
        </div>
        """.format(unique_cuisines), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Columns</div>
        </div>
        """.format(df.shape[1]), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data preview
    st.markdown("""
    <div class="info-card">
        <div class="card-title">📋 Sample Data</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df.head(10), use_container_width=True, height=400)
    
    # Data info columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">📊 Data Types</div>
        </div>
        """, unsafe_allow_html=True)
        
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Type': df.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">🔢 Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
        
        missing_df = pd.DataFrame({
            'Column': df.isnull().sum().index,
            'Missing': df.isnull().sum().values
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

# ---------------------------------
# Tab 2: Model Training
# ---------------------------------
with tab2:
    st.markdown("""
    <div class="section-header">
        <h2>🧠 Model Training Process</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Training steps
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">⚙️ Preprocessing Steps</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ✅ **Step 1:** Handle missing values (filled with "Unknown")
        
        ✅ **Step 2:** Encode binary columns (Yes/No → 1/0)
        
        ✅ **Step 3:** Label encode target variable (Cuisines)
        
        ✅ **Step 4:** Select numerical features for training
        """)
        
        st.markdown("""
        <div class="success-badge">
            ✓ Preprocessing Complete
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">🌲 Model Configuration</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Algorithm | Random Forest |
        | Number of Trees | {n_estimators} |
        | Test Size | {test_size:.0%} |
        | Random State | 42 |
        | Training Samples | {len(X_train):,} |
        | Testing Samples | {len(X_test):,} |
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features used
    st.markdown("""
    <div class="info-card">
        <div class="card-title">📋 Features Used for Training</div>
    </div>
    """, unsafe_allow_html=True)
    
    feature_cols = st.columns(len(features))
    for i, feature in enumerate(features):
        with feature_cols[i]:
            st.info(f"📊 {feature}")

# ---------------------------------
# Model Training (cached)
# ---------------------------------
@st.cache_resource
def train_model(n_est, X_train_data, y_train_data):
    model = RandomForestClassifier(n_estimators=n_est, random_state=42)
    model.fit(X_train_data, y_train_data)
    return model

model = train_model(n_estimators, X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------------------------
# Tab 3: Performance
# ---------------------------------
with tab3:
    st.markdown("""
    <div class="section-header">
        <h2>📈 Model Performance Metrics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Accuracy display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 5px solid #00b894;">
            <div class="metric-value" style="background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {accuracy:.1%}
            </div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two columns for report and feature importance
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">📝 Classification Report</div>
        </div>
        """, unsafe_allow_html=True)
        
        report = classification_report(y_test, y_pred)
        st.markdown(f"""
        <div class="report-container">
            <pre>{report}</pre>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">⭐ Feature Importance</div>
        </div>
        """, unsafe_allow_html=True)
        
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        for _, row in importance_df.iterrows():
            st.markdown(f"""
            <div class="feature-item">
                <span class="feature-name">{row['Feature']}</span>
                <span class="feature-score">{row['Importance']:.3f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.bar_chart(importance_df.set_index("Feature"), color="#FF6B6B")

# ---------------------------------
# Tab 4: Prediction
# ---------------------------------
with tab4:
    st.markdown("""
    <div class="section-header">
        <h2>🔮 Predict Cuisine</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <div class="card-title">🎯 Enter Restaurant Details</div>
        <p style="color: #636e72; margin-top: -0.5rem;">Fill in the details below to predict the cuisine type</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        input_data = {}
        
        col1, col2 = st.columns(2)
        
        for i, col in enumerate(features):
            with col1 if i % 2 == 0 else col2:
                if df_processed[col].dtype in [int, float, np.int64, np.float64]:
                    min_val = float(df_processed[col].min())
                    max_val = float(df_processed[col].max())
                    mean_val = float(df_processed[col].mean())
                    
                    # Custom labels with emojis
                    emoji_map = {
                        "Average Cost for two": "💰",
                        "Price range": "📊",
                        "Has Online delivery": "🚚",
                        "Has Table booking": "📅",
                        "Votes": "👍"
                    }
                    emoji = emoji_map.get(col, "📌")
                    
                    input_data[col] = st.number_input(
                        f"{emoji} {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        help=f"Range: {min_val:.0f} - {max_val:.0f}"
                    )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit = st.form_submit_button("🔮 Predict Cuisine", use_container_width=True)
    
    if submit:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        predicted_cuisine = le.inverse_transform(prediction)[0]
        
        st.markdown(f"""
        <div class="prediction-result">
            <h3>🎉 Prediction Result</h3>
            <p class="cuisine-name">🍽️ {predicted_cuisine}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("""

""", unsafe_allow_html=True)
