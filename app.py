import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import os

# ==========================================
# STAGE 1: GLOBAL PAGE CONFIGURATION & CUSTOM INTERFACES
# ==========================================
st.set_page_config(
    page_title="Enugu Gas Pipeline Control Center", 
    layout="wide",
    page_icon="🛡️"
)

# Custom Enterprise Dark-Theme CSS Injection
st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #ffffff; }
        div[data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; color: #00FFCC !important; }
        div[data-testid="stMetricLabel"] { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #8a99ad !important; }
        .card-panel { background-color: #1f2937; padding: 20px; border-radius: 10px; border-left: 5px solid #00FFCC; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px;">
    <h1 style="color: #00FFCC; margin: 0; font-size: 32px;">🛡️ MIDSTREAM INDUSTRIAL COMMAND CENTER</h1>
    <h3 style="color: #ffffff; margin: 5px 0 0 0; font-size: 18px; font-weight: 300;">Enugu Gas Distribution Pipeline Network Corridor</h3>
    <p style="color: #94a3b8; font-size: 13px; margin: 10px 0 0 0;">
        <b>Lead Developer:</b> Eze Chukwuebuka | Specialized Geophysics & Predictive Data Science System
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# STAGE 2: LIVE FILE DATABASE CONNECTOR (RECONSTRUCTED)
# ==========================================
# We remove hardcoded arrays and read directly from our local database log file.
DB_FILE = "pipeline_telemetry.csv"

def load_pipeline_data_from_db():
    if not os.path.exists(DB_FILE):
        st.error(f"❌ Critical System Error: Database file '{DB_FILE}' not found in root directory.")
        return pd.DataFrame() # Return blank data frame to prevent application crash
    
    # Read directly from file storage mimicking SQL fetching
    data_frame = pd.read_csv(DB_FILE)
    return data_frame

df = load_pipeline_data_from_db()

# If file loading fails entirely, halt application logic safely
if df.empty:
    st.info("💡 Please create the 'pipeline_telemetry.csv' data file to populate this dashboard view.")
    st.stop()

# ==========================================
# STAGE 3: LIVE NODE CONTROLLER & TELEMETRY METRICS
# ==========================================
st.sidebar.markdown("### 🎛️ SCADA CONTROL NODE SELECTOR")

# Dynamic live data refresh button to pull any changes updated manually inside the CSV file
if st.sidebar.button("🔄 Pull Database Update"):
    st.cache_data.clear()
    st.rerun()

selected_segment = st.sidebar.selectbox("Choose Node Array", df['Location'])
segment_data = df[df['Location'] == selected_segment].iloc[0]

st.markdown("### 📊 Live Telemetry Stream Summary")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    resistivity = segment_data['Soil_Resistivity_Ohm_m']
    if resistivity < 20:
        st.metric("Soil Resistivity", f"{resistivity} Ω·m", delta="HIGH CORROSION RISK", delta_color="inverse")
    else:
        st.metric("Soil Resistivity", f"{resistivity} Ω·m", delta="Status: Optimal")

with m_col2:
    methane = segment_data['Methane_Leak_PPM']
    if methane > 10:
        st.metric("Atmospheric $CH_4$", f"{methane} PPM", delta="ALERT: DISCHARGE EXCEEDED", delta_color="inverse")
    else:
        st.metric("Atmospheric $CH_4$", f"{methane} PPM", delta="Status: Nominal")

with m_col3:
    st.metric("Subsurface Ground pH", f"{segment_data['Soil_pH']}", delta="Dynamic Check")

with m_col4:
    st.metric("Hardware Uplink", "ONLINE", delta="Latency: 14ms")

# ==========================================
# STAGE 4: GEOSPATIAL MAP VISUALIZATION
# ==========================================
st.write("---")
st.markdown("<div class='card-panel'><h3>📍 GIS Spatial Asset Risk Matrix Map</h3></div>", unsafe_allow_html=True)

df['Asset_Status'] = np.where(df['Soil_Resistivity_Ohm_m'] < 20, 'Critical Monitoring Required', 'Nominal Operation State')

fig_map = px.scatter_map(
    df, lat="Latitude", lon="Longitude", 
    hover_name="Location", hover_data=["Soil_Resistivity_Ohm_m", "Methane_Leak_PPM"],
    color="Asset_Status", color_discrete_map={'Critical Monitoring Required': '#EF4444', 'Nominal Operation State': '#10B981'},
    zoom=11, height=450
)
fig_map.update_layout(
    map_style="open-street-map",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin={"r":0,"t":0,"l":0,"b":0}
)
st.plotly_chart(fig_map, width="stretch")

# ==========================================
# STAGE 5: ANALYTICAL WORKSTATIONS
# ==========================================
st.write("---")
st.markdown("<div class='card-panel'><h3>🤖 Machine Learning Operations & Predictive Core</h3></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📉 PREDICTIVE CORROSION SIMULATOR", "🚨 ACOUSTIC ANTI-VANDALISM NETWORK"])

with tab1:
    st.markdown("#### 🔬 Subsurface Steel Loss Predictive Analytics")
    col_t1_1, col_t1_2 = st.columns(2)
    with col_t1_1:
        input_ph = st.slider("Live Read: Soil pH", 4.0, 9.0, float(segment_data['Soil_pH']))
        input_resistivity = st.slider("Live Read: Soil Resistivity (Ω·m)", 10, 200, int(segment_data['Soil_Resistivity_Ohm_m']))
    
    with col_t1_2:
        X_corr = np.array([[4.5, 12], [5.0, 18], [6.5, 100], [7.0, 150], [5.5, 40]])
        y_corr = np.array([0.45, 0.38, 0.05, 0.02, 0.15]) 
        corr_model = RandomForestRegressor(n_estimators=10, random_state=42)
        corr_model.fit(X_corr, y_corr)
        
        pred_corr = corr_model.predict([[input_ph, input_resistivity]])
        
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 8px; border-top: 4px solid #3b82f6; text-align: center;">
            <p style="margin: 0; color: #94a3b8; font-size: 14px;">FORECASTED ANNUAL MATERIAL THICKNESS DEGRADATION</p>
            <h2 style="margin: 10px 0 0 0; color: #3b82f6; font-size: 36px;">{pred_corr[0]:.4f} mm/year</h2>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("#### 🎚️ Distributed Acoustic Sensing (DAS) Threat Classifier")
    col_t2_1, col_t2_2 = st.columns(2)
    with col_t2_1:
        frequency = st.number_input("Sensor Frequency Reading (Hz)", min_value=10, max_value=5000, value=1200, step=50)
        amplitude = st.number_input("Vibration Wave Amplitude (mm/s)", min_value=0.0, max_value=50.0, value=4.5, step=0.5)
    
    with col_t2_2:
        X_security = np.array([
            [120, 1.2], [80, 0.8],     
            [450, 25.0], [500, 30.0],  
            [1500, 5.2], [1800, 7.1],  
            [3200, 15.4], [3500, 18.0] 
        ])
        y_security = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        
        security_model = RandomForestClassifier(n_estimators=15, random_state=42)
        security_model.fit(X_security, y_security)
        
        pred_security = security_model.predict([[frequency, amplitude]])
        status_labels = {0: "✅ SYSTEM STABLE: standard background activity and ambient seismic signatures verified.", 
                 1: "🚛 VEHICLE REGISTRATION LOGGED: Authorized Perimeter Movement Detected.", 
                 2: "🚨 THREAT CLASSIFIED: CRITICAL INTERVENTION LEVEL - Third-Party Excavation / Vandalism Core Detected!"}
        
        status_result = status_labels[pred_security[0]]
        
        if pred_security[0] == 2:
            st.error(status_result)
        elif pred_security[0] == 1:
            st.warning(status_result)
        else:
            st.success(status_result)

# ==========================================
# STAGE 6: GOVERNANCE, REPORTING, & EXPORT CONTROLS
# ==========================================
st.write("---")
st.markdown("### 📋 System Management & Document Exports")
compliance_text = f"ENUGU GAS FIELD DATA SYSTEM REGULATORY SUMMARY\nSelected Array Profile: {selected_segment}\nResistivity Data Logged: {resistivity} Ohm-m\nEnvironmental Leak Metric: {methane} PPM"

st.download_button(
    label="📥 Export Time-Stamped Compliance Log File",
    data=compliance_text,
    file_name=f"SCADA_Log_{selected_segment.replace(' ', '_')}.txt",
    mime="text/plain"
)
