import sys
import subprocess
import joblib
import plotly.graph_objects as go
import streamlit as st

# IDLE Auto-Launcher
if not st.runtime.exists():
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit(0)

from new_universal_fraudx_core import predict_transaction_safety

# Page Configuration
st.set_page_config(
    page_title="Universal Fraud Guard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Universal Financial Pre-Transaction Safety Engine")
st.markdown("Dynamic risk assessment routed by **Payment Rail, Network Type & Dataset**.")


# Model Artifact Loader
@st.cache_resource
def load_artifacts():
    return joblib.load('multi_channel_fraud_artifacts.pkl')


try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model file missing. Run 'python3 universal_fraudx_core.py' in Terminal first.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("💳 Select Payment Mode")
channel = st.sidebar.selectbox("Payment Channel", ['UPI', 'CREDIT_CARD', 'E_WALLET'])

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Network Environment (Dynamic)")
network_mode = st.sidebar.selectbox(
    "Network Detection Mode",
    ["Auto-Detect (Live)", "Force Private Network", "Force Public Network"]
)
unsecured_network = st.sidebar.checkbox("Open / Unsecured Public Wi-Fi?")
vpn_or_proxy = st.sidebar.checkbox("VPN / Proxy Active?")

st.sidebar.markdown("---")
st.sidebar.subheader("Live Parameters")

payload = {
    'payment_channel': channel,
    'network_mode': network_mode,
    'unsecured_network': unsecured_network,
    'vpn_or_proxy': vpn_or_proxy,
}

if channel == 'UPI':
    payload['amount'] = st.sidebar.number_input("Amount (₹)", value=2500, step=500, min_value=1)
    payload['amount_zscore'] = st.sidebar.slider("Amount Z-Score", -1.0, 5.0, 0.2, step=0.1)
    payload['is_pull_request'] = 1 if st.sidebar.checkbox("Is Collect Request?") else 0
    payload['is_call_active'] = 1 if st.sidebar.checkbox("Active Phone Call?") else 0
    payload['screen_share_active'] = 1 if st.sidebar.checkbox("Screen Sharing Active?") else 0

elif channel == 'CREDIT_CARD':
    payload['amount'] = st.sidebar.number_input("Amount ($)", value=1200, step=100, min_value=1)
    payload['amount_zscore'] = st.sidebar.slider("Historical Spend Z-Score", -1.0, 5.0, 1.2, step=0.1)
    payload['is_off_hours'] = 1 if st.sidebar.checkbox("Off-Hours Swipe?") else 0
    payload['device_reputation'] = st.sidebar.slider("Device Reputation Score", 0.0, 1.0, 0.85, step=0.05)

elif channel == 'E_WALLET':
    payload['amount'] = st.sidebar.number_input("Transfer Amount", value=3000, step=500, min_value=1)
    payload['oldbalanceOrg'] = st.sidebar.number_input("Old Wallet Balance", value=5000, step=1000, min_value=0)
    payload['newbalanceOrig'] = st.sidebar.number_input("New Balance After Txn", value=2000, step=1000, min_value=0)
    payload['balance_diff_orig'] = payload['oldbalanceOrg'] - payload['newbalanceOrig']
    payload['txn_velocity_10m'] = st.sidebar.slider("Txns in last 10 mins", 0, 10, 1)

# Live dynamic inference
res = predict_transaction_safety(payload, artifacts)
net_ctx = res.get("network_context", {})

# Assessment Display
st.markdown("---")
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📊 Safety Score Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=res['safety_score'],
        title={'text': f"Safety Score ({res['payment_channel']} • {res['network_type']})"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 60], 'color': "#ff4b4b"},
                {'range': [60, 85], 'color': "#ffa15a"},
                {'range': [85, 100], 'color': "#00cc96"}
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch")

    st.caption(
        f"Detected: **{res['network_type']}** | Local IP: `{net_ctx.get('local_ip', 'N/A')}` | "
        f"Public IP: `{net_ctx.get('public_ip', 'N/A')}` ({net_ctx.get('detection_source', '')})"
    )

with col2:
    st.subheader("📈 Risk Assessment Metrics")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Risk Level", res['risk_level'])
    with m_col2:
        prob = res['fraud_probability']
        prob_str = f"{prob * 100:.1f}%" if isinstance(prob, (int, float)) and prob <= 1.0 else str(prob)
        st.metric("Overall Fraud Prob.", prob_str)
    with m_col3:
        net_prob = res['network_risk_probability']
        st.metric(f"{res['network_type']} Net Risk", f"{net_prob * 100:.1f}%")

    st.markdown("---")
    st.markdown("### 🔍 Real-Time Threat Analysis")
    if res['warnings']:
        for alert in res['warnings']:
            st.error(f"🚨 {alert}")
    else:
        st.success("✅ Transaction and network parameters are within normal safety bounds.")
