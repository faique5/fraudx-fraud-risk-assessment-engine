import os
import json
import socket
import ipaddress
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier


# 1. UPI HYBRID ENGINE (Baseline Distributions + Cyber Threat Vectors)
def train_upi_hybrid():
    print("🔹 Training UPI Model (Hybrid Baseline + Vishing/RAT Signals)...")
    np.random.seed(42)
    n = 15000

    amounts = np.random.exponential(2000, size=n)
    z_scores = (amounts - np.mean(amounts)) / (np.std(amounts) + 1e-6)

    is_call = np.random.choice([0, 1], size=n, p=[0.94, 0.06])
    screen_share = np.random.choice([0, 1], size=n, p=[0.98, 0.02])
    is_pull = np.random.choice([0, 1], size=n, p=[0.90, 0.10])

    fraud_labels = []
    for i in range(n):
        risk = 0.01
        if is_call[i] and z_scores[i] > 1.2:
            risk += 0.50
        if screen_share[i]:
            risk += 0.70
        if is_pull[i] and z_scores[i] > 1.5:
            risk += 0.35
        fraud_labels.append(1 if np.random.rand() < risk else 0)

    df = pd.DataFrame({
        'amount': amounts,
        'amount_zscore': z_scores,
        'is_pull_request': is_pull,
        'is_call_active': is_call,
        'screen_share_active': screen_share,
        'is_fraud': fraud_labels
    })

    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=3).fit(X_scaled, y)

    return {'model': model, 'scaler': scaler, 'features': list(X.columns)}


# 2. CREDIT CARD HYBRID ENGINE (Kaggle Dataset + Synthetic Device Reputation)
def train_credit_card_hybrid():
    csv_path = "creditcard.csv"
    np.random.seed(42)

    if os.path.exists(csv_path):
        print(f"🔹 Found '{csv_path}'. Ingesting real-world Kaggle Credit Card dataset...")
        real_df = pd.read_csv(csv_path)
        if len(real_df) > 40000:
            real_df = real_df.sample(n=40000, random_state=42)

        real_df['amount'] = real_df['Amount']
        mean_amt, std_amt = real_df['amount'].mean(), real_df['amount'].std()
        real_df['amount_zscore'] = (real_df['amount'] - mean_amt) / (std_amt + 1e-6)

        n = len(real_df)
        real_df['is_off_hours'] = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
        real_df['device_reputation'] = np.random.uniform(0.1, 1.0, size=n)

        X = real_df[['amount', 'amount_zscore', 'is_off_hours', 'device_reputation']]
        y = real_df['Class']
    else:
        print("🔹 'creditcard.csv' not found. Using robust synthetic dataset generation...")
        n = 10000
        df = pd.DataFrame({
            'amount': np.random.exponential(5000, size=n),
            'amount_zscore': np.random.normal(0, 1.2, size=n),
            'is_off_hours': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
            'device_reputation': np.random.uniform(0.1, 1.0, size=n),
            'is_fraud': np.random.choice([0, 1], size=n, p=[0.99, 0.01])
        })
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=3).fit(X_scaled, y)

    return {'model': model, 'scaler': scaler, 'features': list(X.columns)}


# 3. E-WALLET HYBRID ENGINE (PaySim Dataset + Velocity Injection)
def train_ewallet_hybrid():
    csv_path = "paysim.csv"
    np.random.seed(42)

    if os.path.exists(csv_path):
        print(f"🔹 Found '{csv_path}'. Ingesting real-world PaySim dataset...")
        real_df = pd.read_csv(csv_path)
        real_df = real_df[real_df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
        if len(real_df) > 40000:
            real_df = real_df.sample(n=40000, random_state=42)

        real_df['balance_diff_orig'] = real_df['oldbalanceOrg'] - real_df['newbalanceOrig']
        n = len(real_df)
        real_df['txn_velocity_10m'] = np.random.poisson(0.5, size=n)

        X = real_df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_diff_orig', 'txn_velocity_10m']]
        y = real_df['isFraud']
    else:
        print("🔹 'paysim.csv' not found. Using robust synthetic dataset generation...")
        n = 10000
        df = pd.DataFrame({
            'amount': np.random.exponential(1500, size=n),
            'oldbalanceOrg': np.random.uniform(100, 50000, size=n),
            'newbalanceOrig': np.random.uniform(0, 50000, size=n),
            'txn_velocity_10m': np.random.poisson(0.5, size=n),
            'is_fraud': np.random.choice([0, 1], size=n, p=[0.985, 0.015])
        })
        df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_diff_orig', 'txn_velocity_10m']]
        y = df['is_fraud']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=3).fit(X_scaled, y)

    return {'model': model, 'scaler': scaler, 'features': list(X.columns)}


# 4. DYNAMIC NETWORK DETECTION & RISK PROBABILITY
def get_local_ipv4() -> str | None:
    """Detect the active outbound local IPv4 address without sending external packets."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def is_private_ipv4(ip_str: str | None) -> bool:
    """Return True if the IPv4 address is RFC1918 private, loopback, or link-local."""
    if not ip_str:
        return False
    try:
        addr = ipaddress.ip_address(ip_str)
        return bool(addr.is_private or addr.is_loopback or addr.is_link_local)
    except ValueError:
        return False


def fetch_public_ip_metadata(timeout: float = 1.5) -> dict:
    """Best-effort lookup of public IP and ISP/ASN metadata for dynamic public-network risk."""
    info = {"public_ip": None, "org": "", "is_cellular_or_public_wifi_hint": False}
    try:
        req = urllib.request.Request(
            "https://ipinfo.io/json",
            headers={"User-Agent": "UniversalFraudGuard/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            info["public_ip"] = data.get("ip")
            org = (data.get("org") or "").lower()
            info["org"] = data.get("org") or ""
            risky_keywords = [
                "wifi", "hotspot", "public", "airport", "cafe", "hotel",
                "cloud", "amazon", "aws", "google", "digitalocean", "vpn", "proxy"
            ]
            info["is_cellular_or_public_wifi_hint"] = any(k in org for k in risky_keywords)
    except Exception:
        pass
    return info


def detect_network_context(payload: dict) -> dict:
    """
    Dynamically determine whether the user is on a PRIVATE or PUBLIC network.
    Supports 'Auto-Detect (Live)' or manual simulation override from the UI.
    """
    mode = payload.get("network_mode", "Auto-Detect (Live)")
    local_ip = get_local_ipv4()

    if mode == "Force Private Network":
        net_type = "PRIVATE"
        source = "Manual Override"
        pub_meta = {"public_ip": None, "org": "Local Private LAN / Home / Hostel Wi-Fi", "is_cellular_or_public_wifi_hint": False}
    elif mode == "Force Public Network":
        net_type = "PUBLIC"
        source = "Manual Override"
        pub_meta = {"public_ip": "Simulated Public Node", "org": "Public Wi-Fi / Unverified Hotspot", "is_cellular_or_public_wifi_hint": True}
    else:
        pub_meta = fetch_public_ip_metadata()
        source = "Live Socket + IP Inspection"
        if is_private_ipv4(local_ip) and not pub_meta["is_cellular_or_public_wifi_hint"]:
            net_type = "PRIVATE"
        elif pub_meta["public_ip"] and not is_private_ipv4(local_ip):
            net_type = "PUBLIC"
        elif pub_meta["is_cellular_or_public_wifi_hint"]:
            net_type = "PUBLIC"
        else:
            net_type = "PRIVATE" if is_private_ipv4(local_ip) else "PUBLIC"

    return {
        "network_type": net_type,
        "local_ip": local_ip or "Unavailable",
        "public_ip": pub_meta.get("public_ip") or "N/A",
        "network_org": pub_meta.get("org") or ("Private LAN" if net_type == "PRIVATE" else "Public / External Network"),
        "detection_source": source,
        "captive_or_open_risk": bool(payload.get("unsecured_network", False)),
        "vpn_or_proxy_active": bool(payload.get("vpn_or_proxy", False)),
    }


def compute_network_risk_probability(net_ctx: dict, payload: dict) -> tuple[float, float, list[str]]:
    """
    Return:
      - net_prob_pct (1.0 to 95.0): network fraud/interception risk probability in %
      - net_rule_overlay (0.0 to 40.0): risk points fed into rule_risk_score
      - warnings: network risk alerts
    """
    warnings = []
    net_type = net_ctx["network_type"]
    channel = payload.get("payment_channel", "UPI")
    amount = float(payload.get("amount", 0.0))

    if net_type == "PRIVATE":
        prob = 5.0
        overlay = 4.0
        if net_ctx.get("vpn_or_proxy_active"):
            prob += 7.0
            overlay += 5.0
            warnings.append("🌐 VPN/Proxy active on private network — verify endpoint trust before high-value transfer.")
        if amount >= 25000:
            prob += 4.0
            overlay += 3.0
    else:
        prob = 32.0
        overlay = 22.0
        warnings.append("🌐 Transaction initiated over a PUBLIC network — higher exposure to MITM or rogue Wi-Fi.")
        if net_ctx.get("captive_or_open_risk"):
            prob += 24.0
            overlay += 16.0
            warnings.append("🚨 Open / unencrypted public network flag enabled — severe interception risk.")
        if net_ctx.get("vpn_or_proxy_active"):
            prob -= 5.0
            overlay -= 3.0
        if channel == "UPI" and payload.get("is_pull_request"):
            prob += 10.0
            overlay += 7.0
        if amount >= 10000:
            prob += 10.0
            overlay += 6.0

    prob = float(np.clip(prob, 1.0, 95.0))
    overlay = float(np.clip(overlay, 0.0, 40.0))
    return round(prob, 2), round(overlay, 2), warnings


# 5. DYNAMIC PREDICTION ENGINE
def predict_transaction_safety(raw_txn: dict, artifacts: dict) -> dict:
    channel = raw_txn.get('payment_channel', 'UPI')
    pkg = artifacts.get(channel, artifacts['UPI']) if isinstance(artifacts, dict) and artifacts else None

    if isinstance(pkg, dict) and 'model' in pkg and 'scaler' in pkg and 'features' in pkg:
        model = pkg['model']
        scaler = pkg['scaler']
        required_features = pkg['features']
        input_vector = [float(raw_txn.get(f, 0.0)) for f in required_features]
        df_input = pd.DataFrame([input_vector], columns=required_features)
        scaled_input = scaler.transform(df_input)
        ml_fraud_prob = float(model.predict_proba(scaled_input)[0, 1]) * 100.0
    else:
        z = float(raw_txn.get('amount_zscore', 0.0))
        ml_fraud_prob = float(np.clip(15.0 + max(0.0, z) * 15.0, 2.0, 95.0))

    net_ctx = detect_network_context(raw_txn)
    net_risk_prob, net_rule_overlay, warnings = compute_network_risk_probability(net_ctx, raw_txn)

    rule_risk_score = net_rule_overlay

    if channel == 'UPI':
        z_score = float(raw_txn.get('amount_zscore', 0.0))
        is_call = raw_txn.get('is_call_active', 0)
        screen_share = raw_txn.get('screen_share_active', 0)
        is_pull = raw_txn.get('is_pull_request', 0)

        if z_score > 0.0:
            rule_risk_score += min(40.0, z_score * 12.0)
            if z_score >= 1.5:
                warnings.append(f"⚠️ High Amount Anomaly: Spending is {z_score:.2f} SDs above user average.")

        if is_call == 1:
            penalty = 25.0 + (max(0.0, z_score) * 6.0)
            rule_risk_score += penalty
            warnings.append(f"🚨 Active Phone Call Vector: Live cellular call detected during transfer (+{penalty:.1f}% risk).")

        if screen_share == 1:
            penalty = 40.0 + (max(0.0, z_score) * 8.0)
            rule_risk_score += penalty
            warnings.append(f"🚨 Screen Sharing Active: High Remote Trojan (RAT) threat (+{penalty:.1f}% risk).")

        if is_pull == 1:
            penalty = 15.0 + (max(0.0, z_score) * 5.0)
            rule_risk_score += penalty
            warnings.append(f"⚠️ Unsolicited Collect Request: Incoming UPI pull request (+{penalty:.1f}% risk).")

    elif channel == 'CREDIT_CARD':
        z_score = float(raw_txn.get('amount_zscore', 0.0))
        is_off_hours = raw_txn.get('is_off_hours', 0)
        device_rep = float(raw_txn.get('device_reputation', 1.0))

        if z_score > 0.0:
            rule_risk_score += min(60.0, z_score * 15.0)
            if z_score >= 1.5:
                warnings.append(f"⚠️ Spend Outlier: Transaction amount is {z_score:.2f} SDs above normal.")

        if is_off_hours == 1:
            rule_risk_score += 15.0
            warnings.append("🌙 Off-Hours Anomaly: Transaction attempted during non-standard hours.")

        if device_rep < 1.0:
            dev_penalty = (1.0 - device_rep) * 45.0
            rule_risk_score += dev_penalty
            if device_rep < 0.6:
                warnings.append(f"🛡️ Untrusted Device: Low device reputation score ({device_rep:.2f}).")

    elif channel == 'E_WALLET':
        amount = float(raw_txn.get('amount', 0.0))
        old_bal = float(raw_txn.get('oldbalanceOrg', 0.0))
        new_bal = float(raw_txn.get('newbalanceOrig', 0.0))
        velocity = float(raw_txn.get('txn_velocity_10m', 0))

        if old_bal > 0:
            drain_pct = (amount / old_bal) * 100.0
            if drain_pct > 20.0:
                drain_penalty = min(50.0, (drain_pct - 20.0) * 0.7)
                rule_risk_score += drain_penalty
                warnings.append(f"🚨 Balance Drain Spike: Transaction attempts to move {drain_pct:.1f}% of wallet funds.")

        if old_bal > 0 and new_bal == 0:
            rule_risk_score += 20.0
            warnings.append("🚨 Zero-Balance Event: Account completely emptied.")

        if velocity > 0:
            vel_penalty = min(40.0, velocity * 10.0)
            rule_risk_score += vel_penalty
            if velocity >= 2:
                warnings.append(f"⚡ Rapid Velocity Spike: {int(velocity)} transactions in last 10 minutes.")

    combined_fraud_prob = min(99.9, max(0.1, (0.4 * ml_fraud_prob) + (0.6 * rule_risk_score)))
    safety_score = round(100.0 - combined_fraud_prob, 1)
    final_fraud_prob = round(combined_fraud_prob, 2)

    return {
        'payment_channel': channel,
        'network_type': net_ctx['network_type'],
        'network_risk_probability': net_risk_prob,
        'network_risk_label': f"{net_risk_prob:.2f}%",
        'network_context': net_ctx,
        'safety_score': safety_score,
        'fraud_probability': f"{final_fraud_prob:.2f}%",
        'risk_level': "HIGH" if safety_score < 60.0 else ("MEDIUM" if safety_score < 85.0 else "LOW"),
        'warnings': warnings
    }


def main():
    print("⏳ Starting Hybrid Dataset Multi-Channel Training Pipeline...")
    artifacts = {
        'UPI': train_upi_hybrid(),
        'CREDIT_CARD': train_credit_card_hybrid(),
        'E_WALLET': train_ewallet_hybrid()
    }
    joblib.dump(artifacts, 'multi_channel_fraud_artifacts.pkl')
    print("✅ Model bundle saved successfully to 'multi_channel_fraud_artifacts.pkl'")


if __name__ == '__main__':
    main()
