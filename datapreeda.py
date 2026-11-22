import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set display options for thorough EDA output
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
sns.set_theme(style="whitegrid")
# data loading
def load_data(cc_path='creditcard.csv', paysim_path='paysim.csv'):
    """Loads Credit Card and PaySim datasets with initial shape checks."""
    try:
        df_cc = pd.read_csv(cc_path)
        print(f" Loaded Credit Card Dataset: {df_cc.shape[0]:,} rows, {df_cc.shape[1]} columns")
    except FileNotFoundError:
        print(f" {cc_path} not found. Ensure the dataset is downloaded from Kaggle.")
        df_cc = None

    try:
        df_paysim = pd.read_csv(paysim_path)
        print(f" Loaded PaySim Dataset: {df_paysim.shape[0]:,} rows, {df_paysim.shape[1]} columns")
    except FileNotFoundError:
        print(f"️ {paysim_path} not found. Ensure the dataset is downloaded from Kaggle.")
        df_paysim = None

    return df_cc, df_paysim

#EDA 
def perform_eda_creditcard(df):
    """Exploratory Data Analysis for Credit Card Payment Rail."""
    print("\n" + "="*50)
    print("--- CREDIT CARD DATASET EDA ---")
    print("="*50)

    print("\nMissing Values:", df.isnull().sum().sum())
    

    fraud_counts = df['Class'].value_counts()
    fraud_pct = df['Class'].value_counts(normalize=True) * 100
    print(f"\nClass Distribution:\nLegitimate (0): {fraud_counts[0]:,} ({fraud_pct[0]:.2f}%)\nFraudulent (1): {fraud_counts[1]:,} ({fraud_pct[1]:.2f}%)")


    print("\nTransaction Amount Summary ($):")
    print(df.groupby('Class')['Amount'].describe())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(x='Class', data=df, ax=axes[0], palette=['#2b5c8f', '#d9534f'])
    axes[0].set_title("Credit Card Fraud Class Distribution", fontsize=12)
    axes[0].set_yscale('log') 

    sns.boxplot(x='Class', y='Amount', data=df, ax=axes[1], palette=['#2b5c8f', '#d9534f'], showfliers=False)
    axes[1].set_title("Transaction Amount Distribution (Excluding Outliers)", fontsize=12)

    plt.tight_layout()
    plt.savefig('eda_creditcard.png', dpi=300)
    plt.close()
    print(" Saved Credit Card EDA plots as 'eda_creditcard.png'")


def perform_eda_paysim(df):
    """Exploratory Data Analysis for PaySim (E-Wallet / UPI) Rail."""
    print("\n" + "="*50)
    print("--- PAYSIM / E-WALLET DATASET EDA ---")
    print("="*50)

    print("\nTransaction Count by Channel Type:")
    print(df['type'].value_counts())

    fraud_by_type = df.groupby('type')['isFraud'].sum()
    print("\nFraud Cases per Channel Type:")
    print(fraud_by_type)

    df['orig_balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    print("\nMean Balance Change During Fraud vs Non-Fraud:")
    print(df.groupby('isFraud')[['amount', 'orig_balance_diff']].mean())

#PREPROCESSING
def preprocess_creditcard(df):
    """
    Feature transformation and scaling for Credit Card rail.
    Calculates Z-score: Z = (X - \mu) / \sigma for Amount feature.
    """
    df_clean = df.copy()
    scaler = StandardScaler()
    df_clean['amount_zscore'] = scaler.fit_transform(df_clean[['Amount']])
    df_clean['time_scaled'] = scaler.fit_transform(df_clean[['Time']])


    X = df_clean.drop(columns=['Class', 'Amount', 'Time'])
    y = df_clean['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n Preprocessed Credit Card Pipeline:")
    print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")
    print(f"Train Fraud Ratio: {y_train.mean()*100:.3f}% | Test Fraud Ratio: {y_test.mean()*100:.3f}%")

    return X_train, X_test, y_train, y_test, scaler


def preprocess_paysim(df):
    """
    Filters relevant transfer types and engineers e-wallet risk signals.
    """
    # PaySim fraud only occurs in 'TRANSFER' and 'CASH_OUT'
    df_filtered = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

    df_filtered['balance_err_orig'] = df_filtered['newbalanceOrig'] + df_filtered['amount'] - df_filtered['oldbalanceOrg']
    df_filtered['balance_err_dest'] = df_filtered['oldbalanceDest'] + df_filtered['amount'] - df_filtered['newbalanceDest']


    df_filtered = pd.get_dummies(df_filtered, columns=['type'], drop_first=True)

    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 
                'newbalanceDest', 'balance_err_orig', 'balance_err_dest', 'type_TRANSFER']
    
    X = df_filtered[features]
    y = df_filtered['isFraud']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n Preprocessed PaySim Pipeline:")
    print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    df_cc, df_paysim = load_data()

    if df_cc is not None:
        perform_eda_creditcard(df_cc)
        X_train_cc, X_test_cc, y_train_cc, y_test_cc, scaler_cc = preprocess_creditcard(df_cc)

    if df_paysim is not None:
        perform_eda_paysim(df_paysim)
        X_train_ps, X_test_ps, y_train_ps, y_test_ps, scaler_ps = preprocess_paysim(df_paysim)
