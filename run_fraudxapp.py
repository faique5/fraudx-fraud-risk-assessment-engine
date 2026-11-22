aimportsubprocess
import sys

if __name__ == "__main__":
    # Automatically triggers Streamlit using the current Python environment
    subprocess.run([sys.executable, "-m", "streamlit", "run", "fraudxapp_streamlit.py"])
