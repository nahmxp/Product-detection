#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check if required packages are installed
echo "Checking dependencies..."
pip list | grep -q streamlit || {
    echo "Installing missing dependencies..."
    pip install -r requirements.txt
}

# Launch Streamlit app
echo "🚀 Launching YOLO Training Dashboard..."
streamlit run app.py --server.port 8501 --server.address localhost


