#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Launching Streamlit app..."
streamlit run app.py
