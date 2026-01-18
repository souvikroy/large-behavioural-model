#!/bin/bash

# Run Streamlit frontend for Question Quest

echo "Starting Streamlit Frontend..."
echo "Make sure the FastAPI backend is running on http://localhost:9010"
echo ""

streamlit run app.py
