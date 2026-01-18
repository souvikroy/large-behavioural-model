#!/bin/bash
set -e

echo "Starting Question Quest API..."

# Ensure spaCy model is available
echo "Checking spaCy model..."
python -c "import spacy; spacy.load('en_core_web_sm')" || {
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
}

# Create necessary directories if they don't exist
mkdir -p models knowledge_graphs logs

# Start the application
echo "Starting uvicorn server..."
exec python -m uvicorn src.api.app:app --host 0.0.0.0 --port 9010
