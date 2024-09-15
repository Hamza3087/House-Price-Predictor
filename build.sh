#!/bin/bash

# Check if model.pkl exists in the current directory
if [ ! -f model.pkl ]; then
    echo "model.pkl not found. Running main.py to generate model.pkl..."
    python feature-model-training/main.py
fi

# Move the generated model.pkl to the feature-api directory
if [ -f model.pkl ]; then
    mv model.pkl feature-api/
else
    echo "Error: model.pkl could not be created."
    exit 1
fi
