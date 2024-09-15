#!/bin/bash
python3 feature-model-training/main.py
mkdir -p feature-api
mv model.pkl feature-api/
