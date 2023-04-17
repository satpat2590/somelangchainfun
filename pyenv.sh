#!/bin/bash

# Set environment name and requirements file
ENV_NAME="langbabyagi"
REQUIREMENTS_FILE="requirements.txt"

# Create the virtual environment
echo "Creating virtual environment $ENV_NAME..."
python3 -m venv $ENV_NAME

# Activate the environment
echo "Activating virtual environment $ENV_NAME..."
source $ENV_NAME/bin/activate

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip3 install --upgrade pip setuptools

# Install packages from requirements.txt
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing packages from $REQUIREMENTS_FILE..."
    pip3 install -r $REQUIREMENTS_FILE
else
    echo "No $REQUIREMENTS_FILE found, skipping package installation."
fi

# Install FAISS using pip (unofficial version)
echo "Installing FAISS..."
pip3 install faiss-cpu

echo "Setup completed. To use the new environment, run 'source $ENV_NAME/bin/activate'."
