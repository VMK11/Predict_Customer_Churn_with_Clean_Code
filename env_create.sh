#!/bin/bash

"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 11-09-2023
"""

# Name of the environment
ENV_NAME="predict_customer_churn_with_clean_code"

# Python version
PYTHON_VERSION="3.6"

# List of packages
PACKAGES=(
    "scikit-learn==0.22"
    "shap==0.40.0"
    "joblib==0.11"
    "pandas==0.23.3"
    "numpy==1.19.5"
    "matplotlib==2.1.0"
    "seaborn==0.11.2"
    "pylint==2.7.4"
    "autopep8==1.5.6"
    "PyYAML"
)

# Create new environment
echo "Creating new environment named ${ENV_NAME} with Python ${PYTHON_VERSION}..."
conda create --name $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo "Activating environment..."
conda activate $ENV_NAME

# Install packages
echo "Installing packages..."
for PACKAGE in "${PACKAGES[@]}"
do
    conda install $PACKAGE -y
done

echo "Environment setup complete."
