#!/bin/bash

# Exit on error
set -e

echo "Starting setup process..."

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg

# Install Miniconda
echo "Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /opt/conda
rm miniconda.sh

# Add conda to PATH and initialize
echo "Configuring conda..."
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
# Source bashrc to make conda available in this script session immediately after init
source ~/.bashrc 
/opt/conda/bin/conda init bash
# Source bashrc again to pick up changes from conda init and ensure subsequent commands in this script work
source ~/.bashrc 

# Create and activate conda environment
echo "Creating conda environment..."
ENV_NAME="fine_tune_context"
/opt/conda/bin/conda create -y -n $ENV_NAME python=3.10
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate $ENV_NAME

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Configure Git
echo "Configuring Git..."
git config --global user.name "ocmoney"
git config --global user.email "olliecumming3@gmail.com"

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << EOL
# Miniconda
miniconda/
miniconda.sh
/opt/conda/

# Models
models/
*.pt
*.pth
*.h5
*.ckpt
*.bin
*.onnx

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOL

echo "Setup completed successfully!"
echo "To activate the environment, run: conda activate $ENV_NAME" 