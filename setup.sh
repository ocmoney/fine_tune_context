#!/bin/bash

# Exit on error
set -e

echo "Starting setup process..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg

# Install Miniconda
echo "Installing Miniconda..."
if [ ! -d "/opt/conda" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /opt/conda
    rm miniconda.sh
else
    echo "Miniconda already installed at /opt/conda"
fi

# Add conda to PATH for current session
export PATH="/opt/conda/bin:$PATH"

# Initialize conda for current shell
eval "$(/opt/conda/bin/conda shell.bash hook)"

# Add conda to PATH permanently
if [[ ":$PATH:" != *":/opt/conda/bin:"* ]]; then
    echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
fi

# Create and activate conda environment
echo "Creating conda environment..."
ENV_NAME="fine_tune_context"
if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -y -n $ENV_NAME python=3.10
fi

# Activate the environment
conda activate $ENV_NAME

# Install requirements
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

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
echo "To activate the environment, run: source /opt/conda/etc/profile.d/conda.sh && conda activate $ENV_NAME" 
