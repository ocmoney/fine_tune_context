#!/bin/bash

# Create the project directory
mkdir -p yana-chatbot
cd yana-chatbot

# Copy all necessary files
cp ../Dockerfile .
cp ../docker-compose.yml .
cp ../requirements.txt .
cp ../yana_app.py .

# Create model directory
mkdir -p lora-dino-model
cp -r ../lora-dino-model/* lora-dino-model/

echo "Files have been copied to the yana-chatbot directory"
echo "You can now deploy the stack in Portainer using the docker-compose.yml in this directory" 