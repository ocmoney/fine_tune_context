@echo off

:: Create the project directory
mkdir yana-chatbot
cd yana-chatbot

:: Copy all necessary files
copy ..\Dockerfile .
copy ..\docker-compose.yml .
copy ..\requirements.txt .
copy ..\yana_app.py .

:: Create model directory
mkdir lora-dino-model
xcopy /E /I ..\lora-dino-model\* lora-dino-model\

echo Files have been copied to the yana-chatbot directory
echo You can now deploy the stack in Portainer using the docker-compose.yml in this directory 