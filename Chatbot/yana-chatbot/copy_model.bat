@echo off
echo Copying model files to server...
scp -r lora-dino-model/* root@95.216.217.156:/var/lib/docker/volumes/yana-model-data/_data
echo Done!
pause 