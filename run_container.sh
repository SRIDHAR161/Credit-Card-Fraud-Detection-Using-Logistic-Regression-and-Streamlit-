#!/bin/bash

# Stop any container using port 8501
docker ps --filter "publish=8501" -q | xargs -r docker stop

# Build Docker image
docker build -t paysure-app .

# Run the container
docker run -d -p 8501:8501 --name paysure_container paysure-app
