# Step 1: Use the official Python image as a base image
#FROM python:3.9-slim-bullseye
# Use an official Python runtime as a parent image
FROM python:3.11-alpine

# Set the working directory in the container
WORKDIR /app

# # Install build dependencies (including GCC and other build tools)
# RUN apk update && \
#     apk add --no-cache \
#     build-base \
#     gcc \
#     g++ \
#     openblas-dev \
#     python3-dev \
#     meson \
#     libffi-dev \
#     musl-dev \
#     bash

# Step 4: Copy requirements.txt and install Python dependencies
# COPY requirements.txt /app/
COPY . /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install seaborn
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
# COPY . /app/

# Set the default command to run when the container starts
CMD ["python", "app.py"]
