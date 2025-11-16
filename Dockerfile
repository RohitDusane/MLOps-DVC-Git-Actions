# Use a full Python runtime as a parent image
FROM python:3.11-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the rest of the application code into the container
COPY . /app

# Install system build dependencies for compiling scikit-learn from source if necessary
RUN apt-get update && apt-get install -y
#     build-essential \
#     libopenblas-dev \
#     python3-dev \
#     gfortran \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean


# Copy the requirements.txt file into the container
# COPY requirements.txt /app/

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -e .

EXPOSE 5000

ENV FLASK_APP=app.py


# Set the default command to run the application (adjust this if your entry point is different)
CMD ["python", "app.py"]
 