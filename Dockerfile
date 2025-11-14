# Use a full Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install system build dependencies (gcc, g++, etc.) for compiling scikit-learn from source if necessary
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-dev \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip

# Install scikit-learn (use --prefer-binary to force the use of precompiled binaries)
RUN pip install --prefer-binary scikit-learn

# Now install the rest of the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Set the default command to run when the container starts
CMD ["python", "app.py"]
