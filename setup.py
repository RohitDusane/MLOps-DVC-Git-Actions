# ============================
# ========= SETUP ============
# ============================
from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='MLOPS-PROJECT-1',
    version='1.0',
    description='An End-to-End Credit Scoring Model - MLOps and GCP',
    author='Rohit Dusane',
    author_email='stat.data247@gmail.com',
    packages=find_packages(),  # Automatically discover all packages in the project
    install_requires=requirements,  # Install the dependencies from requirements.txt
    python_requires='>=3.6',  # Change this to a more widely compatible version
    license="MIT",  # Specify the project license
)
