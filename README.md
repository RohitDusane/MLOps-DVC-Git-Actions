---
# CI/CD Deployment of Machine Learning Model to GCP (MLOPS)

This project provides an automated CI/CD pipeline to deploy machine learning models to **Google Cloud Platform (GCP)** using **Google Cloud Run** and **Artifact Registry**. The pipeline leverages **GitHub Actions** to automate the entire process, from building a Docker container to deploying the model on GCP.

---

## **Key Features**

* **Automated Deployment**: CI/CD pipeline that triggers on every push to the `main` branch or pull request.
* **Dockerized ML Models**: The model is containerized with Docker to ensure consistent deployment across environments.
* **Cloud Run Integration**: Deploys the Docker image to **Google Cloud Run**, enabling scalable, serverless model serving.
* **Versioned Images**: Each deployment is tagged with the commit hash to ensure version control of Docker images.
* **Efficient Dependency Management**: The workflow optimizes the installation and caching of dependencies to speed up builds.

---

## **Architecture Overview**

1. **GitHub Actions Workflow**: Automates the CI/CD process on every push or pull request.
2. **Docker Image Creation**: The machine learning model and dependencies are containerized using Docker.
3. **Artifact Registry**: The Docker image is pushed to Google Cloud’s Artifact Registry for secure storage.
4. **Cloud Run Deployment**: The image is deployed to Google Cloud Run for seamless, scalable, serverless serving of the model.

---

## **Workflow Steps**

### 1. **Code Checkout**

* GitHub Actions checks out the latest code from the repository.

### 2. **Install Dependencies**

* Installs both **Python** and **system-level dependencies** required for the machine learning model and training environment.

### 3. **Run Data Pipeline**

* **Data Ingestion**: Collects raw data necessary for training.
* **Data Processing**: Cleans, transforms, and prepares the data.
* **Model Training**: Trains the machine learning model.

### 4. **Docker Image Build and Push**

* Builds the Docker image and tags it with the Git commit hash for versioning.
* Pushes the Docker image to **Google Cloud Artifact Registry**.

### 5. **Deploy to Cloud Run**

* Deploys the Docker image to **Google Cloud Run**, making the model available as a fully managed, scalable web service.

---

## **Prerequisites**

Before using this pipeline, ensure that you have the following:

1. **Google Cloud Project**: Set up a GCP project where Cloud Run and Artifact Registry will be used.
2. **Google Cloud SDK**: Required for managing GCP services like Cloud Run and Artifact Registry.
3. **Service Account Key**: Create a service account in GCP with the required permissions (`roles/cloudrun.admin`, `roles/artifactregistry.writer`, etc.) and store the key as a GitHub secret.

---

## **Setup**

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/ml-cicd-deployment-gcp.git
cd ml-cicd-deployment-gcp
```

### **2. Add Google Cloud Service Account Key to GitHub Secrets**

* In your GitHub repository, go to **Settings > Secrets**.
* Add the following secrets:

  * `GCP_PROJECT_ID`: Your GCP project ID.
  * `GCP_SA_KEY`: The service account key (in JSON format) for GCP authentication.

### **3. Customize Your GCP Settings**

* Update the `REGION` and `ARTIFACT_REGISTRY_URL` in the workflow file (`.github/workflows/ml_pipeline.yml`) to match your GCP setup.

```yaml
REGION: us-central1  # or your preferred GCP region
ARTIFACT_REGISTRY_URL: us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/my-repo/credit-card-risk-prediction
```

### **4. Commit and Push Changes**

```bash
git add .
git commit -m "Set up CI/CD pipeline for ML model deployment"
git push origin main
```

### **5. Trigger the CI/CD Pipeline**

The GitHub Actions workflow will automatically trigger on the `main` branch, starting from code checkout to deployment on Google Cloud Run.

---

## **Deployment Details**

* **Cloud Service**: The model is deployed to **Google Cloud Run**, which automatically scales your service based on incoming traffic. It handles the heavy lifting of provisioning and managing infrastructure for you.
* **Authentication**: The service uses the service account key stored in GitHub Secrets for secure authentication to Google Cloud services.

---

## **Troubleshooting**

* **Logs**: You can check the logs of your Cloud Run deployment by visiting the **Google Cloud Console** under **Cloud Run > [Your Service] > Logs**.
* **Timeouts**: If the container fails to start within the allocated timeout, try increasing the timeout duration or check the port configuration (default is `PORT=8080`).
* **Permission Issues**: Ensure that your service account has the required permissions to access Cloud Run, Artifact Registry, and other necessary resources.

For more details on troubleshooting Cloud Run deployments, visit the [GCP troubleshooting documentation](https://cloud.google.com/run/docs/troubleshooting).

---

## **Contributing**

Feel free to fork this repository, submit issues, or create pull requests. Contributions are welcome to enhance the functionality, optimize the workflow, or fix bugs!

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

* **GitHub Actions**: For continuous integration and deployment.
* **Docker**: For containerizing the machine learning models.
* **Google Cloud Platform (GCP)**: For providing scalable infrastructure with Cloud Run and Artifact Registry.
* **Python Libraries**: For building and training the machine learning models.

---
