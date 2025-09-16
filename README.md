# Wildfire API – CI/CD with Google Cloud Run

This project contains a FastAPI service for wildfire prediction and imputation, deployed automatically to **Google Cloud Run** using **Google Cloud Build**.

---

## Deployment Workflow

### Branching Strategy
- **`dev` branch**  
  - Used for development and testing.  
  - Linked to a separate Cloud Build trigger (optional) that deploys to a staging Cloud Run service.  
  - Allows validation of new endpoints and features before merging.  

- **`main` branch**  
  - Production-ready branch.  
  - Cloud Build Trigger deploys automatically when new commits are pushed.  
  - Deploys to the production Cloud Run service.  

**Flow:**  
```text
   [dev branch]  --->  Test deployment (staging)
        |
        v
   Merge into main  --->  Production deployment (Cloud Run)
```

### CI/CD Pipeline (for `main`)
1. **Push Code to GitHub (`main` branch)**  
   - Cloud Build Trigger fires automatically.  
   - Builds Docker image.  
   - Pushes image to Artifact Registry.  
   - Deploys service to Cloud Run.  

2. **Service is Live**  
   - Accessible at your Cloud Run service URL.  
   - Example:  
     ```
     https://wildfire-api-575281815403.us-central1.run.app/docs#
     ```

---

## Requirements

- GitHub repository: [Jithendra-k/wildfire_api](https://github.com/Jithendra-k/wildfire_api)  
- Google Cloud Project: `code-for-planet`  
- Artifact Registry Repository: `wildfire-repo`  
- Cloud Build Trigger: `wildfire-api` (push to `main`)  
- Cloud Run Service: `wildfire-api` (region: `us-central1`)  

---

## Files

- **`Dockerfile`** – Defines container build (Python 3.12 + FastAPI + requirements).  
- **`requirements.txt`** – Python dependencies.  
- **`cloudbuild.yaml`** – CI/CD pipeline configuration:
  - Step 1: Build Docker image.  
  - Step 2: Push to Artifact Registry.  
  - Step 3: Deploy to Cloud Run.  

---

## Manual Deploy (Optional)

If you want to deploy manually (instead of waiting for a trigger):

```bash
# Build and push image
docker build -t us-central1-docker.pkg.dev/code-for-planet/wildfire-repo/wildfire-api:latest .
docker push us-central1-docker.pkg.dev/code-for-planet/wildfire-repo/wildfire-api:latest

# Deploy to Cloud Run
gcloud run deploy wildfire-api \
  --image us-central1-docker.pkg.dev/code-for-planet/wildfire-repo/wildfire-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4
```
