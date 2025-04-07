# Builds docker image, tags and deploys to Cloud Run
# Set environment variables
source ../config/.gcp_env_vars

# Enable required services
gcloud services enable run artifactregistry.googleapis.com

# Create Artifact Registry repo (if not already done)
gcloud artifacts repositories create REPO_NAM \
  --repository-format=docker \
  --location=$REGION

# Build Docker image
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/REPO_NAM/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/REPO_NAM/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated
