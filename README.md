# 📰 Fake News Detection App

A machine learning-based web application that detects fake news articles using a Scikit-learn pipeline. The model is trained using traditional NLP techniques and deployed on **Google Cloud Run** for scalable, serverless inference.

## 🚀 Features

- Detects fake vs. real news from user input
- Built with **Scikit-learn** and **Python**
- Trained using classical ML techniques (Tfidf + Logistic Regression)
- Deployed using **Docker** on **Google Cloud Run**
- Frontend built with Streamlit for fast prototyping and interactivity
- CI/CD ready with **Cloud Build**
- Logging and monitoring integrated with **Cloud Logging**

---

## 🧠 Model Overview

- **Vectorizer**: TF-IDF
- **Classifier**: Logistic Regression
- **Dataset**: [Fake and real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Evaluation**: Accuracy, precision, recall, and F1-score

---

## 📦 Tech Stack

| Component        | Tool/Library     |
|------------------|------------------|
| ML Model         | Scikit-learn (TfidfVectorizer + Logistic Regression)     |
| Web Framework    |  Python, Flask (for API)       |
| Frontend  | Streamlit                    |
| Containerization | Docker           |
| Deployment       | Google Cloud Run |
| CI/CD            | Cloud Build      |
| Monitoring       | Cloud Logging    |

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/aarnasi/ml-fake-news-detection-app.git
cd fake-news-detector
```
### 2. Local setup
    
 Model Training:
```bash
        cd model
        python3 -m venv .env
        source .env/bin/activate
        pip install -r requirements.txt
        python train_model.py
```
    
Backend Setup
```bash
        cd ../backend
        cp -r ../model/saved_model ./model
        python3 -m venv .env
        source .env/bin/activate
        pip install -r requirements.txt
        uvicorn app:app
```
    
Front-end setup
```bash
        cd ../front-end
        python3 -m venv .env
        source .env/bin/activate
        pip install -r requirements.txt
        streamlit run app.py
```

### 3. GCP setup
Prerequisites:
    
- A billable GCP project.

1. Authenticate CLI to use GCP services
```commandline
gloud auth application-default login
```
2. Add necessary permissions
```commandline
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:USER_EMAIL" \
    --role="roles/artifactregistry.writer"
```
3. Create Artifact Repository on GCP for storing backend image
```commandline
gcloud auth configure-docker \
    us-central1-docker.pkg.dev
gcloud artifacts repositories create REPOSITORY_NAME \
    --repository-format=docker \
    --location=LOCATION \
    [--description="DESCRIPTION"] \
    [--async]
```

#### Deploy backend to GCP

1. Populate backend/config/.gcp_env_vars with your own GCP variables.

2. Start deployment to GCP
```commandline
   cd /backend/scripts
   /bin/bash scripts/deploy_backend.sh
```

#### Deploy frontend to [Streamlit](https://streamlit.io/)

1. Deploy to Streamlit using your Streamlit account.