# 📰 Fake News Detection App

A machine learning-based web application that detects fake news articles using a Scikit-learn pipeline. The model is trained using traditional NLP techniques and deployed on **Google Cloud Run** for scalable, serverless inference.

## 🚀 Features

- Detects fake vs. real news from user input
- Built with **Scikit-learn** and **Python**
- Trained using classical ML techniques (TF-IDF + Logistic Regression)
- Deployed using **Docker** on **Google Cloud Run**
- Frontend built with Streamlit for fast prototyping and interactivity
- CI/CD ready with **Cloud Build**
- Logging and monitoring integrated with **Cloud Logging**

## 🧠 Model Overview

- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classifier**: Logistic Regression
- **Dataset**: [Fake and real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Evaluation**: Accuracy, precision, recall, and F1-score

## 📦 Tech Stack

| Component        | Tool/Library     |
|------------------|------------------|
| ML Model         | Scikit-learn (TfidfVectorizer + Logistic Regression) |
| Backend API      | FastAPI, Uvicorn |
| Frontend         | Streamlit        |
| Containerization | Docker           |
| Deployment       | Google Cloud Run |
| CI/CD            | Cloud Build      |
| Monitoring       | Cloud Logging    |

## 📁 Project Structure

```
ml-fake-news-detection-app/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── Dockerfile          # Docker configuration for backend
│   ├── requirements.txt    # Backend dependencies
│   ├── model/              # Pre-trained model directory
│   └── scripts/
│       └── deploy_backend.sh  # Deployment script
├── frontend/
│   ├── app.py              # Streamlit frontend application
│   └── requirements.txt    # Frontend dependencies
├── model/
│   ├── train_model.py      # Model training script
│   ├── data_loader.py      # Data loading utilities
│   ├── data/
│   │   └── train.csv       # Training dataset
│   └── saved_model/
│       └── fake_news_model.pkl  # Trained model file
└── README.md
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.11+
- pip
- Docker (for containerization)
- Google Cloud SDK (for GCP deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/aarnasi/ml-fake-news-detection-app.git
cd ml-fake-news-detection-app
```

### 2. Local Development Setup

#### Model Training

First, train the model:

```bash
cd model
python3 -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

This will:
- Load and preprocess the training data
- Train a TF-IDF + Logistic Regression pipeline
- Evaluate the model and save it to `saved_model/fake_news_model.pkl`

#### Backend Setup

1. Copy the trained model to the backend directory:

```bash
cd ../backend
cp -r ../model/saved_model ./model
```

2. Set up and run the backend:

```bash
python3 -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`

#### Frontend Setup

In a new terminal:

```bash
cd frontend
python3 -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`

**Note**: Make sure the backend is running before starting the frontend, as the frontend makes API calls to the backend.

## 🔌 API Documentation

The backend exposes the following endpoints:

### Health Check

```http
GET /health
```

Returns the health status of the API and verifies model availability.

**Response:**
```json
{
  "status": "healthy"
}
```

### Predict

```http
POST /predict
```

Accepts a news item's title and text and returns a prediction.

**Request Body:**
```json
{
  "title": "News article title",
  "text": "News article content"
}
```

**Response:**
```json
{
  "prediction": 1
}
```

**Prediction Values:**
- `1` = Fake news
- `0` = Genuine news

### Interactive API Documentation

FastAPI provides automatic interactive API documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## ☁️ Google Cloud Platform (GCP) Deployment

### Prerequisites

- A billable GCP project
- Google Cloud SDK installed and configured

### 1. Authenticate CLI

```bash
gcloud auth application-default login
```

### 2. Add Necessary Permissions

```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:USER_EMAIL" \
    --role="roles/artifactregistry.writer"
```

### 3. Create Artifact Repository

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev

gcloud artifacts repositories create REPOSITORY_NAME \
    --repository-format=docker \
    --location=LOCATION \
    --description="Docker repository for fake news detection backend"
```

### 4. Deploy Backend to Cloud Run

1. Populate `backend/config/.gcp_env_vars` with your GCP variables.

2. Run the deployment script:

```bash
cd backend/scripts
/bin/bash deploy_backend.sh
```

The script will:
- Build the Docker image
- Push it to Artifact Registry
- Deploy to Cloud Run

### 5. Deploy Frontend to Streamlit

Deploy the frontend to [Streamlit Cloud](https://streamlit.io/) using your Streamlit account.

**Note**: Update the API endpoint in `frontend/app.py` to point to your Cloud Run service URL instead of `http://127.0.0.1:8000`.

## 🧪 Usage

1. Start the backend API server
2. Start the Streamlit frontend
3. Enter a news article title and content in the web interface
4. Click "Detect" to get the prediction

The model will analyze the text and classify it as either genuine or fake news.

## 📊 Model Training Details

The model uses:
- **TF-IDF Vectorization**: Converts text to numerical features, removing English stop words and terms appearing in more than 70% of documents
- **Logistic Regression**: Binary classifier with max_iter=1000
- **Train/Test Split**: 80/20 with random_state=42 for reproducibility

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Fake and real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Built with Scikit-learn, FastAPI, and Streamlit

