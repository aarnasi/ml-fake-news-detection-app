# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text data into numerical features (TF-IDF vectors)
from sklearn.linear_model import LogisticRegression  # For building the classification model
from sklearn.pipeline import Pipeline  # For chaining multiple steps (vectorizer and classifier) together
from sklearn.metrics import classification_report  # For evaluating the model performance
import joblib  # For saving the trained model to a file

# Import the custom data loading function
from data_loader import load_and_prepare_data

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Data Loading and Preparation ---

# Load the dataset using the custom function and prepare it for modeling
# Assumes 'load_and_prepare_data' handles reading the CSV and any initial cleaning/preprocessing
df = load_and_prepare_data("data/train.csv")

# Define the features (input text content) and the target variable (labels)
X = df['content']  # Input features (news article content)
y = df['label']    # Target variable (e.g., 'fake' or 'real')

# --- Data Splitting ---

# Split the data into training and testing sets
# X: features, y: target variable
# test_size=0.2: 20% of the data will be used for testing, 80% for training
# random_state=42: Ensures reproducibility of the split (the same split will occur each time the code is run)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Building (Pipeline) ---
#The pipeline first transforms X_train using TfidfVectorizer and then trains LogisticRegression on the transformed
# data and y_train.

# Create a machine learning pipeline
pipeline = Pipeline([
    # Step 1: TF-IDF Vectorizer
    # Converts text documents to a matrix of TF-IDF features.
    # stop_words='english': Removes common English words (like 'the', 'a', 'is') that don't usually add much meaning.
    # max_df=0.7: Ignores terms that appear in more than 70% of the documents (likely too common).
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),

    # Step 2: Logistic Regression Classifier
    # A linear model for classification.
    ('clf', LogisticRegression(max_iter=1000))
])

# --- Model Training ---

# Train the entire pipeline (TF-IDF transformation and Logistic Regression) on the training data
logging.info("Training model ..")
pipeline.fit(X_train, y_train)

# --- Model Evaluation ---

# Make predictions on the unseen test data using the trained pipeline The pipeline automatically applies the same
# TF-IDF transformation learned from the training data to X_test before predicting.
logging.info("Evaluating model ..")
y_pred = pipeline.predict(X_test)

# Print a classification report to evaluate the model's performance
# Compares the true labels (y_test) with the predicted labels (y_pred)
# Shows metrics like precision, recall, F1-score, and support for each class.
logging.info("--- Classification Report ---")
logging.info(classification_report(y_test, y_pred))
logging.info("-----------------------------")

# --- Model Saving ---

# Save the trained pipeline (including the vectorizer and the classifier) to a file
# This allows the model to be loaded and used later for predictions without retraining.
# 'model/fake_news_model.pkl' is the path where the model file will be saved.
joblib.dump(pipeline, 'saved_model/fake_news_model.pkl')

logging.info("✅ Model saved to saved_model/fake_news_model.pkl")