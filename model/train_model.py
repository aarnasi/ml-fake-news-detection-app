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

# Create a machine learning pipeline
# A pipeline sequentially applies a list of transforms and a final estimator.
# This makes the workflow cleaner and prevents data leakage from the test set during transformation.
pipeline = Pipeline([
    # Step 1: TF-IDF Vectorizer
    # Converts text documents to a matrix of TF-IDF features.
    # stop_words='english': Removes common English words (like 'the', 'a', 'is') that don't usually add much meaning.
    # max_df=0.7: Ignores terms that appear in more than 70% of the documents (likely too common).
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),

    # Step 2: Logistic Regression Classifier
    # A linear model for classification.
    # max_iter=1000: Increases the maximum number of iterations for the solver to converge, useful for complex datasets.
    ('clf', LogisticRegression(max_iter=1000))
])

# --- Model Training ---

# Train the entire pipeline (TF-IDF transformation and Logistic Regression) on the training data The pipeline first
# transforms X_train using TfidfVectorizer and then trains LogisticRegression on the transformed data and y_train.
pipeline.fit(X_train, y_train)

# --- Model Evaluation ---

# Make predictions on the unseen test data using the trained pipeline The pipeline automatically applies the same
# TF-IDF transformation learned from the training data to X_test before predicting.
y_pred = pipeline.predict(X_test)

# Print a classification report to evaluate the model's performance
# Compares the true labels (y_test) with the predicted labels (y_pred)
# Shows metrics like precision, recall, F1-score, and support for each class.
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("-----------------------------")

# --- Model Saving ---

# Save the trained pipeline (including the vectorizer and the classifier) to a file
# This allows the model to be loaded and used later for predictions without retraining.
# 'model/fake_news_model.pkl' is the path where the model file will be saved.
joblib.dump(pipeline, 'backend/model/fake_news_model.pkl')

# Print a confirmation message indicating that the model has been saved successfully.
print("✅ Model saved to model/fake_news_model.pkl")