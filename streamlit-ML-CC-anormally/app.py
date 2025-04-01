import os
import pandas as pd
import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from autotrain.params import TextClassificationParams
from autotrain.project import AutoTrainProject
import torch

# Load secrets
HF_USERNAME = st.secrets["HF_USERNAME"]
HF_TOKEN = st.secrets["HF_TOKEN"]

# Set environment variables (for Hugging Face authentication)
os.environ["HF_USERNAME"] = HF_USERNAME
os.environ["HF_TOKEN"] = HF_TOKEN

# Function to load and process CSV
def load_csv(file):
    """ Load and process CSV file """
    df = pd.read_csv(file)
    if 'description' not in df.columns or 'category' not in df.columns:
        st.error("CSV file must contain 'description' and 'category' columns.")
        return None
    df = df[['description', 'category']].dropna().drop_duplicates()
    df = df.rename(columns={'description': 'text', 'category': 'target'})
    return df

# Function to save processed data
def save_processed_data(df):
    """ Save processed data for AutoTrain """
    processed_file = "processed_data.csv"
    df.to_csv(processed_file, index=False)
    return processed_file

# Function to train model using uploaded dataset
def train_model(data_path, model_save_path="trained_model"):
    """ Train a text classification model using AutoTrain and save it locally """
    params = TextClassificationParams(
        model="bert-base-uncased",
        data_path=data_path,
        text_column="text",
        target_column="target",
        train_split="train",
        epochs=3,
        batch_size=8,
        lr=5e-5,
        optimizer="adamw_torch",
        scheduler="linear",
        mixed_precision="bf16",
        project_name="autotrain-llama32-1b-finetune",
        log="tensorboard",
        push_to_hub=False,  # Save locally instead of pushing to Hugging Face
    )

    backend = "local"
    project = AutoTrainProject(params=params, backend=backend, process=True)
    project.create()

    # Save trained model locally
    os.makedirs(model_save_path, exist_ok=True)
    project.save(model_save_path)
    st.success(f"🎉 Model training complete! Model saved to: {model_save_path}")

# Function to load trained model dynamically
def load_model(model_path="trained_model"):
    """ Load trained model from local storage """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to predict new descriptions
def predict(classifier, texts):
    """ Predict categories for unclassified texts """
    predictions = classifier(texts)
    return [(text, pred['label'], pred['score']) for text, pred in zip(texts, predictions)]

# Streamlit UI
st.title("Text Classification App 📝")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File (description, category)", type=['csv'])

if uploaded_file:
    df = load_csv(uploaded_file)
    if df is not None:
        st.write("✅ Data Preview:")
        st.dataframe(df.head())

        # Train Model
        if st.button("Train Model"):
            processed_file = save_processed_data(df)
            st.write("⏳ Training model... This may take a while.")
            train_model(processed_file)
            st.success("🎉 Model training complete!")

# Prediction Section
st.subheader("Test Model with New Data")
test_texts = st.text_area("Enter texts (one per line)", "This is a great phone.\nI love this restaurant.")

if st.button("Predict"):
    classifier = load_model()
    if classifier:
        input_texts = test_texts.split("\n")
        results = predict(classifier, input_texts)

        for text, category, score in results:
            st.write(f"**Text:** {text}")
            st.write(f"**Predicted Category:** {category} (Confidence: {score:.4f})")
            st.write("---")
