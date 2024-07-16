import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

# Load datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels
df_fake["label"] = 0
df_true["label"] = 1

# Concatenate datasets
df_combined = pd.concat([df_fake, df_true])

# Language model fine-tuning for news generation
def fine_tune_language_model(df, model_name="gpt2-medium", epochs=3):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Prepare data for fine-tuning
    text = "\n".join(df["text"].tolist())
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Fine-tune the language model
    model.train()
    model.resize_token_embeddings(len(tokenizer))
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_ids.to("cuda"), labels=input_ids.to("cuda"))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

    # Save fine-tuned model
    model.save_pretrained("fine_tuned_model")

# Fine-tune language model
fine_tune_language_model(df_combined)

# Classification model training
def train_classification_model(df):
    # Data preprocessing
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text

    df["text"] = df["text"].apply(preprocess_text)

    # Split data into features and target
    x = df["text"]
    y = df["label"]

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    # Train classifiers
    lr_classifier = LogisticRegression()
    lr_classifier.fit(xv_train, y_train)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(xv_train, y_train)

    # Evaluate classifiers
    lr_pred = lr_classifier.predict(xv_test)
    nb_pred = nb_classifier.predict(xv_test)

    lr_accuracy = accuracy_score(y_test, lr_pred)
    nb_accuracy = accuracy_score(y_test, nb_pred)

    lr_report = classification_report(y_test, lr_pred)
    nb_report = classification_report(y_test, nb_pred)

    print("Logistic Regression Accuracy:", lr_accuracy)
    print("Naive Bayes Accuracy:", nb_accuracy)

    print("Logistic Regression Classification Report:")
    print(lr_report)

    print("Naive Bayes Classification Report:")
    print(nb_report)

# Train classification model
train_classification_model(df_combined)
