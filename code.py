import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Load the datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels
df_fake["class"] = 0
df_true["class"] = 1

# Remove last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
df_fake = df_fake.iloc[:-10]

df_true_manual_testing = df_true.tail(10)
df_true = df_true.iloc[:-10]

# Combine manual testing data
df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv")

# Concatenate the datasets
df_merge = pd.concat([df_fake, df_true], axis=0)

# Drop unnecessary columns
df = df_merge.drop(["title", "subject", "date"], axis=1)


# Data preprocessing1
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


df["text"] = df["text"].apply(wordopt)

# Split data into features and target
x = df["text"]
y = df["class"]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Vectorize the text data
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
lr_accuracy = accuracy_score(y_test, pred_lr)
lr_report = classification_report(y_test, pred_lr, output_dict=True)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, pred_lr))

# Naive Bayes
NB = MultinomialNB()
NB.fit(xv_train, y_train)
pred_nb = NB.predict(xv_test)
nb_accuracy = accuracy_score(y_test, pred_nb)
nb_report = classification_report(y_test, pred_nb, output_dict=True)
print("Naive Bayes Accuracy:", accuracy_score(y_test, pred_nb))
print("Naive Bayes Classification Report:")
print(classification_report(y_test, pred_nb))

# Compare the two classifiers
print("Logistic Regression Accuracy:", lr_accuracy)
print("Naive Bayes Accuracy:", nb_accuracy)

# Plot accuracy comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=["Logistic Regression", "Naive Bayes"], y=[lr_accuracy, nb_accuracy])
plt.title("Accuracy Comparison")
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.show()

# Plot classification report metrics comparison for Logistic Regression
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(lr_report).iloc[:-1, :].T, annot=True, cmap="coolwarm")
plt.title("Logistic Regression Classification Report")
plt.show()

# Plot classification report metrics comparison for Naive Bayes
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(nb_report).iloc[:-1, :].T, annot=True, cmap="coolwarm")
plt.title("Naive Bayes Classification Report")
plt.show()

# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)
    pred_NB = NB.predict(new_xv_test)

    print("\n\nLogistic Regression Prediction:", output_lable(pred_LR[0]))
    print("Naive Bayes Prediction:", output_lable(pred_NB[0]))


# Helper function for output label
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"


# Manual testing
news = str(input("Enter news text for manual testing: "))
manual_testing(news)
news = str(input("Enter news text for manual testing: "))
manual_testing(news)