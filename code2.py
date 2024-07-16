from mrjob.job import MRJob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import string

class TextClassificationMR(MRJob):

    def mapper(self, _, line):
        # Parse the input and extract text and class label
        text, label = line.split(",")[0], int(line.split(",")[1])

        # Preprocess the text
        text = self.preprocess(text)

        yield label, text

    def reducer(self, label, texts):
        # Initialize TfidfVectorizer
        vectorization = TfidfVectorizer()

        # Vectorize the text data
        xv_train = vectorization.fit_transform(texts)

        # Split data into features and target
        x_train = xv_train
        y_train = [label] * len(texts)

        # Train classifiers
        LR = LogisticRegression()
        LR.fit(x_train, y_train)

        NB = MultinomialNB()
        NB.fit(x_train, y_train)

        # Emit the trained models
        yield label, (LR, NB)

    def preprocess(self, text):
        # Your data preprocessing steps here
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W", " ", text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

if __name__ == '__main__':
    TextClassificationMR.run()
