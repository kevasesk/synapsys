import os
import joblib
import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


MODEL_FILE = 'sentiment_analysys_model.pkl'
SAMPLES_FOR_TRAIN = 1000
MODEL_ITERATIONS = 1000
FILM_REVIEW_PATH = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
DATASET_FILE = os.path.join(FILM_REVIEW_PATH, 'IMDB Dataset.csv')


class NLPModel:
    def __init__(self):
        # Initialize tokenizer and bert_model once to be reused
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        self.bert_model = TFBertModel.from_pretrained('prajjwal1/bert-tiny', from_pt=True)
        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
        else:
            self.model = self.init_model()
            joblib.dump(self.model, MODEL_FILE)

    def init_model(self):
        dataset = pd.read_csv(DATASET_FILE)
        dataset = dataset.sample(n=SAMPLES_FOR_TRAIN, random_state=42)
        dataset['review'] = dataset['review'].apply(lambda row: row.lower())
        dataset['review'] = dataset['review'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

        # Get embeddings for the training data
        X = self._get_bert_embeddings(dataset['review'])
        y = dataset['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        log_reg = LogisticRegression(max_iter=MODEL_ITERATIONS)
        log_reg.fit(X_train, y_train)
        
        # Optional: You can evaluate your model's accuracy
        accuracy = log_reg.score(X_test, y_test)
        print(f"Model accuracy: {accuracy}")

        return log_reg

    def _tokenize_text(self, text_list):
        return self.tokenizer(
            text_list,
            max_length=50,
            truncation=True,
            padding='max_length',
            return_tensors='tf'
        )
    
    def _get_bert_embeddings(self, text_series):
        tokenized = self._tokenize_text(text_series.tolist())
        outputs = self.bert_model(tokenized)
        # The embedding is the output of the [CLS] token
        return outputs.last_hidden_state[:, 0, :].numpy()

    def predict_message(self, message):
        """
        Predicts the sentiment of a single message.
        """

        # 1. Preprocess the text (lowercase, remove special chars)
        processed_message = message.lower()

        # 2. Tokenize the processed message. Note that it expects a list.
        tokenized_message = self._tokenize_text([processed_message])
        
        # 3. Get BERT embeddings
        outputs = self.bert_model(tokenized_message)
        message_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        probability_info = {
            'negative': "{:.2%}".format(self.model.predict_proba(message_embedding)[0][0]),
            'positive': "{:.2%}".format(self.model.predict_proba(message_embedding)[0][1]),
        }

        # 4. Predict using the logistic regression model
        return probability_info, self.model.predict(message_embedding)[0]