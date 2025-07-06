import os
import joblib
import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


SENTIMENT_MODEL_FILE = 'sentiment_model.pkl'
SPAM_MODEL_FILE = 'spam_model.pkl'

SAMPLES_FOR_TRAIN = 1000
MODEL_ITERATIONS = 1000

FILM_REVIEW_PATH = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
DATASET_REVIEW_FILE = os.path.join(FILM_REVIEW_PATH, 'IMDB Dataset.csv')

SPAM_PATH = kagglehub.dataset_download("abdallahwagih/spam-emails")
DATASET_SPAM_FILE = os.path.join(SPAM_PATH, 'spam.csv')


class NLPModel:
    def __init__(self):
        # Initialize tokenizer and bert_model once to be reused
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        self.bert_model = TFBertModel.from_pretrained('prajjwal1/bert-tiny', from_pt=True)

        if os.path.exists(SENTIMENT_MODEL_FILE):
            self.sentiment_model = joblib.load(SENTIMENT_MODEL_FILE)
        else:
            self.sentiment_model = self.init_sentiment_model()
            joblib.dump(self.sentiment_model, SENTIMENT_MODEL_FILE)


        if os.path.exists(SPAM_MODEL_FILE):
            self.spam_model = joblib.load(SPAM_MODEL_FILE)
        else:
            self.spam_model = self.init_spam_model()
            joblib.dump(self.spam_model, SPAM_MODEL_FILE)

    def init_sentiment_model(self):
        dataset = pd.read_csv(DATASET_REVIEW_FILE)
        dataset = dataset.sample(n=SAMPLES_FOR_TRAIN, random_state=42)
        dataset['review'] = dataset['review'].apply(lambda row: row.lower())

        # Get embeddings for the training data
        X = self._get_bert_embeddings(dataset['review'])
        y = dataset['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        log_reg = LogisticRegression(max_iter=MODEL_ITERATIONS)
        log_reg.fit(X_train, y_train)
        
        # Optional: You can evaluate your model's accuracy
        accuracy = log_reg.score(X_test, y_test)
        print(f"Model sentiment accuracy: {accuracy}")

        return log_reg
    
    def init_spam_model(self):
        dataset = pd.read_csv(DATASET_SPAM_FILE)
        dataset = dataset.sample(n=SAMPLES_FOR_TRAIN, random_state=42)
        dataset['Message'] = dataset['Message'].apply(lambda row: row.lower())

        # Get embeddings for the training data
        X = self._get_bert_embeddings(dataset['Message'])
        y = dataset['Category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        log_reg = LogisticRegression(max_iter=MODEL_ITERATIONS)
        log_reg.fit(X_train, y_train)
        
        # Optional: You can evaluate your model's accuracy
        accuracy = log_reg.score(X_test, y_test)
        print(f"Model spam accuracy: {accuracy}")

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
        return outputs.last_hidden_state[:, 0, :].numpy()

    def _get_embedding_for_message(self, message):
        processed_message = message.lower()
        tokenized_message = self._tokenize_text([processed_message])
        outputs = self.bert_model(tokenized_message)
        return outputs.last_hidden_state[:, 0, :].numpy()

    def predict_answer(self, message, mode):
        if mode == 'sentiment':
            return self.predict_sentiment(message)
        elif mode == 'spam':
            return self.predict_spam(message)
        elif mode == 'rag':
            return self.predict_rag(message)

        return None

    def predict_sentiment(self, message):
        message_embedding = self._get_embedding_for_message(message)
        predicted_result = self.sentiment_model.predict(message_embedding)[0]
        predicted_result_proba = self.sentiment_model.predict_proba(message_embedding)[0]

        return {
            'message': predicted_result,
            'Negative': "{:.2%}".format(predicted_result_proba[0]),
            'Positive': "{:.2%}".format(predicted_result_proba[1]),
        }

    def predict_spam(self, message):
        message_embedding = self._get_embedding_for_message(message)
        predicted_result = self.spam_model.predict(message_embedding)[0]
        predicted_result_proba = self.spam_model.predict_proba(message_embedding)[0]

        return {
            'message': 'Not Spam' if predicted_result == 'ham' else 'Spam',
            'Not Spam': "{:.2%}".format(predicted_result_proba[0]),
            'Spam': "{:.2%}".format(predicted_result_proba[1]),
        }
    
    def predict_rag(self, message):
        return {
            'message': 'yea',
        }