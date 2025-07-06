import os
import kagglehub
from .base_model import BaseModel

SENTIMENT_MODEL_FILE = 'sentiment_model.pkl'
FILM_REVIEW_PATH = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
DATASET_REVIEW_FILE = os.path.join(FILM_REVIEW_PATH, 'IMDB Dataset.csv')

class SentimentModel(BaseModel):
    def __init__(self):
        super().__init__(
            model_file=SENTIMENT_MODEL_FILE,
            dataset_path=DATASET_REVIEW_FILE,
            text_column='review',
            label_column='sentiment'
        )

    def predict(self, message):
        message_embedding = self._get_embedding_for_message(message)
        predicted_result = self.model.predict(message_embedding)[0]
        predicted_result_proba = self.model.predict_proba(message_embedding)[0]

        return {
            'message': predicted_result,
            'Negative': "{:.2%}".format(predicted_result_proba[0]),
            'Positive': "{:.2%}".format(predicted_result_proba[1]),
        }
