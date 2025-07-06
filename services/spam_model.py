import os
import kagglehub
from .base_model import BaseModel

SPAM_MODEL_FILE = 'spam_model.pkl'
SPAM_PATH = kagglehub.dataset_download("abdallahwagih/spam-emails")
DATASET_SPAM_FILE = os.path.join(SPAM_PATH, 'spam.csv')

class SpamModel(BaseModel):
    def __init__(self):
        super().__init__(
            model_file=SPAM_MODEL_FILE,
            dataset_path=DATASET_SPAM_FILE,
            text_column='Message',
            label_column='Category'
        )

    def predict(self, message):
        message_embedding = self._get_embedding_for_message(message)
        predicted_result = self.model.predict(message_embedding)[0]
        predicted_result_proba = self.model.predict_proba(message_embedding)[0]

        return {
            'message': 'Not Spam' if predicted_result == 'ham' else 'Spam',
            'Not Spam': "{:.2%}".format(predicted_result_proba[0]),
            'Spam': "{:.2%}".format(predicted_result_proba[1]),
        }
