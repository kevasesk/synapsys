import os
import joblib

MODEL_FILE = 'sentiment_analysys_model.pkl'

class NLPModel:
    def __init__(self):
        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
        else:
            self.model = self.init_model()
            joblib.dump(self.model, MODEL_FILE)

    def init_model(self):
        pass

    def predict_message(self):
        return 'yeap'