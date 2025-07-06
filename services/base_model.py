import os
import joblib
from transformers import TFBertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

class BaseModel:
    def __init__(self, model_file, dataset_path, text_column, label_column, model_iterations=1000, samples_for_train=1000):
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        self.bert_model = TFBertModel.from_pretrained('prajjwal1/bert-tiny', from_pt=True)
        self.model_file = model_file
        self.dataset_path = dataset_path
        self.text_column = text_column
        self.label_column = label_column
        self.model_iterations = model_iterations
        self.samples_for_train = samples_for_train

        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
        else:
            self.model = self._train_model()
            joblib.dump(self.model, self.model_file)

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

    def _train_model(self):
        dataset = pd.read_csv(self.dataset_path)
        dataset = dataset.sample(n=self.samples_for_train, random_state=42)
        dataset[self.text_column] = dataset[self.text_column].apply(lambda row: row.lower())

        X = self._get_bert_embeddings(dataset[self.text_column])
        y = dataset[self.label_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        log_reg = LogisticRegression(max_iter=self.model_iterations)
        log_reg.fit(X_train, y_train)
        
        accuracy = log_reg.score(X_test, y_test)
        print(f"Model accuracy for {self.model_file}: {accuracy}")

        return log_reg

    def predict(self, message):
        raise NotImplementedError("This method should be overridden by subclasses.")
