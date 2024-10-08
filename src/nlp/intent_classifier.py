import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class IntentClassifier:
    def __init__(self, model_name='bert-base-uncased', max_len=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.intents = self.load_intents()

    def load_intents(self):
        with open('data/intents.json', 'r') as f:
            return json.load(f)

    def prepare_data(self):
        texts = []
        labels = []
        for intent in self.intents['intents']:
            texts.extend(intent['patterns'])
            labels.extend([intent['tag']] * len(intent['patterns']))

        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)

        return train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

    def train(self, epochs=10, batch_size=16):
        train_texts, val_texts, train_labels, val_labels = self.prepare_data()

        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer, self.max_len)
        val_dataset = IntentDataset(val_texts, val_labels, self.tokenizer, self.max_len)

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_encoder.classes_),
            output_attentions=False,
            output_hidden_states=False
        )
        self.model = self.model.to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            self._train_epoch(train_data_loader, optimizer)
            self._evaluate(val_data_loader)

    def _train_epoch(self, data_loader, optimizer):
        self.model.train()
        losses = []
        correct_predictions = 0
        total_predictions = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            total_predictions += len(labels)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Train loss {np.mean(losses)} accuracy {correct_predictions.double() / total_predictions}')

    def _evaluate(self, data_loader):
        self.model.eval()
        losses = []
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                _, preds = torch.max(logits, dim=1)

                correct_predictions += torch.sum(preds == labels)
                total_predictions += len(labels)

                losses.append(loss.item())

        print(f'Val loss {np.mean(losses)} accuracy {correct_predictions.double() / total_predictions}')

    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        _, preds = torch.max(outputs.logits, dim=1)
        return self.label_encoder.inverse_transform(preds.cpu().numpy())[0]

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
