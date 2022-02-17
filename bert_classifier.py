import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from CodeBert import *
from bert_dataset import CustomDataset

class CodeBertClassifier:
    def __init__(self, model_path, tokenizer_path, max_len=256, n_classes=2, epochs=15,
                 model_save_path='/save/codebert.pt', batch_size = 64, learning_rate = 2e-5):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels = n_classes)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, config=self.config)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer, max_len=self.max_len)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer, max_len=self.max_len)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.loss_fn(outputs.logits, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()

            progress_bar.set_description(
                f'loss: {loss.item():.3f}')
        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), self.model_save_path)
                best_accuracy = val_acc

        self.model.load_state_dict(torch.load(self.model_save_path))

    def predict(self, text):

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction

class EL_CodeBert:
    def __init__(self, model_path, tokenizer_path, pretrained_model, max_len=256, n_classes=2, epochs=15,
                 model_save_path='/save/codebert.pt', batch_size = 64, learning_rate = 5e-4):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels = n_classes)
        self.model = EL_CodeBert(config=self.config)
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(pretrained_model)
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        for k,v in self.model.named_parameters():
            if('roberta' in k):
                v.requires_grad = False
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer, max_len=self.max_len)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer, max_len=self.max_len)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.loss_fn(outputs.logits, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()

            progress_bar.set_description(
                f'loss: {loss.item():.3f}')
        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), self.model_save_path)
                best_accuracy = val_acc

        self.model.load_state_dict(torch.load(self.model_save_path))

    def predict(self, text):

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction

    def single_predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)
        # print(self.model)
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )
        # print(outputs.attentions[0].squeeze(1).shape)
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        alpha = outputs.attentions[0].squeeze(1).detach().cpu().numpy()
        return prediction, alpha

class EL_CodeBert_wo_LSTM:
    def __init__(self, model_path, tokenizer_path, pretrained_model=None, max_len=256, n_classes=2, epochs=15,
                 model_save_path='/save/codebert.pt', batch_size=64, learning_rate=5e-4):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels=n_classes)
        self.model = EL_CodeBert_wo_LSTM(config=self.config)
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(pretrained_model)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        for k, v in self.model.named_parameters():
            if ('roberta' in k):
                v.requires_grad = False
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer, max_len=self.max_len)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer, max_len=self.max_len)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.loss_fn(outputs.logits, targets)

            self.scheduler.step()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            progress_bar.set_description(
                f'loss: {loss.item():.3f}')
        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model, self.model_save_path)
                best_accuracy = val_acc

        self.model = torch.load(self.model_save_path)

    def predict(self, text):

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction

class EL_CodeBert_wo_Attention:
    def __init__(self, model_path, tokenizer_path, pretrained_model=None, max_len=256, n_classes=2, epochs=15,
                 model_save_path='/save/codebert.pt', batch_size=64, learning_rate=5e-4):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels=n_classes)
        self.model = EL_CodeBert_wo_Attention(config=self.config)
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(pretrained_model)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        for k, v in self.model.named_parameters():
            if ('roberta' in k):
                v.requires_grad = False
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer, max_len=256)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer, max_len=256)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.loss_fn(outputs.logits, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()

            progress_bar.set_description(
                f'loss: {loss.item():.3f}')
        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model, self.model_save_path)
                best_accuracy = val_acc

        self.model = torch.load(self.model_save_path)

    def predict(self, text):

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction
