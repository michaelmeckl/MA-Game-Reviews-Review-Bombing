import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import evaluate
from torch import nn
from transformers import BertModel


# Taken from https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b and https://wellsr.com/python/fine-tuning-bert-for-sentiment-analysis-with-pytorch/
class BERTClassifier(nn.Module):
    def __init__(self, num_classes, model_checkpoint):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_classes)
        """
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )
        """

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        # x = outputs['last_hidden_state'][:, 0, :]
        logits = self.fc(x)
        return logits


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, 10)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(10, 8)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(8, 1)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Sigmoid()

    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X


def train_model(model, data_loader, optimizer, scheduler, criterion, device, progress_bar):
    size = len(data_loader.dataset)
    model.train()  # set the model in training mode
    for batch_number, batch in enumerate(data_loader):
        optimizer.zero_grad()  # reset the gradients after every batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels = labels.type(torch.LongTensor)  # make sure it is a Long Tensor as CrossEntropy loss requires this
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels.squeeze())  # squeeze label tensor to remove the outer dimension
        loss.backward()  # backpropagate the loss and compute the gradients
        optimizer.step()  # update the model weights
        scheduler.step()

        progress_bar.update(1)
        if batch_number % 100 == 0:
            loss, current = loss.item(), (batch_number + 1) * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate_model(model, data_loader, device):
    metric = evaluate.load("glue", "mrpc")
    model.eval()  # set the model in evaluation mode (i.e. deactivate dropout layers)
    predictions = []
    actual_labels = []
    with torch.no_grad():  # deactivate autograd, see https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            metric.add_batch(predictions=preds, references=batch["labels"])

    metrics = metric.compute()
    print(f"Computed metrics: {metrics}")
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def predict_label(text, model, tokenizer, device, max_length=512):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', padding="longest", truncation=True, max_length=max_length)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        return "Not Review Bombing" if preds.item() == 1 else "Is Review Bombing"  # ==1 depends on how it was encoded!
