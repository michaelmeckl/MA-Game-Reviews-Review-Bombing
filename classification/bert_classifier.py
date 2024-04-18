import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import evaluate
from torch import nn
from torch.utils.data import SequentialSampler, DataLoader
from transformers import BertModel, RobertaModel, DistilBertModel, BertForSequenceClassification
from classification.classification_utils import calculate_prediction_results


def get_pretrained_bert_for_sequence(n_classes, model_checkpoint):
    return BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=n_classes)


# Adapted from https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b and
# https://wellsr.com/python/fine-tuning-bert-for-sentiment-analysis-with-pytorch/
class BERTClassifier(nn.Module):
    def __init__(self, num_classes, model_checkpoint):
        super(BERTClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained(model_checkpoint)
        self.bert = DistilBertModel.from_pretrained(model_checkpoint)
        # self.bert = RobertaModel.from_pretrained(model_checkpoint)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_classes)
        # TODO use a bit more layers on top of BERT?
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
        # pooled_output = outputs.pooler_output  # pooled output represents each input sequence / review as an embedding
        # use this instead for DistilBERT and RoBERTa
        hidden_state_output = outputs['last_hidden_state'][:, 0, :]
        x = self.dropout(hidden_state_output)
        logits = self.fc(x)
        return logits


def train_model(model, data_loader, optimizer, scheduler, criterion, device, epoch, writer, history, progress_bar):
    running_losses = []
    total_correct = 0
    total_samples = 0
    dataset_size = len(data_loader.dataset)

    model.train()  # set the model in training mode
    for batch_number, batch in enumerate(data_loader, 0):
        optimizer.zero_grad()  # reset the gradients after every batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device, torch.long)  # make sure this is a Long Tensor as loss function expects it
        labels = labels.squeeze()  # squeeze label tensor to remove the outer dimension
        # labels = batch['labels'].to(device, torch.float)   # for BCEWithLogitsLoss loss

        # calculate the forward pass of the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits = outputs.logits   # use this with BertForSequenceClassification instead of outputs
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()  # backpropagate the loss and compute the gradients
        # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # update the model weights
        scheduler.step()

        # calculate loss and update console output
        batch_loss = loss.item() * input_ids.size(0)  # .item() gets the value as a python number from the tensor
        running_losses.append(batch_loss)
        # noinspection PyUnresolvedReferences
        total_correct += (preds == labels).sum().item()
        # total_correct += (preds == labels.squeeze()).sum().item()   # for BCEWithLogitsLoss loss
        total_samples += labels.shape[0]   # len(labels)

        progress_bar.update(1)
        if batch_number % 100 == 0:
            current_num = (batch_number + 1) * len(input_ids)
            print(f"\nbatch loss: {batch_loss:>7f}  [{current_num:>4d}/{dataset_size:>4d}]")

    # for calculating loss see https://discuss.pytorch.org/t/correct-way-to-calculate-train-and-valid-loss/178974 and
    # https://stackoverflow.com/questions/59584457/pytorch-is-there-a-definitive-training-loop-similar-to-keras-fit
    avg_train_loss = sum(running_losses) / dataset_size
    avg_train_accuracy = round(total_correct / total_samples, 4) * 100
    print(f"\nAverage train loss: {avg_train_loss:.4f}\nTrain Accuracy:{avg_train_accuracy:.2f}%\n")

    history["train_loss"].append(avg_train_loss)
    history["train_accuracy"].append(avg_train_accuracy)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/train", avg_train_accuracy, epoch)
    return avg_train_loss, avg_train_accuracy


def evaluate_model(model, data_loader, criterion, device, epoch, writer, history):
    metric = evaluate.load("glue", "mrpc")
    all_predictions = []
    actual_labels = []
    validation_losses = []
    total_correct = 0
    total_samples = 0
    dataset_size = len(data_loader.dataset)

    model.eval()  # set the model in evaluation mode (i.e. deactivate dropout layers)
    with torch.no_grad():  # deactivate autograd, see https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device, torch.long).squeeze()
            # labels = batch['labels'].to(device, torch.float)   # for BCEWithLogitsLoss loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits = outputs.logits   # use this with BertForSequenceClassification instead of outputs
            _, preds = torch.max(outputs, dim=1)
            # for batch with 2 reviews outputs is tensor([[-0.1297,  0.0863],
            #                                             [-0.1276,  0.0026]])
            #  => torch.max: [1, 1]
            #  => labels: [1, 0]
            all_predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            # actual_labels.extend(labels.squeeze().cpu().tolist())   # for BCEWithLogitsLoss loss
            metric.add_batch(predictions=preds, references=labels)

            # calculate validation loss and accuracy
            loss = criterion(outputs, labels)
            validation_losses.append(loss.item() * labels.size(0))
            # noinspection PyUnresolvedReferences
            total_correct += (preds == labels).sum().item()
            # total_correct += (preds == labels.squeeze()).sum().item()   # for BCEWithLogitsLoss loss
            total_samples += labels.shape[0]   # len(labels)

    avg_val_loss = sum(validation_losses) / dataset_size
    val_accuracy = round(total_correct / total_samples, 4) * 100
    print(f"Average val loss: {avg_val_loss:.4f}\nVal Accuracy:{val_accuracy:.2f}%")
    # print(accuracy_score(actual_labels, all_predictions))   # should be the same as the calculated accuracy above

    # Calculate additional metrics
    f1 = f1_score(actual_labels, all_predictions, average='binary')
    print(f'F1-score: {f1}')
    print(confusion_matrix(actual_labels, all_predictions))
    metrics = metric.compute()
    print(f"Computed metrics: {metrics}")
    report = classification_report(actual_labels, all_predictions)
    print(f"Classification Report:\n{report}")

    # save loss and accuracy per epoch to plot later
    history["val_loss"].append(avg_val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["f1_score"].append(f1)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch)
    return avg_val_loss, val_accuracy


def predict_label(text, target_col: str, model, tokenizer, device, max_length=512):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', padding="longest", truncation=True, max_length=max_length)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    # depends on how it was encoded during training!
    label_encoding = {
        0: "Is Review Bombing" if target_col == "is-review-bombing" else "Is Off-Topic",
        1: "Not Review Bombing" if target_col == "is-review-bombing" else "Not Off-Topic"
    }

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        return label_encoding[preds.item()]


def predict_test_labels(model, test_dataset, batch_size, device, target_col):
    # Taken from https://mccormickml.com/2021/06/29/combining-categorical-numerical-features-with-bert/#342-training-loop
    # Create a DataLoader to batch our test samples for us. We'll use a sequential sampler this time--don't need this
    # to be random!
    prediction_sampler = SequentialSampler(test_dataset)
    prediction_dataloader = DataLoader(test_dataset, sampler=prediction_sampler, batch_size=batch_size)

    print('Predicting labels for {:,} test reviews ...'.format(len(test_dataset)))
    # Put model in evaluation mode
    model.eval()
    predictions, true_labels = [], []

    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask, labels = batch

        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            result = model(input_ids,
                           token_type_ids=None,
                           attention_mask=input_mask,
                           return_dict=True)

        logits = result.logits
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    print('DONE.')
    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)
    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Calculate the F1
    f1 = f1_score(flat_true_labels, flat_predictions)
    print('F1 Score: %.3f' % f1)

    calculate_prediction_results(true_labels, predictions)

    # TODO show some predictions
