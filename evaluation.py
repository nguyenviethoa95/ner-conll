import torch
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def _confusion_matrix(preds, y):
    cm = confusion_matrix(y, preds)
    return confusion_matrix


def _report(preds, y):
    report = classification_report(y, preds, target_names=["B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-MISC", "I-MISC", "O"])
    return report


def accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True) # Get the tag index of max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def f1_macro(preds, y):
    score = f1_score(y, preds, average='macro')
    return score