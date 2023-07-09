from ML import *


class Metrics:
    def __init__(self, criterion: nn.Module, classes: list, binary: bool):
        self.criterion = criterion
        self.classes = classes
        self.binary = binary

    def loss(self, preds, true):
        return self.criterion(preds, true).item()

    def accuracy(self, preds, true):
        func = torch.sigmoid if self.binary else torch.softmax
        preds = torch.argmax(func(preds, dim=1), dim=1)
        tot = 0
        cor = 0
        for pred, tr in zip(preds, true):
            if pred == tr:
                cor += 1
            tot += 1
        return cor / tot

    def precision(self, preds, true):
        func = torch.sigmoid if self.binary else torch.softmax
        preds = torch.argmax(func(preds, dim=1), dim=1)
        tot_prec = 0
        for clz in self.classes:
            tp = 0
            fp = 0
            for pred, tr in zip(preds, true):
                if pred == clz and tr == pred:
                    tp += 1
                if pred == clz and tr != pred:
                    fp += 1
            tot_prec += tp / (tp + fp)
        return tot_prec / len(self.classes)
