from ML import *


class Metrics:

    def __init__(self, criterion: nn.Module, classes: list, dataloader, model,
                 device):
        self.device = device
        self.criterion = criterion
        self.classes = torch.tensor(classes).to(device=self.device)
        self.dataloader = dataloader
        self.model = model.to(device)

    def loss(self):
        tot = 0
        for X, y in self.dataloader:
            preds = self.model(X.to(self.device))
            tot += self.criterion(preds.to(self.device), y.to(self.device))
        return tot / len(self.dataloader)

    def accuracy(self):
        tot = 0
        cor = 0
        for X, y in self.dataloader:
            preds = torch.argmax(torch.softmax(self.model(X.to(self.device)),
                                               dim=1),
                                 dim=1)

            for pred, tr in zip(preds, y):
                if pred == tr:
                    # print(pred, tr, len(self.dataloader))
                    cor += 1
                tot += 1
        return (cor / tot) * 100

    def precision(self, ):
        tp = 0
        fp = 0
        for X, y in self.dataloader:
            preds = torch.argmax(torch.softmax(self.model(X.to(self.device)),
                                               dim=1),
                                 dim=1)
            for clz in self.classes:
                for pred, tr in zip(preds.to(self.device), y.to(self.device)):
                    if pred == clz and tr == pred:
                        # print(pred, clz, tr)
                        tp += 1
                    if pred == clz and tr != pred:
                        # print(pred, clz, tr)
                        fp += 1
                # print(tp, fp)
        return tp / (tp + fp)
