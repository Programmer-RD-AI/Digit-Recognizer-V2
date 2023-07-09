from ML import *


class Metrics:
    def __init__(self, criterion: nn.Module, classes: list, dataloader, model):
        self.device = "cuda"
        self.criterion = criterion
        self.classes = torch.tensor(classes).to(device=self.device)
        self.dataloader = dataloader
        self.model = model

    def loss(self):
        tot = 0
        for X, y in self.dataloader:
            preds = self.model(X.to(self.device))
            tot += self.criterion(preds.to(self.device), y.to(self.device))
        return tot / len(self.dataloader)

    def accuracy(self):
        tot = 0
        for X, y in self.dataloader:
            preds = torch.argmax(torch.softmax(self.model(X.to(self.device)), dim=1), dim=1)
            tot = 0
            cor = 0
            for pred, tr in zip(preds, y):
                if pred == tr:
                    # print(pred, tr, len(self.dataloader))
                    cor += 1
                tot += 1
            tot += cor / tot
        return (tot / len(self.dataloader)) * 100

    def precision(
        self,
    ):
        tot = 0
        for X, y in self.dataloader:
            preds = torch.argmax(torch.softmax(self.model(X.to(self.device)), dim=1), dim=1)
            tot_prec = 0
            for clz in self.classes:
                tp = 0
                fp = 0
                for pred, tr in zip(preds.to(self.device), y.to(self.device)):
                    if pred == clz and tr == pred:
                        # print(pred, clz, tr)
                        tp += 1
                    if pred == clz and tr != pred:
                        # print(pred, clz, tr)
                        fp += 1
                # print(tp, fp)
                try:
                    tot_prec += tp / (tp + fp)
                except:
                    tot_prec += 0
            tot += tot_prec / len(self.classes)
        return tot / len(self.dataloader)
