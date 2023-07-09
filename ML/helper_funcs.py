from ML import *


class Normalizer:

    def __init__(self, path, label_col):
        self.data = pd.read_csv(path).drop(label_col, axis=1)
        self.no = len(self.data)

    def std(self):
        """
        \sigma={\sqrt {\frac {\sum(x_{i}-{\mu})^{2}}{N}}}
        """
        self.avg = self.mean()
        tot = 0
        iter_loop = tqdm(range(len(self.data)))
        for i in iter_loop:
            iter_loop.set_description(f"{np.sum(self.data.iloc[i].tolist())}")
            tot += (np.sum(self.data.iloc[i].tolist()) - self.avg)**2
        return np.sqrt(tot / self.no)

    def mean(self) -> float:
        tot = 0
        for i in range(len(self.data)):
            tot += np.sum(self.data.iloc[i].tolist())
        return float(tot / self.no)


class Training:

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim,
        lr_schedular: optim,
        epochs: int,
        train_dl: DataLoader,
        test_dl: DataLoader,
        valid_dl: DataLoader,
        project_name: str,
        device: str,
        num_classes: int,
        classes: list,
        binary: bool,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_schedular = lr_schedular
        self.epochs = epochs
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.valid_dl = valid_dl
        self.project_name = project_name
        self.device = device
        self.num_classes = num_classes
        self.metrics = Metrics(criterion, classes, binary)

    def train(self, run_name):
        torchinfo.summary(self.model)
        wandb.init(projet=self.project_name, run_name=run_name)
        wandb.watch(self.model, log_graph=True, log="all")
        all_results = []
        for _ in tqdm(range(self.epochs)):
            model.train()
            for X_batch, y_batch in self.train_dl:
                torch.cuda.empty_cache()
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(X_batch)
                self.loss = self.criterion(logits, y_batch)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            if self.lr_schedular:
                self.lr_schedular.step()
            results = self.test()
            all_results.append(results)
            wandb.log(results)
        wandb.save()
        wandb.finish()
        return all_results

    def test(self, ):
        self.model.eval()
        with torch.inference_model():
            dloaders = [self.train_dl, self.test_dl]
            results = {}
            for dl in dloaders:
                metrics = [
                    self.metrics.accuracy,
                    self.metrics.precision,
                    self.metrics.loss,
                ]
                for metric in metrics:
                    tot = 0
                    for X, y in dl:
                        X = X.to(self.device)
                        y = y.to(self.device)
                        logits = self.model(X)
                        tot += metric(logits, y)
                    results[
                        f"{self.dl.__name__} {metric.__name__}"] = tot / len(
                            dl)
        return results

    def make_predictions(self):
        pass
