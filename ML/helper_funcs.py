from ML import *


class Normalizer:
    def __init__(self, path, label_col):
        self.data = pd.read_csv(path).drop(label_col, axis=1)
        self.no = len(self.data)
        self.create_long_list()

    def std(self):
        """
        \sigma={\sqrt {\frac {\sum(x_{i}-{\mu})^{2}}{N}}}
        """
        return np.std(np.array(self.tot_imgs))

    def create_long_list(self):
        self.tot_imgs = []
        for i in range(self.no):
            self.tot_imgs.append(np.array(self.data.iloc[i].tolist()) / 255)
        self.tot_imgs = torch.tensor(self.tot_imgs)
        return self.tot_imgs.squeeze().view(self.no * 784)

    def mean(self) -> float:
        tot = 0
        for i in tqdm(range(len(self.data))):
            tot += (np.array(self.data.iloc[i].tolist()) / 255).mean()
        # print(float(tot / self.no))
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
        classes: list,
        valid_ds: Dataset,
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
        self.num_classes = len(classes)
        self.train_metrics = Metrics(criterion, classes, train_dl, model)
        self.test_metrics = Metrics(criterion, classes, test_dl, model)
        self.valid_ds = valid_ds

    def train(self, run_name):
        torchinfo.summary(self.model)
        wandb.init(project=self.project_name, name=run_name)
        wandb.watch(self.model, log_graph=True, log="all")
        all_results = []
        iterater = tqdm(range(self.epochs))
        for _ in iterater:
            tot = 0
            self.model.train()
            for X_batch, y_batch in self.train_dl:
                torch.cuda.empty_cache()
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(X_batch)
                self.loss = self.criterion(logits, y_batch)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                tot += self.loss.item()
            if self.lr_schedular:
                self.lr_schedular.step()
            results = self.test()
            wandb.log(results)
            wandb.alert(
                title="Results",
                text=f"{results}",
                level=AlertLevel.WARN,  # TODO
                wait_duration=150,
            )
            all_results.append(results)
        img_pred = self.plot_predictions()  # TODO
        wandb.log(img_pred)
        wandb.save()
        wandb.finish()
        predictions = self.make_predictions(run_name)  # TODO
        return all_results, predictions

    def test(
        self,
    ):
        self.model.eval()
        with torch.no_grad():
            results = {}
            results["train accuracy"] = self.train_metrics.accuracy()
            results["test accuracy"] = self.test_metrics.accuracy()
            results["train loss"] = self.train_metrics.loss()
            results["test loss"] = self.test_metrics.loss()
            results["train precision"] = self.train_metrics.precision()
            results["test precision"] = self.test_metrics.precision()
        self.model.train()
        return results

    def make_predictions(self, run_name=None):
        self.model.eval()
        predictions = {}
        for i, image in enumerate(self.valid_dl):
            pred = torch.argmax(
                torch.softmax(self.model(image.view(1, 1, 28, 28).to(device).long()), dim=1), dim=1
            )
            predictions[i] = pred
        if run_name:
            pd.DataFrame(predictions).to_csv(f"ML/predictions/{run_name}.csv", index=False)
        return predictions

    def plot_predictions(self):
        predictions = self.make_predictions()
        ids = predictions.keys()
        preds = predictions.values()
        img_pred = {}
        for _id, pred in zip(ids, preds):
            img = torch.tensor(self.valid_ds[_id]).permute(1, 2, 0)
            plt.title(f"{pred}")
            plt.imshow(img)
            img_pred[pred] = wandb.Image(img)
        return img_pred
