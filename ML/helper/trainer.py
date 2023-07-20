from ML import *
from ML.helper.alert import *
from ML.helper.metrics import *

device = torch.device("cuda")


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
        config: dict,
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
        self.train_metrics = Metrics(
            criterion, classes, train_dl, model, device)
        self.test_metrics = Metrics(criterion, classes, test_dl, model, device)
        self.valid_ds = valid_ds
        self.config = config
        self.all_results = []
        self.resize = transforms.Resize((IMG_SIZE, IMG_SIZE))

    def train(self, run_name):
        torch.cuda.empty_cache()
        torchinfo.summary(self.model)
        wandb.init(project=self.project_name,
                   name=run_name, config=self.config)
        wandb.watch(self.model, log="all")
        iterater = tqdm(range(self.epochs))
        for _ in iterater:
            torch.cuda.empty_cache()
            self.model.train()
            for batch_n, (X_batch, y_batch) in enumerate(self.train_dl):
                torch.cuda.empty_cache()
                self.train_step(X_batch, y_batch)
                iterater.set_description(f"{batch_n}/{len(self.train_dl)}")
                torch.cuda.empty_cache()
            iterater.set_description("Testing")
            threading.Thread(target=self.test_step, args=[run_name]).start()
            if self.lr_schedular:
                self.lr_schedular.step()
        self.make_predictions()
        img_pred = self.plot_predictions(run_name)  # TODO
        wandb.log(img_pred)
        wandb.save()
        wandb.finish()
        self.save_model(run_name)
        return self.all_results, self.predictions

    def test_step(self, run_name):
        results = self.test()
        wandb.log(results)
        self.all_results.append(results)
        self.past_results = results
        self.predictions = self.make_predictions(run_name)

    def train_step(self, X_batch, y_batch) -> Tuple[int, torch.Tensor]:
        torch.cuda.empty_cache()
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        logits = self.model(X_batch)
        self.loss = self.criterion(logits, y_batch)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item(), logits

    def test(
        self,
    ):
        self.model.eval()
        with torch.no_grad():
            results = {}
            results["train accuracy"] = threading.Thread(
                target=self.train_metrics.accuracy, args=[()]
            ).start()
            results["test accuracy"] = threading.Thread(
                target=self.test_metrics.accuracy, args=[()]
            ).start()
            results["train loss"] = threading.Thread(
                target=self.train_metrics.loss, args=[()]
            ).start()
            results["test loss"] = threading.Thread(
                target=self.test_metrics.loss, args=[()]
            ).start()
            results["train precision"] = threading.Thread(
                target=self.train_metrics.precision, args=[()]
            ).start()
            results["test precision"] = threading.Thread(
                target=self.test_metrics.precision, args=[()]
            ).start()
        return results

    def make_predictions(self, run_name=None):
        self.model.eval()
        ids = []
        preds = []
        for i, image in tqdm(enumerate(self.valid_dl)):
            i = i + 1
            pred = torch.argmax(
                torch.softmax(
                    self.model(
                        self.resize(image.view(
                            1, 1, 28, 28).to(device).float())
                    ),
                    dim=1,
                ),
                dim=1,
            )
            threading.Thread(target=preds.append, args=[pred.item()]).start()
            threading.Thread(target=ids.append, args=[i]).start()
        print(pd.DataFrame({"ImageId": ids, "Label": preds}))
        if run_name:
            pd.DataFrame({"ImageId": ids, "Label": preds}).to_csv(
                f"ML/predictions/{run_name}.csv", index=False
            )
        return ids, preds

    def plot_predictions(self, run_name):
        ids, preds = self.make_predictions(run_name)
        img_pred = {}
        for _id, pred in tqdm(zip(ids, preds)):
            img = torch.tensor(self.valid_ds[_id].reshape(1, 28, 28))
            plt.title(f"{pred}")
            plt.imshow(img.permute(1, 2, 0))
            img_pred[f"{pred}"] = wandb.Image(img.float())
        return img_pred

    def save_model(self, run_name):
        torch.save(self.model.state_dict(), f"ML/models/{run_name}_sd.pth")
        torch.save(self.model, f"ML/models/{run_name}_m.pt")
