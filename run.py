from ML import *

from ML.data_loader.train import TrainDataset

n = Normalizer(path="ML/data/train.csv", label_col="label")

train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ColorJitter(0.125, 0.125, 0.125, 0.125),
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(5, (0.1, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(n.mean()), std=(n.std())),
    ]
)
train_dataset = TrainDataset(
    path="ML/data/train.csv", label_col="label", transforms=train_transform, train=True
)
val_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((28, 28)), transforms.ToTensor()]
)
val_dataset = TrainDataset(
    path="ML/data/train.csv", label_col="label", transforms=val_transform, train=False
)
test_dataset = TestDataset(path="ML/data/test.csv")
train_dl = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=round(os.cpu_count() / 2)
)
test_dl = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=round(os.cpu_count() / 2)
)
