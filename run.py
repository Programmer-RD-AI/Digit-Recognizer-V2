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
test_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((28, 28)), transforms.ToTensor()]
)
test_dataset = TrainDataset(
    path="ML/data/train.csv", label_col="label", transforms=test_transform, train=False
)
val_dataset = TestDataset(path="ML/data/test.csv")
train_dl = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=round(os.cpu_count() / 2)
)
valid_dl = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=round(os.cpu_count() / 2)
)
test_dl = DataLoader(
    test_dataset, batch_size=32, shuffle=True, num_workers=round(os.cpu_count() / 2)
)
class_names = train_dataset.classes()
# model = LinearModel().to(device)

# model = CNNModel(1, 4, 256, len(class_names)).to(device)

model = torchvision.models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, len(class_names))


model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
lr_schedular = None
epochs = 25
config = {
    "model": model,
    "epochs": epochs,
    "criterion": criterion,
    "optimizer": optimizer,
    "lr_schedular": lr_schedular,
}
t = Training(
    model,
    criterion,
    optimizer,
    lr_schedular,
    epochs,
    train_dl,
    test_dl,
    valid_dl,
    PROJECT_NAME,
    device,
    class_names,
    val_dataset,
    config,
)
t.train("BaseLine-resnet18")
# print(t.test())
