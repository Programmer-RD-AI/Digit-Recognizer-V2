from ML import *

# Loading Data
n = Normalizer(path="ML/data/train.csv", label_col="label")
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.ColorJitter(0.125, 0.125, 0.125, 0.125),
    # transforms.Grayscale(1),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.GaussianBlur(5, (0.1, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(n.mean()), std=(n.std())),
    # transforms.Normalize(
    #     mean=[np.mean([0.485, 0.456, 0.406])], std=[np.mean([0.229, 0.224, 0.225])]
    # ),
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
# train_transform = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()
# test_transform = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()
train_path = "ML/data/train.csv"
test_path = "ML/data/train.csv"
valid_path = "ML/data/test.csv"
train = [
    train_transform,
    train_path,
    "label",
    BATCH_SIZE,
]
test = [
    test_transform,
    test_path,
    "label",
    BATCH_SIZE,
]
val = [valid_path, 1]
train_dataset, test_dataset, val_dataset, train_dl, valid_dl, test_dl = load_data(
    train, test, val)
class_names = train_dataset.classes()
# Creating Model
optimizers = [
    optim.Adamax,
    optim.ASGD,
    optim.LBFGS,
    optim.SGD,
    optim.Adagrad,
]
for optimizer in optimizers:
    model = efficientnet_v2_s(
        torchvision.models.EfficientNet_V2_S_Weights.DEFAULT).to(device)
    model.features[0][0] = Conv2d(1,
                                  24,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1),
                                  bias=False)
    model.classifier[1] = Linear(1280, len(class_names), bias=True)
    model = torch.compile(model,
                          dynamic=True,
                          mode="max-autotune",
                          disable=True)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optimizer(model.parameters(), lr=0.001)
    lr_schedular = None
    epochs = 1
    config = {
        "model": model,
        "epochs": epochs,
        "criterion": criterion,
        "optimizer": optimizer,
        "lr_schedular": lr_schedular,
    }
    # Training
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
    t.train(f"{optimizer}-{model.__class__.__name__}")
