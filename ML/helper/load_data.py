from ML import *
from ML.data_loader import *


def load_data(train: list, test: list, valid: list):
    train_dataset = TrainDataset(
        path=train[1],
        label_col=train[2],
        transforms=train[0],
        train=True,
    )
    test_dataset = TrainDataset(
        path=test[1],
        label_col=test[2],
        transforms=test[0],
        train=False,
    )
    val_dataset = TestDataset(path=valid[0])
    train_dl = DataLoader(
        train_dataset,
        batch_size=train[3],
        shuffle=True,
        num_workers=round(os.cpu_count() / 2),
        pin_memory=True,
    )
    valid_dl = DataLoader(
        val_dataset,
        batch_size=valid[1],
        shuffle=False,
        num_workers=round(os.cpu_count() / 2),
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=test[3],
        shuffle=True,
        num_workers=round(os.cpu_count() / 2),
        pin_memory=True,
    )
    return (train_dataset, test_dataset, val_dataset, train_dl, valid_dl,
            test_dl)
