from ML import *


class TrainDataset(Dataset):

    def __init__(
        self,
        path: str,
        label_col: str,
        transforms: transforms,
        train: bool = True,
        seed: int = 42,
        test_split: float = 0.25,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        self.data = pd.read_csv(path)
        self.labels = np.array(self.data[label_col].tolist())
        self.images = self.data.drop(label_col, inplace=False, axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.images,
            self.labels,
            shuffle=True,
            random_state=seed,
            train_size=(1 - test_split),
        )
        if train:
            self.images = self.X_train
            self.labels = self.y_train
        else:
            self.images = self.X_test
            self.labels = self.y_test
        self.transforms = transforms

    def __getitem__(self, index) -> Tuple[torch.tensor, int]:
        self.image = (np.array(self.images.iloc[index].tolist()).astype(
            np.uint8).reshape(28, 28, 1))
        self.image = self.transforms(self.image)
        return (
            self.image,
            self.labels[index],
        )

    def shape(self) -> Tuple[Tuple, Tuple]:
        return (self.images.shape, self.labels.shape)

    def __len__(self) -> int:
        return len(self.data)

    def classes(self):
        return np.unique(self.labels)
