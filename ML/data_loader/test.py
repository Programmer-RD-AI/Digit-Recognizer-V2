from ML import *


class TestDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def __getitem__(self, index):
        return np.array(self.data.iloc[index].values.tolist())

    def __len__(self):
        return len(self.data)
