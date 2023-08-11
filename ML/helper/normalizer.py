from ML import *


class Normalizer:

    def __init__(self, path, label_col):
        self.data = pd.read_csv(path).drop(label_col, axis=1)
        self.no = len(self.data)
        self.create_long_list()

    def std(self):
        r"""
        \sigma={\sqrt {\frac {\sum(x_{i}-{\mu})^{2}}{N}}}
        """
        return np.std(np.array(self.tot_imgs))

    def create_long_list(self):
        self.tot_imgs = []
        for i in range(self.no):
            self.tot_imgs.append(np.array(self.data.iloc[i].tolist()) / 255)
        self.tot_imgs = (torch.tensor(self.tot_imgs).squeeze().view(
            self.no * 784).float())
        return self.tot_imgs

    def mean(self) -> float:
        tot = 0
        for i in tqdm(range(len(self.data))):
            tot += (np.array(self.data.iloc[i].tolist()) / 255).mean()
        return float(tot / self.no)
