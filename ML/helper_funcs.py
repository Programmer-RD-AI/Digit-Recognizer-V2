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
    pass
