from ML import *


class LinearModel(nn.Module):

    def __init__(self,
                 in_size: int = 784,
                 hidden_unis: int = 256,
                 out_size: int = 10) -> None:
        self.linblo1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size, hidden_unis),
            nn.BatchNorm1d(hidden_unis),
            nn.Dropout(),
        )
        self.linblo2 = nn.Sequential(
            nn.Linear(hidden_unis, hidden_unis * 2),
            nn.BatchNorm1d(hidden_unis * 2),
        )
        self.linblo3 = nn.Sequential(
            nn.Linear(hidden_unis * 2, hidden_unis * 3),
            nn.BatchNorm1d(hidden_unis * 3),
        )
        self.linblo4 = nn.Sequential(
            nn.Linear(hidden_unis * 3, hidden_unis * 4),
            nn.BatchNorm1d(hidden_unis * 4),
        )
        self.linblo5 = nn.Sequential(
            nn.Linear(hidden_unis * 4, hidden_unis * 5),
            nn.BatchNorm1d(hidden_unis * 5),
            nn.Dropout(),
        )
        self.linblo6 = nn.Sequential(
            nn.Linear(hidden_unis * 5, hidden_unis * 6),
            nn.BatchNorm1d(hidden_unis * 6),
        )
        self.linblo7 = nn.Sequential(
            nn.Linear(hidden_unis * 6, hidden_unis * 7),
            nn.BatchNorm1d(hidden_unis * 7),
        )
        self.linblo8 = nn.Sequential(
            nn.Linear(hidden_unis * 7, hidden_unis * 8),
            nn.BatchNorm1d(hidden_unis * 8),
        )
        self.linblo9 = nn.Sequential(
            nn.Linear(hidden_unis * 8, hidden_unis * 9),
            nn.BatchNorm1d(hidden_unis * 9),
        )
        self.linblo10 = nn.Sequential(
            nn.Linear(hidden_unis * 9, hidden_unis * 8),
            nn.BatchNorm1d(hidden_unis * 8),
            nn.Dropout(),
        )
        self.out = nn.Linear(hidden_unis * 8, out_size)

    def forward(self, X) -> torch.tensor:
        y = self.linblo1(X)
        y = self.linblo2(y)
        y = self.linblo3(y)
        y = self.linblo4(y)
        y = self.linblo5(y)
        y = self.linblo6(y)
        y = self.linblo7(y)
        y = self.linblo8(y)
        y = self.linblo9(y)
        y = self.linblo10(y)
        y = self.out(y)
        return y
