from ML import *


class LinearModel(nn.Module):

    def __init__(
        self,
        in_size: int = 784,
        hidden_unis: int = 64,
        out_size: int = 10,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.linblo1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size, hidden_unis),
            nn.BatchNorm1d(hidden_unis, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linblo2 = nn.Sequential(
            nn.Linear(hidden_unis, hidden_unis * 2),
            nn.BatchNorm1d(hidden_unis * 2, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linblo3 = nn.Sequential(
            nn.Linear(hidden_unis * 2, hidden_unis * 3),
            nn.BatchNorm1d(hidden_unis * 3, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linblo4 = nn.Sequential(
            nn.Linear(hidden_unis * 3, hidden_unis * 4),
            nn.BatchNorm1d(hidden_unis * 4, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linblo5 = nn.Sequential(
            nn.Linear(hidden_unis * 4, hidden_unis * 5),
            nn.BatchNorm1d(hidden_unis * 5, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linblo6 = nn.Sequential(
            nn.Linear(hidden_unis * 5, hidden_unis * 6),
            nn.BatchNorm1d(hidden_unis * 6, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linblo7 = nn.Sequential(
            nn.Linear(hidden_unis * 6, hidden_unis * 7),
            nn.BatchNorm1d(hidden_unis * 7, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linblo8 = nn.Sequential(
            nn.Linear(hidden_unis * 7, hidden_unis * 8),
            nn.BatchNorm1d(hidden_unis * 8, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linblo9 = nn.Sequential(
            nn.Linear(hidden_unis * 8, hidden_unis * 9),
            nn.BatchNorm1d(hidden_unis * 9, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linblo10 = nn.Sequential(
            nn.Linear(hidden_unis * 9, hidden_unis * 8),
            nn.BatchNorm1d(hidden_unis * 8, affine=affine),
            nn.Dropout(),
            nn.ReLU(),
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
