from ML import *


class CNNModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_units_cnn: int,
        hidden_units_linear: int,
        output_classes: int,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 1,
        pool: int = 2,
        pool_type: nn.Module = nn.MaxPool2d,
        activation: nn.Module = nn.ReLU,
        affine: bool = False,
    ):
        super().__init__()
        self.convblo1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_units_cnn,
                      kernel_size, stride, padding),
            pool_type(pool),
            nn.BatchNorm2d(hidden_units_cnn, affine=affine),
            activation(),
        )
        self.convblo2 = nn.Sequential(
            nn.Conv2d(
                hidden_units_cnn, hidden_units_cnn * 2, kernel_size, stride, padding
            ),
            pool_type(pool),
            nn.BatchNorm2d(hidden_units_cnn * 2, affine=affine),
            activation(),
        )
        self.convblo3 = nn.Sequential(
            nn.Conv2d(
                hidden_units_cnn * 2, hidden_units_cnn * 3, kernel_size, stride, padding
            ),
            pool_type(pool),
            activation(),
        )
        self.convblo4 = nn.Sequential(
            nn.Conv2d(
                hidden_units_cnn * 3, hidden_units_cnn * 4, kernel_size, stride, padding
            ),
            pool_type(pool),
            nn.BatchNorm2d(hidden_units_cnn * 4, affine=affine),
            activation(),
        )
        self.convblo5 = nn.Sequential(
            nn.Conv2d(
                hidden_units_cnn * 4, hidden_units_cnn * 5, kernel_size, stride, padding
            ),
            pool_type(pool),
            nn.BatchNorm2d(hidden_units_cnn * 5, affine=affine),
            activation(),
        )
        self.linblo1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30 * 1 * 1, hidden_units_linear),
            nn.BatchNorm1d(hidden_units_linear, affine=affine),
            activation(),
        )
        self.linblo2 = nn.Sequential(
            nn.Linear(hidden_units_linear, hidden_units_linear * 2),
            nn.BatchNorm1d(hidden_units_linear * 2, affine=affine),
            activation(),
        )
        self.linblo3 = nn.Sequential(
            nn.Linear(hidden_units_linear * 2, hidden_units_linear * 3),
            nn.BatchNorm1d(hidden_units_linear * 3, affine=affine),
            activation(),
        )
        self.linblo4 = nn.Sequential(
            nn.Linear(hidden_units_linear * 3, hidden_units_linear * 4),
            nn.BatchNorm1d(hidden_units_linear * 4, affine=affine),
            activation(),
        )
        self.linblo5 = nn.Sequential(
            nn.Linear(hidden_units_linear * 4, hidden_units_linear * 3),
            nn.BatchNorm1d(hidden_units_linear * 3, affine=affine),
            activation(),
        )
        self.out = nn.Linear(hidden_units_linear * 3, output_classes)

    def forward(self, X):
        y = self.convblo1(X)
        y = self.convblo2(y)
        y = self.convblo3(y)
        y = self.convblo4(y)
        y = self.convblo5(y)
        # print(y.shape)
        y = self.linblo1(y)
        y = self.linblo2(y)
        y = self.linblo3(y)
        y = self.linblo4(y)
        y = self.linblo5(y)
        y = self.out(y)
        return y
