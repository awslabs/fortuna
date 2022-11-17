import numpy as np
from fortuna.data.loader import DataLoader


def check_data_loader_is_not_random(data_loader: DataLoader) -> None:
    flag = False
    for (x1, y1), (x2, y2) in zip(data_loader, data_loader):
        if not np.alltrue(x1 == x2) or not np.alltrue(y1 == y2):
            flag = True
            break
    if flag:
        raise ValueError(
            """The data loader randomizes at every iteration. To perform this method, please provide a data loader that 
            generates the same sequence of data when called multiple times."""
        )
