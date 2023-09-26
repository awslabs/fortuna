"""
A large part of this code is a refactoring of
https://github.com/hughsalimbeni/bayesian_benchmarks/blob/master/bayesian_benchmarks/data.py.
"""

import abc
from datetime import datetime
import logging
import os
import tarfile
from typing import (
    List,
    Tuple,
)
from urllib.request import urlopen
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.arff import loadarff

from fortuna.data.loader import DataLoader

SUPPORTED_TASKS = ["regression", "classification"]

_ALL_REGRESSION_DATASETS = {}
_ALL_CLASSIFICATION_DATASETS = {}


def add_regression(C):
    _ALL_REGRESSION_DATASETS.update({C.name: C})
    return C


def add_classification(C):
    _ALL_CLASSIFICATION_DATASETS.update({C.name: C})
    return C


class Dataset:
    def __init__(self, name: str, url: str, task: str, dir: str):
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"`task={task}` not recognized. Only the following tasks are supported: {SUPPORTED_TASKS}."
            )
        if not os.path.isdir(dir):
            raise ValueError(
                f"`dir={dir}` was not found. Please pass an existing directory."
            )
        self.name = name
        self.url = url
        self.task = task
        self.dir = dir

    @property
    def datadir(self):
        return os.path.join(self.dir, self.name)

    @property
    def datapath(self):
        return os.path.join(self.datadir, self.url.split("/")[-1])

    @abc.abstractmethod
    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def shuffle(self, X: np.array, Y: np.array, seed: int = 0):
        N = X.shape[0]
        perm = np.arange(N)
        np.random.seed(seed)
        np.random.shuffle(perm)
        return X[perm], Y[perm]

    def normalize(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z_mean = np.mean(Z, 0, keepdims=True)
        Z_std = 1e-6 + np.std(Z, 0, keepdims=True)
        return (Z - Z_mean) / Z_std, Z_mean, Z_std

    def preprocess(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, self.X_mean, self.X_std = self.normalize(X)
        if self.task == "regression":
            Y, self.Y_mean, self.Y_std = self.normalize(Y)
        return X, Y

    def split(
        self,
        X: np.array,
        Y: np.array,
        prop_train: float = 0.8,
        prop_val: float = 0.1,
        prop_test: int = 0.1,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        if prop_train + prop_val + prop_test != 1.0:
            raise ValueError(
                """The sum of `prop_train`, `prop_val` and `prop_test` must be 1."""
            )
        N = X.shape[0]
        n_train = int(N * prop_train)
        n_val = int(N * prop_val)
        train_data = X[:n_train], Y[:n_train]
        val_data = X[n_train : n_train + n_val], Y[n_train : n_train + n_val]
        test_data = X[n_train + n_val :], Y[n_train + n_val :]
        return train_data, val_data, test_data

    def batch(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int = 128,
        shuffle_train: bool = False,
        shuffle_val: bool = False,
        prefetch: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_data_loader = DataLoader.from_array_data(
            train_data, batch_size=batch_size, shuffle=shuffle_train, prefetch=prefetch
        )
        val_data_loader = DataLoader.from_array_data(
            val_data, batch_size=batch_size, shuffle=shuffle_val, prefetch=prefetch
        )
        test_data_loader = DataLoader.from_array_data(
            test_data, batch_size=batch_size, prefetch=prefetch
        )
        return train_data_loader, val_data_loader, test_data_loader

    def load(
        self,
        prop_train: float = 0.8,
        prop_val: float = 0.1,
        prop_test: int = 0.1,
        batch_size: int = 128,
        shuffle_train: bool = False,
        shuffle_val: bool = False,
        prefetch: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        X, Y = self.read()
        X, Y = self.preprocess(X, Y)
        train_data, val_data, test_data = self.split(
            X, Y, prop_train=prop_train, prop_val=prop_val, prop_test=prop_test
        )
        return self.batch(
            train_data,
            val_data,
            test_data,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            shuffle_val=shuffle_val,
            prefetch=prefetch,
        )

    @property
    def needs_download(self):
        return not os.path.isfile(self.datapath)

    def download(self):
        if self.needs_download:
            logging.info("\nDownloading {} data...".format(self.name))

            if not os.path.isdir(self.datadir):
                os.mkdir(self.datadir)

            filename = os.path.join(self.datadir, self.url.split("/")[-1])
            with urlopen(self.url) as response, open(filename, "wb") as out_file:
                data = response.read()
                out_file.write(data)

            is_zipped = np.any([z in self.url for z in [".gz", ".zip", ".tar"]])
            if is_zipped:
                zip_ref = zipfile.ZipFile(filename, "r")
                zip_ref.extractall(self.datadir)
                zip_ref.close()

            logging.info("Download completed.".format(self.name))
        else:
            logging.info("{} dataset is already available.".format(self.name))


uci_base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


@add_regression
class Boston(Dataset):
    name = "boston"
    url = uci_base_url + "housing/housing.data"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_fwf(self.datapath, header=None).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Concrete(Dataset):
    name = "concrete"
    url = uci_base_url + "concrete/compressive/Concrete_Data.xls"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Energy(Dataset):
    name = "energy"
    url = uci_base_url + "00242/ENB2012_data.xlsx"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_excel(self.datapath).values[:, :-1]
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Kin8mn(Dataset):
    name = "kin8mn"
    url = "https://www.openml.org/data/download/3626/dataset_2175_kin8nm.arff"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    def download(self):
        if self.needs_download:
            logging.info("\nDownloading {} data...".format(self.name))

            if not os.path.isdir(self.datadir):
                os.mkdir(self.datadir)

            with urlopen(self.url) as response, open(self.datapath, "wb") as out_file:
                data = response.read()
                out_file.write(data)
            logging.info("Download completed.".format(self.name))

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.array([list(row) for row in loadarff(self.datapath)[0]])
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Naval(Dataset):
    name = "naval"
    url = uci_base_url + "00316/UCI%20CBM%20Dataset.zip"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    @property
    def datapath(self):
        return os.path.join(self.datadir, "UCI CBM Dataset/data.txt")

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_fwf(self.datapath, header=None).values
        X = data[:, :-2]
        Y = data[:, -2].reshape(-1, 1)
        X = np.delete(X, [8, 11], axis=1)
        return X, Y


@add_regression
class Power(Dataset):
    name = "power"
    url = uci_base_url + "00294/CCPP.zip"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    @property
    def datapath(self):
        return os.path.join(self.datadir, "CCPP/Folds5x2_pp.xlsx")

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Protein(Dataset):
    name = "protein"
    url = uci_base_url + "00265/CASP.csv"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_csv(self.datapath).values
        return data[:, 1:], data[:, 0].reshape(-1, 1)


@add_regression
class WineRed(Dataset):
    name = "winered"
    url = uci_base_url + "wine-quality/winequality-red.csv"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_csv(self.datapath, delimiter=";").values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class WineWhite(Dataset):
    name = "winewhite"
    url = uci_base_url + "wine-quality/winequality-white.csv"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_csv(self.datapath, delimiter=";").values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Yacht(Dataset):
    name = "yacht"
    url = uci_base_url + "/00243/yacht_hydrodynamics.data"
    task = "regression"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_fwf(self.datapath, header=None).values[:-1, :]
        return data[:, :-1], data[:, -1].reshape(-1, 1)


class Delgado(Dataset):
    name = ""
    url = (
        "http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz"
    )
    task = "classification"

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    @property
    def datadir(self):
        return os.path.join(self.dir, "delgado")

    @property
    def datapath(self):
        return os.path.join(self.datadir, self.name, self.name + "_R.dat")

    @property
    def needs_download(self):
        path1 = os.path.join(self.datadir, self.name + "_train_R.dat")
        path2 = os.path.join(self.datadir, self.name + "_test_R.dat")
        return not os.path.isfile(self.datapath) and not (
            os.path.isfile(path1) and os.path.isfile(path2)
        )

    def download(self):
        if self.needs_download:
            logging.info("\nDownloading {} data...".format(self.name))

            if not os.path.isdir(self.datadir):
                os.mkdir(self.datadir)

            filename = os.path.join(self.datadir, "delgado.tar.gz")
            with urlopen(self.url) as response, open(filename, "wb") as out_file:
                data = response.read()
                out_file.write(data)

            tar = tarfile.open(filename)
            tar.extractall(path=self.datadir)
            tar.close()

            logging.info("Download completed.".format(self.name))
        else:
            logging.info("{} dataset already available.".format(self.name))

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        if os.path.isfile(self.datapath):
            data = np.array(
                pd.read_csv(self.datapath, header=0, delimiter="\t").values
            ).astype(float)
        else:
            data_path1 = os.path.join(
                self.datadir, self.name, self.name + "_train_R.dat"
            )
            data1 = np.array(
                pd.read_csv(data_path1, header=0, delimiter="\t").values
            ).astype(float)

            data_path2 = os.path.join(
                self.datadir, self.name, self.name + "_test_R.dat"
            )
            data2 = np.array(
                pd.read_csv(data_path2, header=0, delimiter="\t").values
            ).astype(float)

            data = np.concatenate([data1, data2], 0)

        return data[:, :-1], data[:, -1].astype("int32")


delgado_datasets = [
    "heart-va",
    "connect-4",
    "wine",
    "tic-tac-toe",
    "fertility",
    "statlog-german-credit",
    "car",
    "libras",
    "spambase",
    "pittsburg-bridges-MATERIAL",
    "hepatitis",
    "acute-inflammation",
    "pittsburg-bridges-TYPE",
    "arrhythmia",
    "musk-2",
    "twonorm",
    "nursery",
    "breast-cancer-wisc-prog",
    "seeds",
    "lung-cancer",
    "waveform",
    "audiology-std",
    "trains",
    "horse-colic",
    "miniboone",
    "pittsburg-bridges-SPAN",
    "breast-cancer-wisc-diag",
    "statlog-heart",
    "blood",
    "primary-tumor",
    "cylinder-bands",
    "glass",
    "contrac",
    "statlog-shuttle",
    "zoo",
    "musk-1",
    "hill-valley",
    "hayes-roth",
    "optical",
    "credit-approval",
    "pendigits",
    "pittsburg-bridges-REL-L",
    "dermatology",
    "soybean",
    "ionosphere",
    "planning",
    "energy-y1",
    "acute-nephritis",
    "pittsburg-bridges-T-OR-D",
    "letter",
    "titanic",
    "adult",
    "lymphography",
    "statlog-australian-credit",
    "chess-krvk",
    "bank",
    "statlog-landsat",
    "heart-hungarian",
    "flags",
    "mushroom",
    "conn-bench-sonar-mines-rocks",
    "image-segmentation",
    "congressional-voting",
    "annealing",
    "semeion",
    "echocardiogram",
    "statlog-image",
    "wine-quality-white",
    "lenses",
    "plant-margin",
    "post-operative",
    "thyroid",
    "monks-2",
    "molec-biol-promoter",
    "chess-krvkp",
    "balloons",
    "low-res-spect",
    "plant-texture",
    "haberman-survival",
    "spect",
    "plant-shape",
    "parkinsons",
    "oocytes_merluccius_nucleus_4d",
    "conn-bench-vowel-deterding",
    "ilpd-indian-liver",
    "heart-cleveland",
    "synthetic-control",
    "vertebral-column-2clases",
    "teaching",
    "cardiotocography-10clases",
    "heart-switzerland",
    "led-display",
    "molec-biol-splice",
    "wall-following",
    "statlog-vehicle",
    "ringnorm",
    "energy-y2",
    "oocytes_trisopterus_nucleus_2f",
    "yeast",
    "oocytes_merluccius_states_2f",
    "oocytes_trisopterus_states_5b",
    "breast-cancer-wisc",
    "steel-plates",
    "mammographic",
    "monks-3",
    "balance-scale",
    "ecoli",
    "spectf",
    "monks-1",
    "page-blocks",
    "magic",
    "pima",
    "breast-tissue",
    "ozone",
    "iris",
    "waveform-noise",
    "cardiotocography-3clases",
    "wine-quality-red",
    "vertebral-column-3clases",
    "breast-cancer",
    "abalone",
]
delgado_datasets.sort()

for name in delgado_datasets:

    @add_classification
    class C(Delgado):
        name1 = name
        name = name
        url = Delgado.url
        task = Delgado.task

        def __init__(self, dir, name=name1, url=url, task=task):
            super().__init__(name=name, url=url, task=task, dir=dir)


class NYTaxiBase(Dataset, abc.ABC):
    name = "nytaxi"
    url = "https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data"
    task = "regression"
    x_bounds = [-74.04, -73.75]
    y_bounds = [40.62, 40.86]
    too_close_radius = 0.00001
    min_duration = 30
    max_duration = 3 * 3600

    def __init__(self, dir, name=name, url=url, task=task):
        super().__init__(name=name, url=url, task=task, dir=dir)

    @property
    def datapath(self):
        return os.path.join(self.datadir, "train.csv")

    def download(self):
        if self.needs_download:
            filename = os.path.join(self.dir, "nyc-taxi-trip-duration.zip")
            if not os.path.isfile(filename):
                raise FileNotFoundError(
                    """In order to use this datasets, you need to manually download it from
                `<{}>`_. Then you need to store it in {}/nyc-taxi-trip-duration.zip`.""".format(
                        self.url, self.dir
                    )
                )
            else:
                logging.info("\nUnzipping the {} data...".format(self.name))
                zip_ref = zipfile.ZipFile(filename, "r")
                zip_ref.extractall(self.datadir)
                zip_ref.close()

                zip_ref = zipfile.ZipFile(os.path.join(self.datadir, "train.zip"), "r")
                zip_ref.extractall(self.datadir)
                zip_ref.close()
                logging.info("Unzipping completed.")
        else:
            logging.info("{} dataset is already available.".format(self.name))

    def rescale(self, x, a, b):
        return b[0] + (b[1] - b[0]) * x / (a[1] - a[0])

    def convert_to_day_minute(self, d):
        day_of_week = self.rescale(float(d.weekday()), [0, 6], [0, 2 * np.pi])
        time_of_day = self.rescale(
            d.time().hour * 60 + d.time().minute, [0, 24 * 60], [0, 2 * np.pi]
        )
        return day_of_week, time_of_day

    def process_time(self, pickup_datetime, dropoff_datetime):
        d_pickup = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
        d_dropoff = datetime.strptime(dropoff_datetime, "%Y-%m-%d %H:%M:%S")
        duration = (d_dropoff - d_pickup).total_seconds()

        pickup_day_of_week, pickup_time_of_day = self.convert_to_day_minute(d_pickup)
        dropoff_day_of_week, dropoff_time_of_day = self.convert_to_day_minute(d_dropoff)

        return [
            pickup_day_of_week,
            pickup_time_of_day,
            dropoff_day_of_week,
            dropoff_time_of_day,
            duration,
        ]

    def _read(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = pd.read_csv(self.datapath)
        data = data.values

        pickup_loc = np.array((data[:, 5], data[:, 6])).T
        dropoff_loc = np.array((data[:, 7], data[:, 8])).T

        ind = np.ones(len(data)).astype(bool)
        ind[data[:, 5] < self.x_bounds[0]] = False
        ind[data[:, 5] > self.x_bounds[1]] = False
        ind[data[:, 6] < self.y_bounds[0]] = False
        ind[data[:, 6] > self.y_bounds[1]] = False

        ind[data[:, 7] < self.x_bounds[0]] = False
        ind[data[:, 7] > self.x_bounds[1]] = False
        ind[data[:, 8] < self.y_bounds[0]] = False
        ind[data[:, 8] > self.y_bounds[1]] = False

        logging.info(
            "Discarding {} out of bounds {} {}.".format(
                np.sum(np.invert(ind).astype(int)), self.x_bounds, self.y_bounds
            )
        )

        early_stop = (data[:, 5] - data[:, 7]) ** 2 + (
            data[:, 6] - data[:, 8]
        ) ** 2 < self.too_close_radius
        ind[early_stop] = False
        logging.info(
            "Discarding {} trip less than {} gp dist.".format(
                np.sum(early_stop.astype(int)), self.too_close_radius**0.5
            )
        )

        times = np.array(
            [
                self.process_time(d_pickup, d_dropoff)
                for (d_pickup, d_dropoff) in data[:, 2:4]
            ]
        )
        pickup_time = times[:, :2]
        dropoff_time = times[:, 2:4]
        duration = times[:, 4]

        short_journeys = duration < self.min_duration
        ind[short_journeys] = False
        logging.info(
            "Discarding {} less than {}s journeys.".format(
                np.sum(short_journeys.astype(int)), self.min_duration
            )
        )

        long_journeys = duration > self.max_duration
        ind[long_journeys] = False
        logging.info(
            "Discarding {} more than {}h journeys.".format(
                np.sum(long_journeys.astype(int)), self.max_duration / 3600.0
            )
        )

        pickup_loc = pickup_loc[ind, :]
        dropoff_loc = dropoff_loc[ind, :]
        pickup_time = pickup_time[ind, :]
        dropoff_time = dropoff_time[ind, :]
        duration = duration[ind]

        logging.info(
            "{} total rejected journeys.".format(np.sum(np.invert(ind).astype(int)))
        )
        return pickup_loc, dropoff_loc, pickup_time, dropoff_time, duration


@add_regression
class NYTaxiTimePrediction(NYTaxiBase):
    name = NYTaxiBase.name + "-time"

    def read(self):
        path = os.path.join(self.datadir, "taxitime_preprocessed.npz")
        if os.path.isfile(path):
            with open(path, "rb") as file:
                f = np.load(file)
                X, Y = f["X"], f["Y"]

        else:
            (
                pickup_loc,
                dropoff_loc,
                pickup_datetime,
                dropoff_datetime,
                duration,
            ) = self._read()

            pickup_sc = np.array(
                [
                    np.sin(pickup_datetime[:, 0]),
                    np.cos(pickup_datetime[:, 0]),
                    np.sin(pickup_datetime[:, 1]),
                    np.cos(pickup_datetime[:, 1]),
                ]
            ).T

            X = np.concatenate([pickup_loc, dropoff_loc, pickup_sc], 1)
            Y = duration.reshape(-1, 1)
            X, Y = np.array(X).astype(float), np.array(Y).astype(float)

            with open(path, "wb") as file:
                np.savez(file, X=X, Y=Y)
        return X, Y


@add_regression
class NYTaxiLocationPrediction(NYTaxiBase):
    name = NYTaxiBase.name + "-loc"

    def read(self):
        path = os.path.join(self.datadir, "taxiloc_preprocessed.npz")
        if os.path.isfile(path):
            with open(path, "rb") as file:
                f = np.load(file)
                X, Y = f["X"], f["Y"]
        else:
            (
                pickup_loc,
                dropoff_loc,
                pickup_datetime,
                dropoff_datetime,
                duration,
            ) = self._read()

            pickup_sc = np.array(
                [
                    np.sin(pickup_datetime[:, 0]),
                    np.cos(pickup_datetime[:, 0]),
                    np.sin(pickup_datetime[:, 1]),
                    np.cos(pickup_datetime[:, 1]),
                ]
            ).T
            X = np.concatenate([pickup_loc, pickup_sc], 1)
            Y = dropoff_loc
            X, Y = np.array(X).astype(float), np.array(Y).astype(float)

            with open(path, "wb") as file:
                np.savez(file, X=X, Y=Y)

        return X, Y


class Wilson(Dataset):
    name = "wilson"
    url = "https://drive.google.com/open?id=0BxWe_IuTnMFcYXhxdUNwRHBKTlU"
    task = "regression"

    @property
    def datadir(self):
        return os.path.join(self.dir, "wilson")

    @property
    def datapath(self):
        return "{}/{}/{}.mat".format(self.datadir, self.name, self.name)

    def download(self):
        if self.needs_download:
            filename = os.path.join(self.dir, "wilson.tar.gz")
            if not os.path.isfile(filename):
                raise FileNotFoundError(
                    """In order to use this dataset, you must obtain permission to download a zipped file from
                the following url: {}. Then you must rename it as `wilson.tar.gz`, and add it to the following
                path: {}.""".format(
                        self.url, filename
                    )
                )
            else:
                logging.info("\nUnzipping `wilson.tar.gz`...")

                def members(tf):
                    ll = len("uci/")
                    for member in tf.getmembers():
                        if member.path.startswith("uci/"):
                            member.path = member.path[ll:]
                            yield member

                with tarfile.open(filename) as tar:
                    tar.extractall(path=self.datadir, members=members(tar))
                tar.close()
                logging.info("Unzipping completed.")
        else:
            logging.info("{} dataset is already available.".format(self.name))

    def read(self):
        data = loadmat(self.datapath)["data"]
        return data[:, :-1], data[:, -1, None]


wilson_datasets = [
    "3droad",
    "challenger",
    "gas",
    "servo",
    "tamielectric",
    "airfoil",
    "concrete",
    "machine",
    "skillcraft",
    "wine",
    "autompg",
    "concreteslump",
    "houseelectric",
    "parkinsons",
    "slice",
    "yacht",
    "autos",
    "elevators",
    "housing",
    "pendulum",
    "sml",
    "bike",
]

for name in wilson_datasets:

    @add_regression
    class C(Wilson):
        name1 = name
        name = name
        url = Wilson.url
        task = Wilson.task

        def __init__(self, dir, name=name, url=url, task=task):
            super().__init__(name=name, url=url, task=task, dir=dir)


regression_datasets = list(_ALL_REGRESSION_DATASETS.keys())
regression_datasets.sort()

classification_datasets = list(_ALL_CLASSIFICATION_DATASETS.keys())
classification_datasets.sort()


def download_regression_dataset(name, dir, *args, **kwargs):
    dataset = _ALL_REGRESSION_DATASETS[name](dir, *args, **kwargs)
    dataset.download()


def download_classification_dataset(name, dir, *args, **kwargs):
    dataset = _ALL_CLASSIFICATION_DATASETS[name](dir, *args, **kwargs)
    dataset.download()


def download_all_regression_datasets(dir, *args, **kwargs):
    for name in list(_ALL_REGRESSION_DATASETS.keys()):
        download_regression_dataset(name, dir, *args, **kwargs)


def download_all_classification_datasets(dir, *args, **kwargs):
    for name in list(_ALL_CLASSIFICATION_DATASETS.keys()):
        download_classification_dataset(name, dir, *args, **kwargs)


def load_regression_dataset(
    name, dir, *args, **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = _ALL_REGRESSION_DATASETS[name](dir)
    return dataset.load(*args, **kwargs)


def load_classification_dataset(
    name, dir, *args, **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = _ALL_CLASSIFICATION_DATASETS[name](dir)
    return dataset.load(*args, **kwargs)


def get_all_classification_dataset_names() -> List[str]:
    return classification_datasets


def get_all_regression_dataset_names() -> List[str]:
    return regression_datasets
