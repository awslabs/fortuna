import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import flax.linen as nn
import requests
from fortuna.model.cnn import CNN
from fortuna.model.mlp import MLP
from jax import random
from tests.make_data import make_array_random_inputs
from tqdm import tqdm


def download(ckpt_dir, url):
    name = url[url.rfind("/") + 1 : url.rfind("?")]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, "flaxmodels")
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: "{url[:url.rfind("?")]}" to {ckpt_file}')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + ".temp")
        with open(ckpt_file_temp, "wb") as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("An error occured while downloading, please try again.")
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file


def load_config(path):
    return json.loads(
        open(path, "r", encoding="utf-8").read(),
        object_hook=lambda d: SimpleNamespace(**d),
    )


class TestMLP(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_inputs = (3,)
        self.output_dim = 2
        self.n_inputs = 10
        self.inputs = make_array_random_inputs(
            n_inputs=self.n_inputs, shape_inputs=self.shape_inputs
        )
        self.rng = random.PRNGKey(0)

    def test_mlp(self):
        mlp = MLP(output_dim=self.output_dim)
        params = mlp.init(self.rng, self.inputs)
        outputs = mlp.apply(params, self.inputs)
        assert outputs.shape == (self.n_inputs, self.output_dim)
        assert mlp.output_dim == self.output_dim

    def test_mlp_widths_activations(self):
        mlp = MLP(
            widths=[1, 2, 3, 4],
            activations=[nn.sigmoid, nn.tanh, nn.leaky_relu, nn.relu],
            output_dim=self.output_dim,
        )
        params = mlp.init(self.rng, self.inputs)
        outputs = mlp.apply(params, self.inputs)
        assert outputs.shape == (self.n_inputs, self.output_dim)

        mlp = MLP(
            widths=[1, 2, 3],
            activations=[nn.sigmoid, nn.tanh, nn.leaky_relu, nn.relu],
            output_dim=self.output_dim,
        )
        self.assertRaises(Exception, mlp.init, self.rng, self.inputs)


class TestCNN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_inputs = (6, 5, 4)
        self.output_dim = 2
        self.n_inputs = 10
        self.dropout_rate = 0.2
        self.inputs = make_array_random_inputs(
            n_inputs=self.n_inputs, shape_inputs=self.shape_inputs
        )
        self.rng = random.PRNGKey(0)

    def test_CNN(self):
        cnn = CNN(output_dim=self.output_dim, dropout_rate=self.dropout_rate)
        key1, key2, key3 = random.split(self.rng, 3)
        variables = cnn.init(dict(params=key1, dropout=key2), self.inputs)
        outputs = cnn.apply(variables, self.inputs, rngs=dict(dropout=key3))
        assert outputs.shape == (self.n_inputs, self.output_dim)
        assert cnn.dropout_rate is self.dropout_rate
        assert cnn.output_dim == self.output_dim
