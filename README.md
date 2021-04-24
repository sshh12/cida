# Continuously Indexed Domain Adaptation (CIDA)

An unofficial refactor of the code used by the ICML 2020 paper [Continuously Indexed Domain Adaptation](http://wanghao.in/paper/ICML20_CIDA.pdf).

- [Official Code](https://github.com/hehaodele/CIDA)

## Install

```
$ pip install git+https://github.com/sshh12/cida.git --upgrade
```

## Usage

> The code should work for any dataset that serves data in the correct format (see [RotatedMNIST](https://github.com/sshh12/cida/blob/main/cida/datasets/rotated_mnist.py)), although I created this repo for my specific case so some trivial/generic features may be missing. Fill free to create an issue/PR.

### Classification (Rotated MNIST)

```python
from torch.utils.data import DataLoader
from cida.datasets import RotatedMNIST
from cida.conv_model import ConvPCIDAClassifier
import os

if __name__ == "__main__":
    # RotatedMNIST.download()
    dataset = RotatedMNIST(
        os.path.join("data", "MNIST", "processed", "training.pt"), rotate_range=(0, 360), train_range=(0, 45)
    )
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=100,
        num_workers=1,
    )
    val_dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=100,
        num_workers=1,
    )
    model = ConvPCIDAClassifier(
        classes=10, input_size=28 * 28, domain_dims=1, domains_to_labels=RotatedMNIST.domains_to_labels, verbose=True
    )
    model = model.to("cpu")
    model.fit(dataloader, val_dataloader, epochs=100)
    print(model.predict(next(iter(val_dataloader))))
```

### Regression

```python
from torch.utils.data import DataLoader
from cida.regr_model import PCIDARegressor
import os

def mse(y_pred, y):
    return ((y_pred - y) ** 2).mean()

if __name__ == "__main__":
    dataset = MyRegressionDataset()
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=128,
        num_workers=1,
    )
    val_dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=128,
        num_workers=1,
    )
    model = PCIDARegressor(
        input_size=1000,
        domain_dims=1,
        domains_to_labels=dataset.domains_to_labels,
        lr=3e-4,
        beta1=0.9,
        gamma=0.99,
        weight_decay=5e-4,
        dropout=0.5,
        lambda_gan=lambda epoch: 3.0,
        encoder_hidden_size=128,
        discriminator_hidden_size=128,
        latent_size=64,
        test_domain_known=True,
        metrics={"mse": mse},
        verbose=True,
    )
    model = model.to("cuda")
    model.fit(dataloader, val_dataloader, epochs=100, save_metric="test_mse", save_fn="cida-best.pth")
```
