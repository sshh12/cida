# Continuously Indexed Domain Adaptation (CIDA)

An unofficial refactor of the code used by the ICML 2020 paper [Continuously Indexed Domain Adaptation](http://wanghao.in/paper/ICML20_CIDA.pdf).

- [Official Code](https://github.com/hehaodele/CIDA)

## Install

```
$ pip install git+https://github.com/sshh12/cida.git --upgrade
```

## Usage

### Rotated MNIST

```python
from torch.utils.data import DataLoader
from cida.datasets import RotatedMNIST
from cida.model import ConvPCIDAClassifier
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
