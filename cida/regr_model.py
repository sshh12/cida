import torch.nn.functional as F
import torch.nn as nn
import torch

from cida.util import (
    ensure_tensor,
    ensure_numpy,
    init_weights,
    set_requires_grad,
    convert_Avec_to_A,
    neg_guassian_likelihood,
)

from collections import Counter
import numpy as np
import tqdm


class Encoder(nn.Module):
    def __init__(self, *, domain_dims, input_size, hidden_size, latent_size, dropout, encode_domain):
        super(Encoder, self).__init__()

        self.fc_inp = nn.Sequential(
            nn.Linear(domain_dims + input_size if encode_domain else input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.fc_feats = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(True),
        )

        self.fc_pred = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
        )

        self.input_size = input_size
        self.encode_domain = encode_domain

    def forward(self, x, domain):
        if self.encode_domain:
            input_ = self.fc_inp(torch.cat([domain, x], 1))
        else:
            input_ = self.fc_inp(x)
        enc = self.fc_feats(input_)
        y = self.fc_pred(enc)
        return y, enc


class Discriminator(nn.Module):
    def __init__(self, *, hidden_size, latent_size, domain_dims):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, domain_dims * 2),
        )

    def forward(self, x):
        return self.net(x)


class PCIDARegressor(nn.Module):
    def __init__(
        self,
        lr=2e-4,
        beta1=0.9,
        gamma=0.5 ** (1 / 50),
        weight_decay=5e-4,
        dropout=0.2,
        lambda_gan=2.0,
        domain_dims=1,
        encoder_hidden_size=256,
        discriminator_hidden_size=512,
        latent_size=100,
        input_size=100,
        test_domain_known=True,
        loss=F.mse_loss,
        domains_to_labels=None,
        verbose=False,
        metrics={},
        save_metric="test_mse",
        save_fn="cida-best.pth",
    ):
        super(PCIDARegressor, self).__init__()

        assert len(metrics) > 0, "No metrics provided"
        assert domains_to_labels is not None, "`domains_to_labels` function not provided"

        self.net_encoder = Encoder(
            domain_dims=domain_dims,
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            latent_size=latent_size,
            dropout=dropout,
            encode_domain=test_domain_known,
        )
        self.optimizer_generator = torch.optim.Adam(
            self.net_encoder.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay
        )
        self.lr_sch_generator = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_generator, gamma=gamma)

        self.net_discriminator = Discriminator(
            hidden_size=discriminator_hidden_size, latent_size=latent_size, domain_dims=domain_dims
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.net_discriminator.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay
        )
        self.lr_sch_discriminator = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_discriminator, gamma=gamma
        )

        self.lr_schedulers = [self.lr_sch_generator, self.lr_sch_discriminator]
        self.device = None
        self.lambda_gan = lambda_gan
        self.domains_to_labels = domains_to_labels
        self.verbose = verbose
        self.metrics = metrics
        self.save_fn = save_fn
        self.save_metric = save_metric
        self.loss = loss

        init_weights(self.net_encoder)

    def forward(self, x, domain):
        y_pred, encoded = self.net_encoder(x.float(), domain.float())
        y_pred = torch.squeeze(y_pred)
        return y_pred, encoded

    def backward_discriminator(self, encoded, domain, is_train):
        domain_pred = self.net_discriminator(encoded.detach())
        D_src = neg_guassian_likelihood(domain_pred[is_train], domain[is_train])
        D_tgt = neg_guassian_likelihood(domain_pred[~is_train], domain[~is_train])
        loss_discriminator = (D_src + D_tgt) / 2
        loss_discriminator.backward()

    def backward_generator(self, epoch, encoded, y_pred, domain, y, is_train):
        domain_pred = self.net_discriminator(encoded)

        E_gan_src = neg_guassian_likelihood(domain_pred[is_train], domain[is_train])
        E_gan_tgt = neg_guassian_likelihood(domain_pred[~is_train], domain[~is_train])

        loss_E_gan = -(E_gan_src + E_gan_tgt) / 2
        loss_E_pred = self.loss(y[is_train], y_pred[is_train])

        if hasattr(self.lambda_gan, "__call__"):
            lambda_gan = self.lambda_gan(epoch)
        else:
            lambda_gan = self.lambda_gan

        loss_encoder = loss_E_gan * lambda_gan + loss_E_pred
        loss_encoder.backward()

    def _fit_batch(self, epoch, x, y, domain, is_train):
        y_pred, encoded = self.forward(x, domain)

        set_requires_grad(self.net_discriminator, True)
        self.optimizer_discriminator.zero_grad()
        self.backward_discriminator(encoded, domain, is_train)
        self.optimizer_discriminator.step()

        set_requires_grad(self.net_discriminator, False)
        self.optimizer_generator.zero_grad()
        self.backward_generator(epoch, encoded, y_pred, domain, y, is_train)
        self.optimizer_generator.step()

    def _fit_epoch(self, epoch, dataloader):
        self.train()
        if self.verbose:
            dataloader = tqdm.tqdm(dataloader)
        for batch in dataloader:
            x, y, domain, is_train = [ensure_tensor(_, self.device) for _ in batch]
            self._fit_batch(epoch, x.float(), y.float(), domain.float(), is_train)
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def predict(self, batch):
        x, y, domain, is_train = [ensure_tensor(_, self.device) for _ in batch]
        y_pred, _ = self.forward(x, domain)
        return ensure_numpy(y_pred)

    def _eval(self, test_dataloader):
        self.eval()

        domains = []
        is_trains = []
        y_preds = []
        ys = []

        for batch in test_dataloader:

            x, y, domain, is_train = [ensure_tensor(_, self.device) for _ in batch]
            y_pred, encoded = self.forward(x, domain)

            domains.append(ensure_numpy(domain))
            is_trains.append(ensure_numpy(is_train))
            y_preds.append(ensure_numpy(y_pred))
            ys.append(ensure_numpy(y))

        domains = np.vstack(domains)
        domain_labels = self.domains_to_labels(domains)
        is_trains = np.hstack(is_trains)
        y_preds = np.hstack(y_preds)
        ys = np.hstack(ys)

        metric_vals = {}
        for metric_name, metric_func in self.metrics.items():
            metric_vals["train_" + metric_name] = metric_func(y_preds[is_trains], ys[is_trains])
            metric_vals["test_" + metric_name] = metric_func(y_preds[~is_trains], ys[~is_trains])
            for domain_label in sorted(np.unique(domain_labels)):
                metric_vals[domain_label + "_" + metric_name] = metric_func(
                    y_preds[domain_labels == domain_label], ys[domain_labels == domain_label]
                )

        return metric_vals

    def fit(self, dataloader, val_dataloader, epochs=100):
        self.device = next(self.parameters()).device
        best_score = 0
        for epoch in range(epochs):
            if self.verbose:
                print("Epoch {}/{}".format(epoch + 1, epochs))
            self._fit_epoch(epoch, dataloader)
            metrics = self._eval(val_dataloader)
            if metrics[self.save_metric] > best_score:
                best_score = metrics[self.save_metric]
                if self.verbose:
                    print("-> New Best!")
            if self.verbose:
                print(
                    " - ".join(
                        [
                            "{}: {:.6f}".format(k, m_val)
                            for k, m_val in metrics.items()
                            if k.startswith("train_") or k.startswith("test_")
                        ]
                    )
                )
                for metric_name in self.metrics.keys():
                    print(
                        metric_name + ":",
                        {
                            k.replace("_" + metric_name, ""): round(m_val, 3)
                            for k, m_val in metrics.items()
                            if k.endswith("_" + metric_name) and not (k.startswith("train_") or k.startswith("test_"))
                        },
                    )
                print()
        self.eval()