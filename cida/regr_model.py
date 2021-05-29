import torch.nn.functional as F
import torch.nn as nn
import torch

from cida.util import (
    ensure_tensor,
    ensure_numpy,
    init_weights,
    set_requires_grad,
    neg_guassian_likelihood,
)

import numpy as np
import tqdm


class Encoder(nn.Module):
    def __init__(self, *, domain_dims, output_dims, input_dims, hidden_size, latent_size, dropout, encode_domain):
        super(Encoder, self).__init__()

        self.fc_feats = nn.Sequential(
            nn.Linear(domain_dims + input_dims if encode_domain else input_dims, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
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
            nn.Linear(hidden_size, output_dims),
        )

        self.input_dims = input_dims
        self.encode_domain = encode_domain

    def forward(self, x, domain):
        if self.encode_domain:
            input_ = torch.cat([domain, x], 1)
        else:
            input_ = x
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
        target_gan_weight=0.5,
        domain_dims=1,
        output_dims=1,
        encoder_hidden_dims=256,
        discriminator_hidden_dims=512,
        latent_size=100,
        input_dims=100,
        loss=F.mse_loss,
        domains_to_labels=None,
        test_domain_known=True,
        test_metric_gap=None,
        verbose=False,
        metrics={},
    ):
        super(PCIDARegressor, self).__init__()

        assert len(metrics) > 0, "No metrics provided"
        assert domains_to_labels is not None, "`domains_to_labels` function not provided"

        self.net_encoder = Encoder(
            domain_dims=domain_dims,
            output_dims=output_dims,
            input_dims=input_dims,
            hidden_size=encoder_hidden_dims,
            latent_size=latent_size,
            dropout=dropout,
            encode_domain=test_domain_known,
        )
        self.optimizer_generator = torch.optim.Adam(
            self.net_encoder.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay
        )
        self.lr_sch_generator = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_generator, gamma=gamma)

        self.net_discriminator = Discriminator(
            hidden_size=discriminator_hidden_dims, latent_size=latent_size, domain_dims=domain_dims
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
        self.target_gan_weight = target_gan_weight
        self.domains_to_labels = domains_to_labels
        self.verbose = verbose
        self.metrics = metrics
        self.loss = loss
        self.output_dims = output_dims
        self.test_metric_gap = test_metric_gap

        init_weights(self.net_encoder)

    def forward(self, x, domain):
        y_pred, encoded = self.net_encoder(x.float(), domain.float())
        if self.output_dims == 1:
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

        loss_E_gan = -(E_gan_src * (1 - self.target_gan_weight) + E_gan_tgt * self.target_gan_weight)
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
        x, _, domain, _ = [ensure_tensor(_, self.device) for _ in batch]
        y_pred, _ = self.forward(x, domain)
        return ensure_numpy(y_pred)

    def eval_on_data(self, test_dataloader):
        self.eval()

        domains = []
        is_trains = []
        y_preds = []
        ys = []

        for batch in test_dataloader:

            x, y, domain, is_train = [ensure_tensor(_, self.device) for _ in batch]
            y_pred, _ = self.forward(x, domain)

            domains.append(ensure_numpy(domain))
            is_trains.append(ensure_numpy(is_train))
            y_preds.append(ensure_numpy(y_pred))
            ys.append(ensure_numpy(y))

        domains = np.vstack(domains)
        domain_labels = self.domains_to_labels(domains)
        is_trains = np.hstack(is_trains)
        if self.output_dims == 1:
            y_preds = np.hstack(y_preds)
            ys = np.hstack(ys)
            is_nan = np.isnan(ys)
        else:
            y_preds = np.vstack(y_preds)
            ys = np.vstack(ys)
            is_nan = np.isnan(ys)[:, 0]

        test_domains = sorted(set(domain_labels[~is_trains]))
        if self.test_metric_gap is None:
            test_mask = np.array([True for _ in domain_labels])
        else:
            test_mask = np.array([dl in test_domains[self.test_metric_gap :] for dl in domain_labels])
        test_mask &= ~is_trains & ~is_nan

        metric_vals = {}
        for metric_name, metric_func in self.metrics.items():
            metric_vals["train_" + metric_name] = metric_func(y_preds[is_trains], ys[is_trains])
            metric_vals["test_" + metric_name] = metric_func(y_preds[test_mask], ys[test_mask])
            for domain_label in sorted(np.unique(domain_labels)):
                metric_vals[domain_label + "_" + metric_name] = metric_func(
                    y_preds[domain_labels == domain_label], ys[domain_labels == domain_label]
                )

        return metric_vals

    def fit(
        self,
        dataloader,
        val_dataloader,
        epochs=100,
        save_metric="test_mse",
        save_fn="cida-best.pth",
        per_era_metrics=True,
    ):
        self.device = next(self.parameters()).device
        metric_hist = []
        best_score = None
        for epoch in range(epochs):
            if self.verbose:
                print("Epoch {}/{}".format(epoch + 1, epochs))
            self._fit_epoch(epoch, dataloader)
            metrics = self.eval_on_data(val_dataloader)
            metric_hist.append(metrics)
            if save_metric is None:
                self.save(save_fn)
            elif best_score is None or metrics[save_metric] > best_score:
                best_score = metrics[save_metric]
                self.save(save_fn)
                if self.verbose:
                    print("-> New Best!")
            if self.verbose:
                _print_metrics(self.metrics, metrics, per_era_metrics)

        self.eval()
        return metric_hist

    def save(self, fn):
        return torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))


def _print_metrics(metric_types, metric_values, per_era_metrics):
    print(
        " - ".join(
            [
                "{}: {:.6f}".format(k, m_val)
                for k, m_val in metric_values.items()
                if k.startswith("train_") or k.startswith("test_")
            ]
        )
    )
    if per_era_metrics:
        for metric_name in metric_types.keys():
            print(
                metric_name + ":",
                {
                    k.replace("_" + metric_name, ""): round(m_val, 3)
                    for k, m_val in metric_values.items()
                    if k.endswith("_" + metric_name) and not (k.startswith("train_") or k.startswith("test_"))
                },
            )
    print()
