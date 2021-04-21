from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from collections import Counter
import numpy as np
import torch
import tqdm
import os


class RotateMNIST(Dataset):
    def __init__(self, data_dir, rotate_range=(0, 360), train_range=(0, 45)):
        self.data, self.targets = torch.load(data_dir)
        self.rotate_range = rotate_range
        self.train_range = train_range

    def __getitem__(self, index):
        from torchvision.transforms.functional import rotate
        from PIL import Image

        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")

        rot_min, rot_max = self.rotate_range
        angle = np.random.rand() * (rot_max - rot_min) + rot_min

        img = torchvision.transforms.functional.rotate(img, angle)
        img = torchvision.transforms.ToTensor()(img).to(torch.float)

        train_min, train_max = self.train_range
        is_train = angle >= train_min and angle <= train_max

        return img, target, np.array([angle / 360.0], dtype=np.float32), is_train

    def __len__(self):
        return len(self.data)

    @staticmethod
    def domains_to_labels(domain):
        return np.array(["{}-{}°".format(int(angle[0] * 8) * 45, int(angle[0] * 8 + 1) * 45) for angle in domain])


def ensure_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x


def ensure_numpy(x):
    return x.detach().cpu().numpy()


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, val=0)


def set_requires_grad(nets, requires_grad):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def convert_Avec_to_A(A_vec):
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)

    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 3:
        A_dim = 2
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.triu_indices(A_dim, A_dim)
    A = A_vec.new_zeros((A_vec.shape[0], A_dim, A_dim))
    A[:, idx[0], idx[1]] = A_vec
    A[:, idx[1], idx[0]] = A_vec
    return A.squeeze()


def neg_guassian_likelihood(d, u):
    """-N(u; mu, var)"""
    B, dim = u.shape
    assert d.shape[1] == dim * 2
    mu, logvar = d[:, :dim], d[:, dim:]
    return 0.5 * (((u - mu) ** 2) / torch.exp(logvar) + logvar).mean()


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class EncoderSTN(nn.Module):
    def __init__(self, *, domain_dims, input_size, hidden_size, latent_size, dropout, classes):
        super(EncoderSTN, self).__init__()

        self.fc_stn = nn.Sequential(
            nn.Linear(domain_dims + input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_size, 3, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, latent_size, 4, 1, 0),
            nn.ReLU(True),
        )

        self.fc_pred = nn.Sequential(
            nn.Conv2d(latent_size, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            Squeeze(),
            nn.Linear(hidden_size, classes),
        )

        self.input_size = input_size

    def spatial_transformation(self, x, domain):
        A_vec = self.fc_stn(torch.cat([domain, x.reshape(-1, self.input_size)], 1))
        A = convert_Avec_to_A(A_vec)
        _, evs = torch.symeig(A, eigenvectors=True)
        tcos, tsin = evs[:, 0:1, 0:1], evs[:, 1:2, 0:1]

        theta_0 = torch.cat([tcos, tsin, tcos * 0], 2)
        theta_1 = torch.cat([-tsin, tcos, tcos * 0], 2)
        theta = torch.cat([theta_0, theta_1], 1)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x, domain):
        transformed = self.spatial_transformation(x, domain)
        conv_feats = self.conv(transformed)
        y = self.fc_pred(conv_feats)
        return F.log_softmax(y, dim=1), transformed, conv_feats


class DiscriminatorConv(nn.Module):
    def __init__(self, *, hidden_size, latent_size, domain_dims):
        super(DiscriminatorConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_size, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(),
            Squeeze(),
            nn.Linear(hidden_size, domain_dims * 2),
        )

    def forward(self, x):
        return self.net(x)


class ConvPCIDAClassifier(nn.Module):
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
        input_size=784,
        classes=10,
        domains_to_labels=None,
        verbose=False,
        save_fn="cida-best-acc.pth",
    ):
        super(ConvPCIDAClassifier, self).__init__()

        self.net_encoder = EncoderSTN(
            domain_dims=domain_dims,
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            latent_size=latent_size,
            dropout=dropout,
            classes=classes,
        )
        self.optimizer_generator = torch.optim.Adam(
            self.net_encoder.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay
        )
        self.lr_sch_generator = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_generator, gamma=gamma)

        self.net_discriminator = DiscriminatorConv(
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
        self.save_fn = save_fn

        init_weights(self.net_encoder)

    def forward(self, x, domain):
        y_pred, _, encoded = self.net_encoder(x, domain)
        pred_labels = torch.argmax(y_pred.detach(), dim=1)
        return y_pred, encoded, pred_labels

    def backward_discriminator(self, encoded, domain, is_train):
        domain_pred = self.net_discriminator(encoded.detach())
        D_src = neg_guassian_likelihood(domain_pred[is_train], domain[is_train])
        D_tgt = neg_guassian_likelihood(domain_pred[~is_train], domain[~is_train])
        loss_discriminator = (D_src + D_tgt) / 2
        loss_discriminator.backward()

    def backward_generator(self, encoded, y_pred, domain, y, is_train):
        domain_pred = self.net_discriminator(encoded)

        E_gan_src = neg_guassian_likelihood(domain_pred[is_train], domain[is_train])
        E_gan_tgt = neg_guassian_likelihood(domain_pred[~is_train], domain[~is_train])

        loss_E_gan = -(E_gan_src + E_gan_tgt) / 2
        loss_E_pred = F.nll_loss(y_pred[is_train], y[is_train])

        loss_encoder = loss_E_gan * self.lambda_gan + loss_E_pred
        loss_encoder.backward()

    def _fit_batch(self, x, y, domain, is_train):
        y_pred, encoded, _ = self.forward(x, domain)

        set_requires_grad(self.net_discriminator, True)
        self.optimizer_discriminator.zero_grad()
        self.backward_discriminator(encoded, domain, is_train)
        self.optimizer_discriminator.step()

        set_requires_grad(self.net_discriminator, False)
        self.optimizer_generator.zero_grad()
        self.backward_generator(encoded, y_pred, domain, y, is_train)
        self.optimizer_generator.step()

    def _fit_epoch(self, dataloader):
        self.train()
        if self.verbose:
            dataloader = tqdm.tqdm(dataloader)
        for batch in dataloader:
            x, y, domain, is_train = [ensure_tensor(_, self.device) for _ in batch]
            self._fit_batch(x, y, domain, is_train)

    def predict(self, batch):
        x, y, domain, is_train = [ensure_tensor(_, self.device) for _ in batch]
        _, _, pred_labels = self.forward(x, domain)
        return ensure_numpy(pred_labels)

    def _eval(self, test_dataloader):
        self.eval()
        test_cnt, train_cnt = 0, 0
        test_correct, train_correct = 0, 0
        by_domain_label_cnt, by_domain_label_correct = Counter(), Counter()

        for batch in test_dataloader:

            x, y, domain, is_train = [ensure_tensor(_, self.device) for _ in batch]
            y_pred, encoded, pred_labels = self.forward(x, domain)

            domain = ensure_numpy(domain)
            is_train = ensure_numpy(is_train)
            y = ensure_numpy(y)
            pred_labels = ensure_numpy(pred_labels)
            correct = (y == pred_labels).astype(np.int32)

            train_cnt += is_train.sum()
            test_cnt += (~is_train).sum()
            train_correct += correct[is_train].sum()
            test_correct += correct[~is_train].sum()

            if self.domains_to_labels is not None:
                domain_labels = self.domains_to_labels(domain)
                by_domain_label_cnt.update(domain_labels)
                by_domain_label_correct.update(domain_labels[correct == 1])

        train_acc = train_correct / train_cnt
        test_acc = test_correct / test_cnt
        by_domain_acc = {
            domain_label: by_domain_label_correct[domain_label] / cnt
            for domain_label, cnt in by_domain_label_cnt.items()
        }
        return train_acc, test_acc, by_domain_acc

    def fit(self, dataloader, val_dataloader, epochs=100):
        self.device = next(self.parameters()).device
        best_acc = 0
        for epoch in range(epochs):
            if self.verbose:
                print("Epoch {}/{}".format(epoch + 1, epochs))
            self._fit_epoch(dataloader)
            train_acc, test_acc, by_domain_acc = self._eval(val_dataloader)
            if self.verbose:
                print("train_acc: {:.3f} - test_acc: {:.3f}".format(train_acc, test_acc))
                print("domain_acc:", {d: round(acc, 3) for d, acc in by_domain_acc.items()})
                print()
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.state_dict(), self.save_fn)
        self.load_state_dict(torch.load(self.save_fn))
        self.eval()


if __name__ == "__main__":
    # RotateMNIST.download()
    dataset = RotateMNIST(
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
        classes=10, input_size=28 * 28, domain_dims=1, domains_to_labels=RotateMNIST.domains_to_labels, verbose=True
    )
    model = model.to("cpu")
    model.fit(dataloader, val_dataloader, epochs=100)
    print(model.predict(next(iter(val_dataloader))))
