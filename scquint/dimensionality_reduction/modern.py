from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp_sparse
import torch

from .vae import Dataset, VAE


class SCQuintVAE:
    """Phase 1 AnnData-first wrapper with modern scvi-tools-like API.

    Example
    -------
    SCQuintVAE.setup_anndata(adata, intron_group_key="intron_group")
    model = SCQuintVAE(adata, n_latent=20)
    model.train(max_epochs=300, lr=1e-2)
    z = model.get_latent_representation()
    """

    _REGISTRY_KEY = "_scquint_registry"

    @staticmethod
    def setup_anndata(adata, intron_group_key: str = "intron_group") -> None:
        if intron_group_key not in adata.var.columns:
            raise KeyError(
                f"`{intron_group_key}` not found in `adata.var`. "
                "Please add intron group ids before calling setup_anndata()."
            )

        adata.uns[SCQuintVAE._REGISTRY_KEY] = {
            "intron_group_key": intron_group_key,
        }

    def __init__(
        self,
        adata,
        n_latent: int = 20,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.25,
        use_cuda: Optional[bool] = None,
        linearly_decoded: bool = True,
        loss_introns: str = "dirichlet-multinomial",
        input_transform: str = "log",
        regularization_gaussian_std=None,
        feature_addition=None,
    ):
        if self._REGISTRY_KEY not in adata.uns:
            raise RuntimeError(
                "AnnData is not set up. Call `SCQuintVAE.setup_anndata(adata, ...)` first."
            )

        intron_group_key = adata.uns[self._REGISTRY_KEY]["intron_group_key"]
        if intron_group_key != "intron_group":
            adata.var["intron_group"] = adata.var[intron_group_key].values

        self.adata = adata
        self.dataset = Dataset(adata)

        if use_cuda is None:
            use_cuda = torch.cuda.is_available()

        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        self.module = VAE(
            self.dataset.n_genes,
            self.dataset.n_introns,
            self.dataset.n_intron_groups,
            self.dataset.intron_groups,
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_cuda=self.use_cuda,
            loss_introns=loss_introns,
            linearly_decoded=linearly_decoded,
            input_transform=input_transform,
            feature_addition=feature_addition,
            linearly_encoded=False,
            regularization_gaussian_std=regularization_gaussian_std,
        ).to(self.device)

        self.is_trained_ = False

    def _slice_to_tensor(self, idx):
        x = self.adata.X[idx]
        if sp_sparse.issparse(x):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x).to(self.device)

    def train(
        self,
        max_epochs: int = 300,
        lr: float = 1e-2,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> None:
        self.module.train()
        optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)

        n_cells = self.dataset.n_cells
        order = np.arange(n_cells)

        for epoch in range(max_epochs):
            if shuffle:
                np.random.shuffle(order)

            total_loss = 0.0
            n_seen = 0

            for start in range(0, n_cells, batch_size):
                idx = order[start : start + batch_size]
                x = self._slice_to_tensor(idx)

                local_l_mean = torch.zeros((x.shape[0], 1), device=self.device)
                local_l_var = torch.ones((x.shape[0], 1), device=self.device)

                optimizer.zero_grad()
                reconst_loss, kl_divergence, _ = self.module(x, local_l_mean, local_l_var)
                loss = (reconst_loss + kl_divergence).mean()
                loss.backward()
                optimizer.step()

                bs = x.shape[0]
                total_loss += float(loss.detach().cpu().item()) * bs
                n_seen += bs

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == max_epochs):
                print(f"Epoch {epoch + 1}/{max_epochs} - loss: {total_loss / max(n_seen, 1):.4f}")

        self.module.eval()
        self.is_trained_ = True

    @torch.no_grad()
    def get_latent_representation(
        self,
        batch_size: int = 512,
        give_mean: bool = True,
    ) -> np.ndarray:
        self.module.eval()

        n_cells = self.dataset.n_cells
        z_all = []
        for start in range(0, n_cells, batch_size):
            idx = slice(start, min(start + batch_size, n_cells))
            x = self._slice_to_tensor(idx)
            z = self.module.sample_from_posterior_z(x, give_mean=give_mean)
            z_all.append(z.detach().cpu().numpy())

        return np.concatenate(z_all, axis=0)
