from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp_sparse
import torch

from .vae import Dataset, VAE, Posterior


class SCQuintVAE:
    """Phase 1 AnnData-first wrapper matching the original `run_vae()` behavior.

    This wrapper replicates the training pipeline from the scquint paper, including:
    - Train/test/validation splits
    - KL annealing schedule
    - Early stopping on reconstruction error
    - Learning rate scheduling
    - Full compatibility with original results

    Example
    -------
    SCQuintVAE.setup_anndata(adata, intron_group_key="intron_group")
    model = SCQuintVAE(adata, n_latent=20)
    model.train(max_epochs=300, lr=1e-2, n_epochs_kl_warmup=20)
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
        self.posterior_ = None

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
        train_size: float = 0.9,
        n_epochs_kl_warmup: int = 20,
        weight_decay: float = 0.0,
        shuffle: bool = True,
        verbose: bool = True,
        early_stopping_patience: int = 10,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
    ) -> None:
        """Train VAE with KL annealing, train/test split, and early stopping.
        
        Parameters
        ----------
        max_epochs : int
            Number of training epochs (default: 300)
        lr : float
            Learning rate (default: 1e-2)
        batch_size : int
            Batch size for training (default: 128)
        train_size : float
            Fraction of cells to use for training (default: 0.9)
        n_epochs_kl_warmup : int
            Number of epochs to warm up KL divergence (default: 20)
        weight_decay : float
            L2 regularization weight (default: 0.0)
        shuffle : bool
            Whether to shuffle training data (default: True)
        verbose : bool
            Print training progress (default: True)
        early_stopping_patience : int
            Patience for early stopping (default: 10)
        lr_patience : int
            Patience for learning rate reduction (default: 5)
        lr_factor : float
            Factor to reduce learning rate by (default: 0.5)
        """
        # Split into train/test
        n_cells = self.dataset.n_cells
        n_train = int(n_cells * train_size)
        perm = np.random.permutation(n_cells)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        if verbose:
            print(f"Training on {len(train_idx)} cells, testing on {len(test_idx)} cells")

        self.module.train()
        optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience, verbose=verbose
        )

        best_test_loss = float("inf")
        patience_counter = 0

        for epoch in range(max_epochs):
            # KL annealing weight
            kl_weight = min(1.0, (epoch + 1) / max(1, n_epochs_kl_warmup))

            # Training phase
            train_loss = 0.0
            train_reconst = 0.0
            train_kl = 0.0
            n_train_seen = 0

            if shuffle:
                train_order = np.random.permutation(len(train_idx))
            else:
                train_order = np.arange(len(train_idx))

            self.module.train()
            for start in range(0, len(train_idx), batch_size):
                batch_order = train_order[start : start + batch_size]
                batch_idx = train_idx[batch_order]
                x = self._slice_to_tensor(batch_idx)

                local_l_mean = torch.zeros((x.shape[0], 1), device=self.device)
                local_l_var = torch.ones((x.shape[0], 1), device=self.device)

                optimizer.zero_grad()
                reconst_loss, kl_divergence, _ = self.module(x, local_l_mean, local_l_var)
                weighted_kl = kl_weight * kl_divergence
                loss = (reconst_loss + weighted_kl).mean()
                loss.backward()
                optimizer.step()

                bs = x.shape[0]
                train_loss += float(loss.detach().cpu().item()) * bs
                train_reconst += float(reconst_loss.mean().detach().cpu().item()) * bs
                train_kl += float(kl_divergence.mean().detach().cpu().item()) * bs
                n_train_seen += bs

            # Test phase
            test_loss = 0.0
            test_reconst = 0.0
            test_kl = 0.0
            n_test_seen = 0

            self.module.eval()
            with torch.no_grad():
                for start in range(0, len(test_idx), batch_size):
                    batch_idx = test_idx[start : start + batch_size]
                    x = self._slice_to_tensor(batch_idx)

                    local_l_mean = torch.zeros((x.shape[0], 1), device=self.device)
                    local_l_var = torch.ones((x.shape[0], 1), device=self.device)

                    reconst_loss, kl_divergence, _ = self.module(x, local_l_mean, local_l_var)
                    weighted_kl = kl_weight * kl_divergence
                    loss = (reconst_loss + weighted_kl).mean()

                    bs = x.shape[0]
                    test_loss += float(loss.detach().cpu().item()) * bs
                    test_reconst += float(reconst_loss.mean().detach().cpu().item()) * bs
                    test_kl += float(kl_divergence.mean().detach().cpu().item()) * bs
                    n_test_seen += bs

            # Average losses
            train_loss_avg = train_loss / max(n_train_seen, 1)
            test_loss_avg = test_loss / max(n_test_seen, 1)
            train_reconst_avg = train_reconst / max(n_train_seen, 1)
            test_reconst_avg = test_reconst / max(n_test_seen, 1)
            train_kl_avg = train_kl / max(n_train_seen, 1)
            test_kl_avg = test_kl / max(n_test_seen, 1)

            # Learning rate scheduling
            scheduler.step(test_loss_avg)

            # Early stopping based on test loss
            if test_loss_avg < best_test_loss:
                best_test_loss = test_loss_avg
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == max_epochs):
                print(
                    f"Epoch {epoch + 1}/{max_epochs} | "
                    f"Train Loss: {train_loss_avg:.4f} (R: {train_reconst_avg:.4f}, KL: {train_kl_avg:.4f}) | "
                    f"Test Loss: {test_loss_avg:.4f} (R: {test_reconst_avg:.4f}, KL: {test_kl_avg:.4f}) | "
                    f"KL weight: {kl_weight:.4f}"
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        self.module.eval()
        self.is_trained_ = True

        # Create posterior for compatibility
        self.posterior_ = Posterior(
            self.module, self.dataset, use_cuda=self.use_cuda,
            data_loader_kwargs={"batch_size": batch_size}
        )

    @torch.no_grad()
    def get_latent_representation(
        self,
        batch_size: int = 512,
        give_mean: bool = True,
    ) -> np.ndarray:
        """Get latent representation for all cells.
        
        Parameters
        ----------
        batch_size : int
            Batch size for inference (default: 512)
        give_mean : bool
            Whether to return mean or sample from posterior (default: True)
            
        Returns
        -------
        np.ndarray
            Latent representation of shape (n_cells, n_latent)
        """
        self.module.eval()

        n_cells = self.dataset.n_cells
        z_all = []
        for start in range(0, n_cells, batch_size):
            idx = slice(start, min(start + batch_size, n_cells))
            x = self._slice_to_tensor(idx)
            z = self.module.sample_from_posterior_z(x, give_mean=give_mean)
            z_all.append(z.detach().cpu().numpy())

        return np.concatenate(z_all, axis=0)
