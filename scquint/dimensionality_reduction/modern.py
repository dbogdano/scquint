from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp_sparse
import torch

from .vae import Dataset, VAE, Posterior, UnsupervisedTrainer, run_vae


class SCQuintVAE:
    """Phase 1 AnnData-first wrapper with modern scvi-tools-like API.

    This wrapper provides a modern scvi-tools-style interface while delegating to the 
    proven `run_vae()` implementation for training to ensure compatibility with the 
    original scquint paper results.

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
        """Register intron group information in AnnData object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object with intron data
        intron_group_key : str
            Column name in `adata.var` containing intron group assignments
        """
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
        """Initialize SCQuint VAE model.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object (must be setup with setup_anndata first)
        n_latent : int
            Dimensionality of latent space (default: 20)
        n_layers : int
            Number of hidden layers in encoder/decoder (default: 1)
        n_hidden : int
            Number of hidden units (default: 128)
        dropout_rate : float
            Dropout probability (default: 0.25)
        use_cuda : bool or None
            Whether to use GPU; auto-detect if None (default: None)
        linearly_decoded : bool
            Use linear decoder for introns (default: True, matches paper)
        loss_introns : str
            Loss function for introns: "dirichlet-multinomial" or "multinomial" (default: "dirichlet-multinomial")
        input_transform : str
            Input transformation: "log" or "frequency-smoothed" (default: "log")
        regularization_gaussian_std : float or None
            Gaussian regularization std for weights (default: None)
        feature_addition : array-like or None
            Feature addition for frequency-smoothed transform (default: None)
        """
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

        # Store hyperparameters
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.loss_introns = loss_introns
        self.input_transform = input_transform
        self.regularization_gaussian_std = regularization_gaussian_std
        self.feature_addition = feature_addition

        self.is_trained_ = False
        self.posterior_ = None
        self.trainer_ = None

    def train(
        self,
        max_epochs: int = 300,
        lr: float = 1e-2,
        batch_size: int = 128,
        n_epochs_kl_warmup: int = 20,
        verbose: bool = True,
    ) -> "SCQuintVAE":
        """Train VAE using the proven `run_vae()` pipeline.
        
        Delegates to the original run_vae() to ensure full compatibility with 
        the scquint paper's training procedure.
        
        Parameters
        ----------
        max_epochs : int
            Number of training epochs (default: 300)
        lr : float
            Learning rate (default: 1e-2)
        batch_size : int
            Batch size for training (default: 128)
        n_epochs_kl_warmup : int
            Number of epochs to warm up KL divergence (default: 20)
        verbose : bool
            Print training progress (default: True)
            
        Returns
        -------
        self
            For method chaining
        """
        if verbose:
            print(f"Training VAE with:")
            print(f"  max_epochs: {max_epochs}")
            print(f"  lr: {lr}")
            print(f"  batch_size: {batch_size}")
            print(f"  n_epochs_kl_warmup: {n_epochs_kl_warmup}")

        # Use original run_vae() for training
        latent, trained_model = run_vae(
            self.adata,
            n_epochs=max_epochs,
            use_cuda=self.use_cuda,
            n_latent=self.n_latent,
            n_layers=self.n_layers,
            dropout_rate=self.dropout_rate,
            n_hidden=self.n_hidden,
            lr=lr,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            linearity="linear" if self.module.introns_decoder.__class__.__name__ == "LinearIntronsDecoder" else "nonlinear",
            loss_introns=self.loss_introns,
            input_transform=self.input_transform,
            regularization_gaussian_std=self.regularization_gaussian_std,
            feature_addition=self.feature_addition,
            sample=True,
        )

        # Update module with trained weights
        self.module = trained_model
        self.is_trained_ = True
        self.latent_ = latent

        if verbose:
            print(f"Training complete. Latent shape: {latent.shape}")

        return self

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
        if not self.is_trained_:
            raise RuntimeError("Model must be trained before calling get_latent_representation()")

        if hasattr(self, "latent_"):
            return self.latent_

        # Fallback: compute on the fly
        self.module.eval()
        n_cells = self.dataset.n_cells
        z_all = []

        for start in range(0, n_cells, batch_size):
            idx = slice(start, min(start + batch_size, n_cells))
            x = self.adata.X[idx]
            if sp_sparse.issparse(x):
                x = x.toarray()
            x = np.asarray(x, dtype=np.float32)
            x = torch.from_numpy(x).to(self.device)

            z = self.module.sample_from_posterior_z(x, give_mean=give_mean)
            z_all.append(z.detach().cpu().numpy())

        return np.concatenate(z_all, axis=0)
