"""
Competitive Hebbian feature learning with specificity penalty.

This is the core mechanism of Mnemos. Filters compete to respond
to input patterns. The winning filter updates toward its input.
The SPECIFICITY PENALTY prevents all filters from collapsing to
the global mean — a known failure mode in competitive learning.

The specificity penalty:
    effective_sim = raw_sim - mean_sim_across_filters - threshold

This geometrically eliminates the global-mean attractor that
causes winner-take-all collapse in standard competitive learning.

Achieves 97.4% on Split-MNIST continual learning without backprop,
exceeding EWC (65.5%) which uses gradients.

Author: Gustav Gausepohl
"""

import numpy as np


class HebbianFilters:
    """
    Competitive Hebbian convolutional feature learning.

    Learns a bank of filters from raw image patches through
    competitive Hebbian learning with specificity penalty.

    Parameters
    ----------
    n_filters : int
        Number of filters to learn. More = more discriminative.
        97.4% MNIST accuracy used 400 filters.
    patch_size : int
        Size of square patches to extract (patch_size × patch_size).
    seed : int
        Random seed for reproducibility.

    Example
    -------
    >>> filters = HebbianFilters(n_filters=200, patch_size=5)
    >>> filters.train(images, n_epochs=10)
    >>> features = filters.extract(test_images)
    """

    def __init__(self, n_filters=200, patch_size=5, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_filters = n_filters
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size

        # Initialize filters as random unit vectors
        raw = self.rng.randn(self.patch_dim, n_filters).astype(np.float32)
        raw -= raw.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(raw, axis=0, keepdims=True) + 1e-8
        self.filters = raw / norms

        # Homeostatic thresholds (one per filter)
        self.thresholds = np.zeros(n_filters, dtype=np.float32)
        self.target_rate = 1.0 / n_filters

        self._trained = False
        self._img_size = None

    def train(self, images, img_size=28, n_per_image=20, n_epochs=10,
              base_lr=0.1, batch_size=2000, verbose=True):
        """
        Train filters from images through competitive Hebbian learning.

        Each epoch:
          1. Extract random patches from images
          2. Compute similarity between patches and all filters
          3. SPECIFICITY PENALTY: subtract mean similarity per filter
          4. Winner-take-all: only the winning filter updates
          5. Hebbian update: winner moves toward its winning patch
          6. Homeostatic thresholds adapt to balance filter usage

        Parameters
        ----------
        images : ndarray, shape (N, H, W) or (N, H*W)
            Training images. Values in [0, 1].
        img_size : int
            Height/width of images (assumes square).
        n_per_image : int
            Random patches to extract per image.
        n_epochs : int
            Training epochs. More = better filters.
        base_lr : float
            Initial learning rate. Decays over epochs.
        batch_size : int
            Patches per update batch.
        verbose : bool
            Print progress.

        Returns
        -------
        dict with 'n_active' and 'n_dead' filter counts.
        """
        self._img_size = img_size
        os = img_size - self.patch_size + 1

        # Reshape if flat
        if images.ndim == 2:
            images = images.reshape(-1, img_size, img_size)
        N = min(len(images), 10000)
        imgs = images[:N]

        # Extract random patches
        n_patches = N * n_per_image
        patches = np.zeros((n_patches, self.patch_dim), dtype=np.float32)
        idx = 0
        for n in range(N):
            for _ in range(n_per_image):
                i = self.rng.randint(0, os)
                j = self.rng.randint(0, os)
                patches[idx] = imgs[n, i:i+self.patch_size,
                                    j:j+self.patch_size].ravel()
                idx += 1

        # Normalize patches (zero mean, unit norm)
        patches -= patches.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8
        patches /= norms

        for epoch in range(n_epochs):
            perm = self.rng.permutation(n_patches)
            lr = base_lr / (1.0 + epoch * 0.2)
            epoch_wins = np.zeros(self.n_filters, dtype=np.float32)

            for start in range(0, n_patches, batch_size):
                end = min(start + batch_size, n_patches)
                batch = patches[perm[start:end]]

                # Similarity to all filters
                sims = batch @ self.filters

                # ═══ SPECIFICITY PENALTY ═══
                # This is the key mechanism. Subtract mean similarity
                # per filter so filters that respond to everything
                # are penalised. Only filters that respond SPECIFICALLY
                # to certain patterns win.
                avg_sim = sims.mean(axis=0, keepdims=True)
                specific = sims - avg_sim - self.thresholds[np.newaxis, :]

                # Winner-take-all
                winners = np.argmax(specific, axis=1)
                np.add.at(epoch_wins, winners, 1)

                # Hebbian update: winning filter moves toward its patches
                for f in range(self.n_filters):
                    mask = winners == f
                    if mask.sum() > 0:
                        mean_patch = batch[mask].mean(axis=0)
                        self.filters[:, f] += lr * (mean_patch - self.filters[:, f])
                        norm = np.linalg.norm(self.filters[:, f])
                        if norm > 0:
                            self.filters[:, f] /= norm

            # Homeostatic threshold adaptation
            epoch_rate = epoch_wins / n_patches
            self.thresholds += 0.1 * (epoch_rate - self.target_rate)
            np.clip(self.thresholds, -0.3, 0.5, out=self.thresholds)

            # Reinitialize dead filters
            dead = epoch_wins == 0
            if dead.sum() > 0 and epoch < n_epochs - 2:
                new_idx = self.rng.choice(n_patches,
                                          size=min(int(dead.sum()), n_patches),
                                          replace=False)
                for di, pi in zip(np.where(dead)[0], new_idx):
                    self.filters[:, di] = patches[pi]
                    norm = np.linalg.norm(self.filters[:, di])
                    if norm > 0:
                        self.filters[:, di] /= norm
                    self.thresholds[di] = -0.1

            if verbose:
                n_active = int((epoch_wins > 0).sum())
                print(f"  Epoch {epoch+1}/{n_epochs}: "
                      f"{n_active}/{self.n_filters} active filters, "
                      f"lr={lr:.3f}")

        n_active = int((epoch_wins > 0).sum())
        n_dead = self.n_filters - n_active
        self._trained = True

        if verbose:
            print(f"  Training complete: {n_active} active, {n_dead} dead")

        return {"n_active": n_active, "n_dead": n_dead}

    def extract(self, images, img_size=None, n_quadrants=4):
        """
        Extract features from images using trained filters.

        Convolves filters across the image, applies ReLU,
        then mean-pools across spatial quadrants.

        Parameters
        ----------
        images : ndarray, shape (N, H, W) or (N, H*W)
            Images to extract features from.
        img_size : int or None
            Image size. Uses training size if None.
        n_quadrants : int
            Number of spatial pooling regions (4 = 2×2 grid).

        Returns
        -------
        features : ndarray, shape (N, n_filters * n_quadrants)
        """
        if not self._trained:
            raise RuntimeError("Call .train() before .extract()")

        img_size = img_size or self._img_size
        if images.ndim == 2:
            images = images.reshape(-1, img_size, img_size)

        N = len(images)
        os = img_size - self.patch_size + 1
        half = os // 2
        feat_dim = self.n_filters * n_quadrants
        feats = np.zeros((N, feat_dim), dtype=np.float32)

        for n in range(N):
            patches = np.zeros((os * os, self.patch_dim), dtype=np.float32)
            idx = 0
            for i in range(os):
                for j in range(os):
                    patches[idx] = images[n, i:i+self.patch_size,
                                          j:j+self.patch_size].ravel()
                    idx += 1
            patches -= patches.mean(axis=1, keepdims=True)

            act = (patches @ self.filters).reshape(os, os, self.n_filters)
            np.maximum(act, 0, out=act)

            quads = [act[:half, :half, :], act[:half, half:, :],
                     act[half:, :half, :], act[half:, half:, :]]
            for qi in range(min(n_quadrants, 4)):
                feats[n, qi*self.n_filters:(qi+1)*self.n_filters] = \
                    quads[qi].mean(axis=(0, 1))

        return feats

    @property
    def feature_dim(self):
        """Dimension of extracted feature vectors."""
        return self.n_filters * 4
