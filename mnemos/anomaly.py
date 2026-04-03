"""
Hebbian anomaly detection — learns 'normal' without labels.

Feed it raw data. It learns what's normal through competitive
Hebbian prototype learning. Then it flags anything that doesn't
match the learned prototypes as anomalous.

No labels needed. No training/test split. No backprop.
The detector adapts continuously — it never stops learning.

Use cases:
  - Network traffic monitoring (detect intrusions)
  - Vibration sensors (detect machine failure)
  - Heartbeat monitoring (detect arrhythmia)
  - Temperature sensors (detect equipment malfunction)
  - Any time-series where you need to learn "normal" and flag "weird"

Author: Gustav Gausepohl
"""

import numpy as np


class AnomalyDetector:
    """
    Hebbian anomaly detector.

    Learns a set of prototypes that represent "normal" patterns
    through competitive Hebbian learning. New inputs are scored
    by how far they are from the nearest prototype — high distance
    means anomaly.

    Parameters
    ----------
    n_prototypes : int
        Number of normal-pattern prototypes. More = finer model
        of normality but slower. 16-64 typical.
    threshold_percentile : float
        Percentile of training distances to set as anomaly threshold.
        95.0 means the top 5% of distances during training are
        considered anomalous. Adjust based on expected anomaly rate.
    adapt : bool
        If True, detector continues learning from non-anomalous
        inputs after initial training. The model adapts to drift.
    seed : int
        Random seed.

    Example
    -------
    >>> detector = mnemos.AnomalyDetector(n_prototypes=32)
    >>> detector.fit(normal_data)              # learn what's normal
    >>> scores = detector.score(new_data)      # anomaly scores
    >>> anomalies = detector.detect(new_data)  # True/False per sample
    >>> detector.update(new_sample)            # adapt to one new sample
    """

    def __init__(self, n_prototypes=64, threshold_percentile=95.0,
                 adapt=True, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_proto = n_prototypes
        self.threshold_pct = threshold_percentile
        self.adapt = adapt

        self.prototypes = None
        self.thresholds = None    # specificity thresholds
        self.target_rate = None
        self.threshold = None     # anomaly distance threshold
        self.dim = None

        # Statistics
        self.n_seen = 0
        self.n_anomalies = 0
        self.running_mean = None
        self.running_std = None
        self._fitted = False

    def fit(self, data, n_epochs=10, lr=0.1, verbose=True):
        """
        Learn normal patterns from data.

        Competitive Hebbian learning with specificity penalty.
        Same mechanism as HebbianFilters but applied to arbitrary
        feature vectors instead of image patches.

        Parameters
        ----------
        data : ndarray, shape (N, dim)
            Normal training data. Should contain mostly normal
            patterns (a few anomalies in training won't break it).
        n_epochs : int
            Training epochs.
        lr : float
            Learning rate.
        verbose : bool
            Print progress.

        Returns
        -------
        self
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        N, dim = data.shape
        self.dim = dim

        # Normalize data
        self.running_mean = data.mean(axis=0)
        self.running_std = data.std(axis=0) + 1e-8
        normed = (data - self.running_mean) / self.running_std

        # Initialize prototypes from random samples
        n_proto = min(self.n_proto, N)
        init_idx = self.rng.choice(N, size=n_proto, replace=False)
        self.prototypes = normed[init_idx].copy()

        # Specificity thresholds
        self.thresholds = np.zeros(n_proto, dtype=np.float32)
        self.target_rate = 1.0 / n_proto

        # Train
        for epoch in range(n_epochs):
            perm = self.rng.permutation(N)
            epoch_wins = np.zeros(n_proto, dtype=np.float32)
            epoch_lr = lr / (1.0 + epoch * 0.2)

            for i in perm:
                x = normed[i]

                # Euclidean distance to all prototypes (inverted as similarity)
                dists = np.linalg.norm(self.prototypes - x[np.newaxis, :], axis=1)
                sims = -dists  # closer = higher

                # SPECIFICITY PENALTY
                mean_sim = sims.mean()
                specific = sims - mean_sim - self.thresholds

                # Winner
                winner = int(np.argmax(specific))
                epoch_wins[winner] += 1

                # Hebbian update: winner moves toward input
                self.prototypes[winner] += epoch_lr * (x - self.prototypes[winner])

            # Homeostatic threshold adaptation
            epoch_rate = epoch_wins / N
            self.thresholds += 0.1 * (epoch_rate - self.target_rate)
            np.clip(self.thresholds, -0.3, 0.5, out=self.thresholds)

            if verbose:
                n_active = int((epoch_wins > 0).sum())
                print(f"  Epoch {epoch+1}/{n_epochs}: "
                      f"{n_active}/{n_proto} active prototypes")

        # Compute anomaly threshold from training distances
        distances = self._distances(normed)
        # Use mean + k*std instead of raw percentile
        # This is more robust to outliers in training data
        d_mean = float(np.mean(distances))
        d_std = float(np.std(distances))
        # Also compute percentile-based threshold
        d_pct = float(np.percentile(distances, self.threshold_pct))
        # Take the more conservative (higher) of the two
        self.threshold = max(d_pct, d_mean + 2.5 * d_std)
        self._train_dist_mean = d_mean
        self._train_dist_std = d_std
        self._fitted = True

        if verbose:
            print(f"  Anomaly threshold: {self.threshold:.4f} "
                  f"(mean={d_mean:.3f}, std={d_std:.3f})")

        return self

    def _normalize(self, data):
        """Normalize using training statistics."""
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return (data - self.running_mean) / self.running_std

    def _distances(self, normed_data):
        """
        Compute anomaly score using k-nearest prototype averaging.
        More robust than single nearest — reduces noise from
        individual prototype positions.
        """
        k = min(3, len(self.prototypes))
        distances = np.zeros(len(normed_data), dtype=np.float32)
        for i in range(len(normed_data)):
            x = normed_data[i]
            dists = np.linalg.norm(self.prototypes - x[np.newaxis, :], axis=1)
            # Average of k nearest distances (more robust)
            top_k = np.sort(dists)[:k]
            distances[i] = float(np.mean(top_k))
        return distances

    def score(self, data):
        """
        Compute anomaly scores for data.

        Higher score = more anomalous. Score of 0 = perfect match
        to a prototype. Score of 1 = completely unlike anything learned.

        Parameters
        ----------
        data : ndarray, shape (N, dim) or (dim,)

        Returns
        -------
        scores : ndarray, shape (N,)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")
        normed = self._normalize(data)
        return self._distances(normed)

    def detect(self, data):
        """
        Detect anomalies. Returns boolean array.

        Parameters
        ----------
        data : ndarray, shape (N, dim) or (dim,)

        Returns
        -------
        is_anomaly : ndarray of bool, shape (N,)
        """
        scores = self.score(data)
        return scores > self.threshold

    def update(self, sample, lr=0.01):
        """
        Online update: adapt to a single new sample.

        Only updates if the sample is NOT anomalous (don't learn
        from anomalies — that would corrupt the normal model).

        Parameters
        ----------
        sample : ndarray, shape (dim,)
        lr : float
            Online learning rate (should be small).

        Returns
        -------
        dict with 'is_anomaly', 'score', 'updated'.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        normed = self._normalize(sample)[0]

        # Score (k-nearest average distance)
        dists = np.linalg.norm(self.prototypes - normed[np.newaxis, :], axis=1)
        k = min(3, len(self.prototypes))
        distance = float(np.mean(np.sort(dists)[:k]))
        
        sims = -dists
        mean_sim = sims.mean()
        specific = sims - mean_sim - self.thresholds
        is_anomaly = distance > self.threshold

        self.n_seen += 1
        if is_anomaly:
            self.n_anomalies += 1

        # Only learn from normal samples
        updated = False
        if self.adapt and not is_anomaly:
            winner = int(np.argmax(specific))
            self.prototypes[winner] += lr * (normed - self.prototypes[winner])
            updated = True

        return {
            'is_anomaly': bool(is_anomaly),
            'score': float(distance),
            'updated': updated,
        }

    def f1_score(self, data, labels):
        """
        Compute F1 score given ground truth labels.

        Parameters
        ----------
        data : ndarray, shape (N, dim)
        labels : ndarray, shape (N,)
            1 = anomaly, 0 = normal.

        Returns
        -------
        dict with 'f1', 'precision', 'recall', 'accuracy'.
        """
        predictions = self.detect(data)
        labels = np.asarray(labels, dtype=bool)

        tp = int((predictions & labels).sum())
        fp = int((predictions & ~labels).sum())
        fn = int((~predictions & labels).sum())
        tn = int((~predictions & ~labels).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

        return {
            'f1': f1, 'precision': precision,
            'recall': recall, 'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }

    @property
    def anomaly_rate(self):
        """Fraction of inputs flagged as anomalous."""
        return self.n_anomalies / self.n_seen if self.n_seen > 0 else 0.0

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return (f"AnomalyDetector(prototypes={self.n_proto}, "
                f"status={status}, "
                f"anomaly_rate={self.anomaly_rate:.1%})")
