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

    def __init__(self, n_prototypes=128, threshold_percentile=95.0,
                 adapt=True, smooth_window=5, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_proto = n_prototypes
        self.threshold_pct = threshold_percentile
        self.adapt = adapt
        self._smooth_window = smooth_window

        self.prototypes = None
        self.proto_radius = None    # per-prototype local radius
        self.thresholds = None    # specificity thresholds
        self.target_rate = None
        self.threshold = None     # anomaly distance threshold
        self._adaptive_threshold = None  # adapts during streaming
        self.dim = None

        # Statistics
        self.n_seen = 0
        self.n_anomalies = 0
        self.running_mean = None
        self.running_std = None
        self._fitted = False

        # Temporal state (for streaming)
        from collections import deque
        self._score_buffer = deque(maxlen=100)
        self._conf_buffer = deque(maxlen=100)
        self._anomaly_streak = 0

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

        # ── Learn per-prototype local radius ──
        # Each prototype tracks how far its normal points typically are.
        # Points beyond the local radius are anomalous for THAT region.
        # This handles noisy data where some areas are naturally wider.
        self.proto_radius = np.ones(n_proto, dtype=np.float32)
        self.proto_assigned = np.zeros(n_proto, dtype=np.float32)

        for i in range(N):
            x = normed[i]
            dists = np.linalg.norm(self.prototypes - x[np.newaxis, :], axis=1)
            nearest = int(np.argmin(dists))
            self.proto_assigned[nearest] += 1

        # Compute radius per prototype: mean + 2*std of assigned distances
        for p in range(n_proto):
            assigned_dists = []
            for i in range(N):
                x = normed[i]
                dists = np.linalg.norm(self.prototypes - x[np.newaxis, :], axis=1)
                if np.argmin(dists) == p:
                    assigned_dists.append(dists[p])
            if len(assigned_dists) > 2:
                ad = np.array(assigned_dists)
                self.proto_radius[p] = float(np.mean(ad) + 2.0 * np.std(ad))
            else:
                self.proto_radius[p] = 999.0  # unused prototype, don't flag

        # Compute anomaly threshold from training distances
        distances = self._distances(normed)
        d_mean = float(np.mean(distances))
        d_std = float(np.std(distances))
        d_pct = float(np.percentile(distances, self.threshold_pct))
        self.threshold = max(d_pct, d_mean + 2.5 * d_std)
        self._train_dist_mean = d_mean
        self._train_dist_std = d_std
        self._fitted = True
        self._adaptive_threshold = self.threshold  # starts at learned threshold

        if verbose:
            n_active_protos = int((self.proto_assigned > 0).sum())
            print(f"  Anomaly threshold: {self.threshold:.4f} "
                  f"(mean={d_mean:.3f}, std={d_std:.3f}, "
                  f"{n_active_protos} active prototypes)")

        return self

    def _normalize(self, data):
        """Normalize using training statistics."""
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return (data - self.running_mean) / self.running_std

    def _distances(self, normed_data):
        """
        Compute anomaly score using LOCAL RADIUS normalization.

        For each point:
          1. Find nearest prototype
          2. Divide distance by that prototype's local radius
          3. Average with k-nearest for robustness

        This means: a point that's "far" from a tight prototype
        scores higher than the same distance from a wide prototype.
        Handles noisy data where some regions are naturally spread out.
        """
        k = min(3, len(self.prototypes))
        distances = np.zeros(len(normed_data), dtype=np.float32)
        for i in range(len(normed_data)):
            x = normed_data[i]
            raw_dists = np.linalg.norm(self.prototypes - x[np.newaxis, :], axis=1)

            # Normalize each distance by that prototype's local radius
            normalized_dists = raw_dists / (self.proto_radius + 1e-8)

            # Average of k nearest normalized distances
            top_k = np.sort(normalized_dists)[:k]
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
        Detect anomalies (batch mode). Returns boolean array.

        For streaming with temporal smoothing, use .process() instead.
        """
        scores = self.score(data)
        return scores > self.threshold

    def confidence(self, data):
        """
        LAYER 2: Confidence estimation.

        Maps raw scores to [0, 1] confidence that something is anomalous.
        Uses the training score distribution:
          confidence = sigmoid((score - threshold) / scale)

        Near threshold → ~0.5 (uncertain)
        Far above threshold → ~1.0 (definitely anomalous)
        Far below threshold → ~0.0 (definitely normal)

        Parameters
        ----------
        data : ndarray, shape (N, dim) or (dim,)

        Returns
        -------
        confidence : ndarray, shape (N,) — values in [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")
        scores = self.score(data)
        # Scale based on training score distribution
        scale = self._train_dist_std * 1.5 + 1e-8
        z = (scores - self.threshold) / scale
        return 1.0 / (1.0 + np.exp(-z * 3.0))  # steep sigmoid

    def process(self, sample, lr=0.01):
        """
        FULL THREE-LAYER PIPELINE for streaming.

        Layer 1 (Perception): Raw anomaly score from prototypes
        Layer 2 (Stability): Confidence + temporal smoothing
        Layer 3 (Decision): Adaptive threshold, final verdict

        This is the method to use for real-time streams.
        Considers recent history before flagging anomalies.

        Parameters
        ----------
        sample : ndarray, shape (dim,)
        lr : float
            Online learning rate.

        Returns
        -------
        dict with:
            'score': float — raw anomaly score
            'confidence': float — how sure we are (0-1)
            'is_anomaly': bool — final decision
            'streak': int — consecutive anomalous samples
            'adapted': bool — whether prototypes were updated
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        normed = self._normalize(sample)[0]

        # ── LAYER 1: PERCEPTION (raw score) ──
        raw_dists = np.linalg.norm(self.prototypes - normed[np.newaxis, :], axis=1)
        normalized_dists = raw_dists / (self.proto_radius + 1e-8)
        k = min(3, len(self.prototypes))
        score = float(np.mean(np.sort(normalized_dists)[:k]))

        # ── LAYER 2: STABILITY (confidence + temporal smoothing) ──
        scale = self._train_dist_std * 1.5 + 1e-8
        z = (score - self._adaptive_threshold) / scale
        conf = 1.0 / (1.0 + np.exp(-z * 3.0))

        # Temporal buffer
        self._score_buffer.append(score)
        self._conf_buffer.append(conf)

        # Temporal smoothing: average confidence over recent window
        if len(self._conf_buffer) >= 3:
            recent_conf = list(self._conf_buffer)[-self._smooth_window:]
            smoothed_conf = float(np.mean(recent_conf))
            # Count high-confidence anomalies in window
            high_conf_count = sum(1 for c in recent_conf if c > 0.6)
        else:
            smoothed_conf = conf
            high_conf_count = 1 if conf > 0.6 else 0

        # ── LAYER 3: DECISION (adaptive threshold + drift detection) ──

        # Decision: anomaly if smoothed confidence > 0.5 AND
        # at least 2 of last smooth_window samples are suspicious
        is_anomaly = (smoothed_conf > 0.5 and
                      high_conf_count >= min(2, len(self._conf_buffer)))

        # Track streaks (consecutive anomalies)
        if is_anomaly:
            self._anomaly_streak += 1
        else:
            self._anomaly_streak = 0

        # ── DRIFT DETECTION ──
        # Key insight: if EVERYTHING is anomalous for a long time,
        # it's not anomalies — it's a new normal.
        # Long streak → distribution shift → aggressive adaptation
        drift_detected = self._anomaly_streak > 15

        if drift_detected:
            # This is a distribution shift, not sustained anomalies
            is_anomaly = False  # stop flagging as anomaly
            self._anomaly_streak = 0

            # Aggressively update threshold to new distribution
            recent = list(self._score_buffer)[-30:]
            new_mean = float(np.mean(recent))
            new_std = float(np.std(recent))
            self._adaptive_threshold = new_mean + 2.5 * new_std
            self._train_dist_std = new_std
            self._train_dist_mean = new_mean

        # Normal adaptive threshold: gentle adjustment
        elif len(self._score_buffer) >= 20:
            recent_scores = list(self._score_buffer)[-50:]
            recent_std = float(np.std(recent_scores))
            recent_mean = float(np.mean(recent_scores))
            target = recent_mean + 2.5 * recent_std
            # Adapt faster when far from target (0.05), slower when close
            gap = abs(target - self._adaptive_threshold)
            rate = min(0.05, 0.005 + gap * 0.1)
            self._adaptive_threshold += rate * (target - self._adaptive_threshold)

        # Update statistics
        self.n_seen += 1
        if is_anomaly:
            self.n_anomalies += 1

        # Online adaptation
        adapted = False
        sims = -raw_dists
        mean_sim = sims.mean()
        specific = sims - mean_sim - self.thresholds
        winner = int(np.argmax(specific))

        if self.adapt:
            if drift_detected:
                # During drift: learn aggressively from everything
                self.prototypes[winner] += 0.1 * (normed - self.prototypes[winner])
                # Also update local radius
                d = float(raw_dists[winner])
                self.proto_radius[winner] = 0.9 * self.proto_radius[winner] + 0.1 * d * 1.5
                adapted = True
            elif not is_anomaly and conf < 0.3:
                # Normal: gentle learning from confident normals
                self.prototypes[winner] += lr * (normed - self.prototypes[winner])
                adapted = True

        return {
            'score': score,
            'confidence': float(conf),
            'smoothed_confidence': smoothed_conf,
            'is_anomaly': is_anomaly,
            'streak': self._anomaly_streak,
            'adapted': adapted,
        }

    def update(self, sample, lr=0.01):
        """
        Simple online update (backward compatible).
        For full pipeline, use .process() instead.
        """
        return self.process(sample, lr=lr)

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
