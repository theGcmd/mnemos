"""
Real-time stream processing with Hebbian learning.

Watches a continuous data stream. Learns normal patterns.
Detects anomalies. Adapts when the distribution shifts.
Never stops learning. Never needs retraining.

Designed for edge deployment:
  - Drones (detect unusual sensor readings mid-flight)
  - IoT sensors (learn normal, flag anomalies, adapt to drift)
  - Industrial monitoring (predict machine failure)
  - Network security (detect intrusions in real-time)

The processor has three modes:
  LEARNING  — building initial model of normality
  WATCHING  — monitoring for anomalies
  ADAPTING  — detected distribution shift, rapid relearning

Author: Gustav Gausepohl
"""

import numpy as np
from collections import deque


class StreamProcessor:
    """
    Real-time Hebbian stream processor.

    Processes one sample at a time from a continuous stream.
    Learns what's normal, detects what's not, adapts to change.

    Parameters
    ----------
    dim : int
        Dimension of input vectors.
    n_prototypes : int
        Normal-pattern prototypes. More = finer model.
    window_size : int
        Recent history window for drift detection.
    warmup : int
        Samples to collect before switching from LEARNING to WATCHING.
    drift_sensitivity : float
        How sensitive to distribution shifts. Lower = more sensitive.
    seed : int
        Random seed.

    Example
    -------
    >>> stream = mnemos.StreamProcessor(dim=10, n_prototypes=16)
    >>> for sample in data_stream:
    ...     result = stream.process(sample)
    ...     if result['anomaly']:
    ...         alert(f"Anomaly at t={result['t']}: score={result['score']:.3f}")
    ...     if result['drift']:
    ...         log(f"Distribution shift detected at t={result['t']}")
    """

    LEARNING = 'learning'
    WATCHING = 'watching'
    ADAPTING = 'adapting'

    def __init__(self, dim, n_prototypes=16, window_size=100,
                 warmup=200, drift_sensitivity=0.3, seed=42):
        self.rng = np.random.RandomState(seed)
        self.dim = dim
        self.n_proto = n_prototypes
        self.window_size = window_size
        self.warmup = warmup
        self.drift_sensitivity = drift_sensitivity

        # Prototypes (normal pattern models)
        self.prototypes = self.rng.randn(n_prototypes, dim).astype(np.float32) * 0.01
        for i in range(n_prototypes):
            norm = np.linalg.norm(self.prototypes[i])
            if norm > 0:
                self.prototypes[i] /= norm

        # Specificity thresholds
        self.thresholds = np.zeros(n_prototypes, dtype=np.float32)
        self.target_rate = 1.0 / n_prototypes
        self.win_counts = np.zeros(n_prototypes, dtype=np.float32)

        # Running statistics for normalisation
        self.running_mean = np.zeros(dim, dtype=np.float32)
        self.running_var = np.ones(dim, dtype=np.float32)
        self.running_alpha = 0.01  # EMA smoothing

        # Anomaly threshold (learned from data)
        self.score_history = deque(maxlen=window_size)
        self.anomaly_threshold = 0.5  # initial, adapts
        self.anomaly_pct = 95.0       # percentile for threshold

        # Drift detection
        self.recent_scores = deque(maxlen=window_size)
        self.baseline_scores = deque(maxlen=window_size)
        self.drift_detected = False
        self.last_drift_t = -1

        # State
        self.state = self.LEARNING
        self.t = 0
        self.n_anomalies = 0
        self.n_drifts = 0

        # Event log
        self.events = deque(maxlen=1000)

    def _normalize(self, x):
        """Normalize using running statistics (zero mean, unit variance)."""
        return (x - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)

    def _update_stats(self, x):
        """Update running mean and variance (EMA)."""
        alpha = self.running_alpha
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        diff = x - self.running_mean
        self.running_var = (1 - alpha) * self.running_var + alpha * (diff ** 2)

    def _score(self, x_normed):
        """Compute anomaly score: min Euclidean distance to prototypes."""
        dists = np.linalg.norm(self.prototypes - x_normed[np.newaxis, :], axis=1)
        return float(np.min(dists))

    def _learn(self, x_normed, lr=0.05):
        """Competitive Hebbian update with specificity penalty."""
        dists = np.linalg.norm(self.prototypes - x_normed[np.newaxis, :], axis=1)
        sims = -dists  # closer = higher
        mean_sim = sims.mean()
        specific = sims - mean_sim - self.thresholds

        winner = int(np.argmax(specific))
        self.win_counts[winner] += 1

        # Hebbian update
        self.prototypes[winner] += lr * (x_normed - self.prototypes[winner])

        # Homeostatic threshold
        total = self.win_counts.sum()
        if total > 0:
            rates = self.win_counts / total
            self.thresholds += 0.01 * (rates - self.target_rate)
            np.clip(self.thresholds, -0.3, 0.5, out=self.thresholds)

    def _detect_drift(self):
        """Detect distribution shift by comparing recent vs baseline scores."""
        if len(self.recent_scores) < self.window_size // 4:
            return False
        if len(self.baseline_scores) < self.window_size // 4:
            return False

        recent_mean = np.mean(list(self.recent_scores))
        baseline_mean = np.mean(list(self.baseline_scores))
        baseline_std = np.std(list(self.baseline_scores)) + 1e-8

        # Z-score of recent mean relative to baseline
        z = (recent_mean - baseline_mean) / baseline_std
        return z > (2.0 * self.drift_sensitivity / 0.3)  # ~2 sigma default

    def process(self, sample):
        """
        Process one sample from the stream.

        The main loop. Call this for every new data point.

        Parameters
        ----------
        sample : ndarray, shape (dim,)
            One input vector from the stream.

        Returns
        -------
        dict with:
            't': int — time step
            'state': str — current mode (learning/watching/adapting)
            'score': float — anomaly score (0 = normal, 1 = very anomalous)
            'anomaly': bool — is this sample anomalous?
            'drift': bool — was a distribution shift detected this step?
        """
        self.t += 1
        x = np.asarray(sample, dtype=np.float32)

        # Update running statistics
        self._update_stats(x)

        # Normalize
        x_normed = self._normalize(x)

        # Score
        score = self._score(x_normed)
        self.score_history.append(score)
        self.recent_scores.append(score)

        # ── STATE MACHINE ──

        if self.state == self.LEARNING:
            # Still in warmup: learn aggressively
            self._learn(x_normed, lr=0.1)

            if self.t >= self.warmup:
                # Set anomaly threshold from warmup data
                scores_arr = np.array(list(self.score_history))
                self.anomaly_threshold = float(
                    np.percentile(scores_arr, self.anomaly_pct))
                # Save baseline
                self.baseline_scores = deque(self.score_history,
                                              maxlen=self.window_size)
                self.state = self.WATCHING
                self.events.append({
                    't': self.t, 'type': 'state_change',
                    'desc': f'LEARNING → WATCHING (threshold={self.anomaly_threshold:.4f})'
                })

            return {
                't': self.t, 'state': self.state,
                'score': score, 'anomaly': False, 'drift': False,
            }

        # WATCHING or ADAPTING
        is_anomaly = score > self.anomaly_threshold
        if is_anomaly:
            self.n_anomalies += 1

        # Drift detection
        drift = False
        if self.state == self.WATCHING:
            drift = self._detect_drift()
            if drift:
                self.state = self.ADAPTING
                self.n_drifts += 1
                self.last_drift_t = self.t
                self.events.append({
                    't': self.t, 'type': 'drift',
                    'desc': f'Distribution shift detected'
                })

        # Learning policy
        if self.state == self.WATCHING:
            # Gentle adaptation from normal samples only
            if not is_anomaly:
                self._learn(x_normed, lr=0.005)
        elif self.state == self.ADAPTING:
            # Aggressive relearning during adaptation
            self._learn(x_normed, lr=0.05)

            # Update threshold
            if len(self.score_history) > 20:
                scores_arr = np.array(list(self.score_history))
                self.anomaly_threshold = float(
                    np.percentile(scores_arr, self.anomaly_pct))

            # Return to watching after adaptation window
            if self.t - self.last_drift_t > self.window_size:
                self.state = self.WATCHING
                self.baseline_scores = deque(self.recent_scores,
                                              maxlen=self.window_size)
                self.events.append({
                    't': self.t, 'type': 'state_change',
                    'desc': 'ADAPTING → WATCHING'
                })

        if is_anomaly:
            self.events.append({
                't': self.t, 'type': 'anomaly',
                'desc': f'Anomaly (score={score:.4f})'
            })

        return {
            't': self.t, 'state': self.state,
            'score': score, 'anomaly': is_anomaly, 'drift': drift,
        }

    def process_batch(self, data, verbose=False):
        """
        Process a batch of samples (convenience wrapper).

        Parameters
        ----------
        data : ndarray, shape (N, dim)
        verbose : bool

        Returns
        -------
        list of result dicts.
        """
        results = []
        for i in range(len(data)):
            result = self.process(data[i])
            results.append(result)
            if verbose and (result['anomaly'] or result['drift']):
                if result['drift']:
                    print(f"  t={result['t']:6d}: ⚡ DRIFT DETECTED")
                elif result['anomaly']:
                    print(f"  t={result['t']:6d}: ⚠ anomaly "
                          f"(score={result['score']:.3f})")
        return results

    def summary(self):
        """Get processor status summary."""
        return {
            'state': self.state,
            't': self.t,
            'n_anomalies': self.n_anomalies,
            'n_drifts': self.n_drifts,
            'anomaly_rate': self.n_anomalies / max(self.t, 1),
            'anomaly_threshold': self.anomaly_threshold,
            'recent_events': list(self.events)[-5:],
        }

    def __repr__(self):
        return (f"StreamProcessor(state={self.state}, t={self.t}, "
                f"anomalies={self.n_anomalies}, drifts={self.n_drifts})")
