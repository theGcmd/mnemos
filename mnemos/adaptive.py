"""
Hebbian Adaptive Head — plug into any pre-trained model.

The pitch:
  "You trained your model with PyTorch. Now plug in Mnemos.
   Your model adapts to new data without retraining,
   without forgetting, using 100x less compute."

How it works:
  1. You have a pre-trained backbone (ResNet, CNN, whatever)
  2. Freeze it. Remove the classification head.
  3. Plug in AdaptiveHead instead.
  4. Feed features through → Hebbian prototypes learn instantly
  5. New classes? Just show examples. No retraining.
  6. Distribution drift? Prototypes adapt continuously.
  7. All without backprop. All with local Hebbian rules.

Why companies care:
  - Fine-tuning with backprop: store gradients for every parameter,
    compute backward pass through every layer, run optimizer.
    Cost: O(parameters × samples × epochs)
  - Hebbian adaptation: update a few prototypes with outer product.
    Cost: O(prototypes × feature_dim)
    That's 100-1000x less compute.

Author: Gustav Gausepohl
License: Mnemos Dual License (free research, commercial requires license)
"""

import numpy as np
import time
from collections import defaultdict


class AdaptiveHead:
    """
    Hebbian classification head that replaces backprop fine-tuning.

    Attach this to any frozen feature extractor. It learns to
    classify using competitive multi-prototype matching with
    specificity penalty — the same mechanism that beats
    IsolationForest on anomaly detection.

    Parameters
    ----------
    n_proto : int
        Prototypes per class. More = captures more variation.
        3-5 typical. Higher for complex distributions.
    adapt_lr : float
        Online adaptation learning rate. How fast prototypes
        move toward new examples. 0.01 = gentle, 0.1 = aggressive.
    seed : int
        Random seed.

    Example
    -------
    >>> # Assume backbone gives 512-dim features
    >>> head = mnemos.AdaptiveHead(n_proto=5)
    >>>
    >>> # Learn from labeled features (no backprop)
    >>> head.fit(train_features, train_labels)
    >>>
    >>> # Predict
    >>> predictions = head.predict(test_features)
    >>>
    >>> # Add a new class without touching the backbone
    >>> head.add_class("cat", cat_features)
    >>>
    >>> # Online adaptation (one sample at a time)
    >>> head.adapt(new_feature, label="cat")
    """

    def __init__(self, n_proto=5, adapt_lr=0.01, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_proto = n_proto
        self.adapt_lr = adapt_lr

        # Per-class prototypes: class_name → list of prototype vectors
        self.prototypes = {}
        self.proto_counts = {}

        # Specificity thresholds (homeostatic)
        self.thresholds = {}
        self.win_counts = {}
        self.total_predictions = 0

        # Per-prototype local radius (from AnomalyDetector)
        self.proto_radius = {}

        # Tracking
        self.classes = []
        self.feat_dim = None
        self._fitted = False

        # Compute tracking (for efficiency comparison)
        self._ops_count = 0
        self._fit_time = 0
        self._predict_time = 0
        self._n_adapted = 0

    def fit(self, features, labels, verbose=True):
        """
        Learn class prototypes from labeled features.

        No backprop. No gradient computation. No optimizer.
        Just competitive Hebbian clustering: prototypes move
        toward their assigned examples.

        Parameters
        ----------
        features : ndarray, shape (N, feat_dim)
            Feature vectors from a frozen backbone.
        labels : array-like, shape (N,)
            Class labels (int or str).

        Returns
        -------
        dict with training statistics.
        """
        t0 = time.time()
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels)
        N, D = features.shape
        self.feat_dim = D

        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features_normed = features / norms

        unique_labels = sorted(set(labels))

        for label in unique_labels:
            name = str(label)
            mask = labels == label
            class_feats = features_normed[mask]
            n_examples = len(class_feats)

            if n_examples == 0:
                continue

            n_proto = min(self.n_proto, n_examples)

            # Initialize prototypes from random examples
            init_idx = self.rng.choice(n_examples, size=n_proto, replace=False)
            protos = [class_feats[i].copy() for i in init_idx]
            counts = [1.0] * n_proto

            # Competitive Hebbian clustering (5 iterations)
            for iteration in range(5):
                assignments = [[] for _ in range(n_proto)]
                for i in range(n_examples):
                    sims = [float(class_feats[i] @ p) for p in protos]
                    best = int(np.argmax(sims))
                    assignments[best].append(i)

                for k in range(n_proto):
                    if assignments[k]:
                        protos[k] = class_feats[assignments[k]].mean(axis=0)
                        norm = np.linalg.norm(protos[k])
                        if norm > 0:
                            protos[k] /= norm
                        counts[k] = float(len(assignments[k]))

                self._ops_count += n_examples * n_proto  # similarity ops

            # Compute local radius per prototype
            radii = []
            for k in range(n_proto):
                if assignments[k]:
                    dists = [float(np.linalg.norm(class_feats[i] - protos[k]))
                             for i in assignments[k]]
                    radii.append(float(np.mean(dists) + 2.0 * np.std(dists)))
                else:
                    radii.append(1.0)

            self.prototypes[name] = protos
            self.proto_counts[name] = counts
            self.proto_radius[name] = radii
            self.thresholds[name] = 0.0
            self.win_counts[name] = 0.0

            if name not in self.classes:
                self.classes.append(name)

            if verbose:
                sizes = ", ".join(str(int(c)) for c in counts)
                print(f"  Class '{name}': {n_proto} prototypes "
                      f"from {n_examples} examples ({sizes})")

        self._fitted = True
        self._fit_time = time.time() - t0

        if verbose:
            print(f"  {len(self.classes)} classes, "
                  f"{sum(len(p) for p in self.prototypes.values())} total prototypes")
            print(f"  Fit time: {self._fit_time:.3f}s")
            print(f"  Operations: {self._ops_count:,}")

        return {
            'n_classes': len(self.classes),
            'n_prototypes': sum(len(p) for p in self.prototypes.values()),
            'fit_time': self._fit_time,
            'ops': self._ops_count,
        }

    def predict(self, features, top_k=1):
        """
        Predict class labels.

        Compares each input against ALL prototypes of ALL classes.
        Uses max similarity per class with specificity penalty.

        Parameters
        ----------
        features : ndarray, shape (N, feat_dim) or (feat_dim,)
        top_k : int
            Return top-k predictions per sample.

        Returns
        -------
        If top_k=1: ndarray of predicted labels, shape (N,)
        If top_k>1: list of [(label, confidence), ...] per sample
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        t0 = time.time()
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms

        N = len(features)
        predictions = []

        for i in range(N):
            x = features[i]

            # Best similarity per class (max across prototypes)
            class_sims = {}
            for name, protos in self.prototypes.items():
                best_sim = max(float(x @ p) for p in protos)
                class_sims[name] = best_sim
                self._ops_count += len(protos)

            # Specificity penalty
            if class_sims:
                mean_sim = np.mean(list(class_sims.values()))
                specific = {}
                for name, sim in class_sims.items():
                    specific[name] = sim - mean_sim - self.thresholds.get(name, 0.0)

                # Homeostatic threshold update
                self.total_predictions += 1
                winner = max(specific, key=specific.get)
                self.win_counts[winner] = self.win_counts.get(winner, 0) + 1
                target = 1.0 / max(len(self.prototypes), 1)
                for name in self.prototypes:
                    rate = self.win_counts.get(name, 0) / max(self.total_predictions, 1)
                    self.thresholds[name] += 0.002 * (rate - target)

                if top_k == 1:
                    predictions.append(winner)
                else:
                    sorted_preds = sorted(specific.items(),
                                          key=lambda x: x[1], reverse=True)
                    predictions.append(sorted_preds[:top_k])
            else:
                predictions.append("?" if top_k == 1 else [("?", 0.0)])

        self._predict_time += time.time() - t0

        if top_k == 1:
            return np.array(predictions)
        return predictions

    def accuracy(self, features, labels):
        """Compute classification accuracy."""
        preds = self.predict(features)
        labels_str = np.array([str(l) for l in labels])
        return float((preds == labels_str).mean())

    def add_class(self, name, features, verbose=True):
        """
        Add a new class WITHOUT retraining existing classes.

        This is the key advantage over backprop fine-tuning:
        backprop must retrain the entire head (and often the backbone)
        to add a class. Mnemos just adds new prototypes.

        Parameters
        ----------
        name : str
            New class name.
        features : ndarray, shape (N, feat_dim)
            Example features for the new class.
        """
        t0 = time.time()
        name = str(name)
        features = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms

        n = len(features)
        n_proto = min(self.n_proto, n)

        # Initialize from random examples
        init_idx = self.rng.choice(n, size=n_proto, replace=False)
        protos = [features[i].copy() for i in init_idx]
        counts = [1.0] * n_proto

        # Cluster
        for _ in range(5):
            assignments = [[] for _ in range(n_proto)]
            for i in range(n):
                sims = [float(features[i] @ p) for p in protos]
                assignments[np.argmax(sims)].append(i)
            for k in range(n_proto):
                if assignments[k]:
                    protos[k] = features[assignments[k]].mean(axis=0)
                    norm = np.linalg.norm(protos[k])
                    if norm > 0:
                        protos[k] /= norm
                    counts[k] = float(len(assignments[k]))

        # Local radius
        radii = []
        for k in range(n_proto):
            if assignments[k]:
                dists = [float(np.linalg.norm(features[i] - protos[k]))
                         for i in assignments[k]]
                radii.append(float(np.mean(dists) + 2.0 * np.std(dists)))
            else:
                radii.append(1.0)

        self.prototypes[name] = protos
        self.proto_counts[name] = counts
        self.proto_radius[name] = radii
        self.thresholds[name] = 0.0
        self.win_counts[name] = 0.0

        if name not in self.classes:
            self.classes.append(name)

        add_time = time.time() - t0
        if verbose:
            print(f"  Added class '{name}': {n_proto} prototypes "
                  f"from {n} examples in {add_time:.4f}s")

        return {'time': add_time, 'n_proto': n_proto}

    def adapt(self, feature, label, lr=None):
        """
        Online adaptation: update prototypes from a single example.

        This is continuous learning. The model improves with every
        new example, without storing gradients, without backward
        passes, without retraining.

        Parameters
        ----------
        feature : ndarray, shape (feat_dim,)
        label : str or int
            True class label.
        lr : float or None
            Learning rate. Uses self.adapt_lr if None.
        """
        lr = lr or self.adapt_lr
        label = str(label)
        feature = np.asarray(feature, dtype=np.float32)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm

        if label not in self.prototypes:
            # New class from single example
            self.prototypes[label] = [feature.copy()]
            self.proto_counts[label] = [1.0]
            self.proto_radius[label] = [1.0]
            self.thresholds[label] = 0.0
            self.win_counts[label] = 0.0
            if label not in self.classes:
                self.classes.append(label)
            self._n_adapted += 1
            return

        # Find nearest prototype of the correct class
        protos = self.prototypes[label]
        sims = [float(feature @ p) for p in protos]
        winner = int(np.argmax(sims))

        # Hebbian update: move toward example
        protos[winner] += lr * (feature - protos[winner])
        norm = np.linalg.norm(protos[winner])
        if norm > 0:
            protos[winner] /= norm

        self.proto_counts[label][winner] += 1
        self._n_adapted += 1
        self._ops_count += len(protos)

    def forget_class(self, name):
        """Remove a class entirely."""
        name = str(name)
        if name in self.prototypes:
            del self.prototypes[name]
            del self.proto_counts[name]
            del self.proto_radius[name]
            if name in self.thresholds:
                del self.thresholds[name]
            if name in self.win_counts:
                del self.win_counts[name]
            if name in self.classes:
                self.classes.remove(name)

    def compute_savings(self, backbone_params=None, n_samples=None):
        """
        Estimate compute savings vs backprop fine-tuning.

        Parameters
        ----------
        backbone_params : int or None
            Number of parameters in the backbone (e.g., 11M for ResNet-18).
            If None, estimates from feature dim.
        n_samples : int or None
            Number of adaptation samples. If None, uses actual count.

        Returns
        -------
        dict with estimated savings.
        """
        n_samples = n_samples or max(self._n_adapted, 1)
        total_protos = sum(len(p) for p in self.prototypes.values())
        feat_dim = self.feat_dim or 512

        if backbone_params is None:
            # Rough estimate: feat_dim * 100 (typical for small CNNs)
            backbone_params = feat_dim * 100

        # Backprop fine-tuning cost (per sample):
        #   Forward: backbone_params MACs
        #   Backward: ~2x backbone_params MACs (gradient + weight update)
        #   Optimizer: ~backbone_params (momentum, Adam state)
        #   Total: ~4x backbone_params per sample
        backprop_ops_per_sample = 4 * backbone_params
        backprop_total = backprop_ops_per_sample * n_samples

        # Hebbian adaptation cost (per sample):
        #   Compare to all prototypes: total_protos * feat_dim
        #   Update winner: feat_dim
        #   Total: total_protos * feat_dim + feat_dim
        hebbian_ops_per_sample = total_protos * feat_dim + feat_dim
        hebbian_total = hebbian_ops_per_sample * n_samples

        # Memory comparison
        # Backprop: gradients + optimizer state = ~3x parameters
        backprop_memory = 3 * backbone_params * 4  # bytes (float32)
        # Hebbian: just prototypes
        hebbian_memory = total_protos * feat_dim * 4  # bytes

        ratio = backprop_total / (hebbian_total + 1) if hebbian_total > 0 else float('inf')
        memory_ratio = backprop_memory / (hebbian_memory + 1) if hebbian_memory > 0 else float('inf')

        return {
            'backprop_ops': int(backprop_total),
            'hebbian_ops': int(hebbian_total),
            'compute_ratio': float(ratio),
            'backprop_memory_bytes': int(backprop_memory),
            'hebbian_memory_bytes': int(hebbian_memory),
            'memory_ratio': float(memory_ratio),
            'backbone_params': backbone_params,
            'n_prototypes': total_protos,
            'n_samples': n_samples,
        }

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def n_prototypes(self):
        return sum(len(p) for p in self.prototypes.values())

    def __repr__(self):
        return (f"AdaptiveHead(classes={self.n_classes}, "
                f"prototypes={self.n_prototypes}, "
                f"adapted={self._n_adapted})")
