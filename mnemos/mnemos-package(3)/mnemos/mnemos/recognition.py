"""
Multi-prototype competitive recognition with specificity penalty.

Each concept has multiple prototypes in feature space to capture
variation (e.g., different handwriting styles for the same digit).
Recognition compares input against ALL prototypes of ALL concepts
and picks the best match with specificity penalty.

Author: Gustav Gausepohl
"""

import numpy as np


class PrototypeBridge:
    """
    Multi-prototype competitive recognition.

    Parameters
    ----------
    n_proto : int
        Number of prototypes per concept. More = captures more
        variation but uses more memory.
    seed : int
        Random seed.

    Example
    -------
    >>> bridge = PrototypeBridge(n_proto=3)
    >>> bridge.train(features, labels)
    >>> predictions = bridge.predict(test_features)
    >>> accuracy = (predictions == test_labels).mean()
    """

    def __init__(self, n_proto=3, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_proto = n_proto

        self.prototypes = {}       # name → list of prototype vectors
        self.proto_counts = {}     # name → list of counts
        self.thresholds = {}       # per-concept homeostatic threshold
        self.win_counts = {}
        self.total_recognitions = 0

        self._trained = False
        self._labels = []

    def train(self, features, labels, concept_names=None):
        """
        Train multi-prototypes from labeled feature vectors.

        For each concept:
          1. Collect all features with that label
          2. Initialize n_proto prototypes from random examples
          3. Competitive assignment (5 iterations):
             - Each example assigned to nearest prototype
             - Prototypes update to mean of assigned examples
          4. Result: n_proto cluster centres per concept

        Parameters
        ----------
        features : ndarray, shape (N, feat_dim)
            Feature vectors (e.g., from HebbianFilters.extract).
        labels : ndarray, shape (N,)
            Integer labels for each feature vector.
        concept_names : dict or None
            Maps label integers to concept name strings.
            If None, uses str(label).

        Returns
        -------
        dict mapping concept names to number of prototypes.
        """
        unique_labels = sorted(set(labels))
        result = {}

        for label in unique_labels:
            name = concept_names[label] if concept_names else str(label)
            mask = labels == label
            digit_feats = features[mask].copy()
            n_examples = len(digit_feats)

            if n_examples == 0:
                continue

            # Normalize features
            norms = np.linalg.norm(digit_feats, axis=1, keepdims=True) + 1e-8
            digit_feats /= norms

            # Initialize prototypes from random examples
            n_proto = min(self.n_proto, n_examples)
            init_idx = self.rng.choice(n_examples, size=n_proto, replace=False)
            protos = [digit_feats[i].copy() for i in init_idx]
            counts = [1.0] * n_proto

            # Competitive clustering (5 iterations)
            for _ in range(5):
                assignments = [[] for _ in range(n_proto)]
                for i in range(n_examples):
                    sims = [float(digit_feats[i] @ p) for p in protos]
                    best = int(np.argmax(sims))
                    assignments[best].append(i)

                for k in range(n_proto):
                    if assignments[k]:
                        protos[k] = digit_feats[assignments[k]].mean(axis=0)
                        norm = np.linalg.norm(protos[k])
                        if norm > 0:
                            protos[k] /= norm
                        counts[k] = float(len(assignments[k]))

            self.prototypes[name] = protos
            self.proto_counts[name] = counts
            self.thresholds[name] = 0.0
            self.win_counts[name] = 0.0
            result[name] = n_proto

            if name not in self._labels:
                self._labels.append(name)

        self._trained = True
        return result

    def recognize(self, feature_vec, top_k=3):
        """
        Recognize with specificity penalty.

        1. For each concept, compute MAX similarity across prototypes
        2. SPECIFICITY: subtract mean similarity across concepts
        3. Subtract homeostatic threshold
        4. Winner = highest specific similarity

        Parameters
        ----------
        feature_vec : ndarray, shape (feat_dim,)
            Feature vector to recognise.
        top_k : int
            Number of top results to return.

        Returns
        -------
        list of (concept_name, confidence) tuples.
        """
        if not self._trained:
            raise RuntimeError("Call .train() before .recognize()")

        feat = feature_vec.astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            return []
        feat /= norm

        # Best similarity per concept (max across prototypes)
        raw_sims = {}
        for name, protos in self.prototypes.items():
            raw_sims[name] = max(float(feat @ p) for p in protos)

        if not raw_sims:
            return []

        # ═══ SPECIFICITY PENALTY ═══
        mean_sim = np.mean(list(raw_sims.values()))
        specific = {}
        for name, sim in raw_sims.items():
            specific[name] = sim - mean_sim - self.thresholds.get(name, 0.0)

        # Homeostatic threshold update
        self.total_recognitions += 1
        if specific:
            winner = max(specific, key=specific.get)
            self.win_counts[winner] = self.win_counts.get(winner, 0) + 1
            target = 1.0 / max(len(self.prototypes), 1)
            for name in self.prototypes:
                rate = self.win_counts.get(name, 0) / max(self.total_recognitions, 1)
                self.thresholds[name] += 0.003 * (rate - target)

        results = [(n, max(s, 0.001)) for n, s in specific.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def predict(self, features):
        """
        Predict labels for a batch of feature vectors.

        Parameters
        ----------
        features : ndarray, shape (N, feat_dim)

        Returns
        -------
        predictions : list of str
        """
        return [self.recognize(f, top_k=1)[0][0]
                if self.recognize(f, top_k=1) else "?"
                for f in features]

    def accuracy(self, features, labels, concept_names=None):
        """
        Compute recognition accuracy.

        Parameters
        ----------
        features : ndarray, shape (N, feat_dim)
        labels : ndarray, shape (N,)
        concept_names : dict or None

        Returns
        -------
        float : accuracy in [0, 1].
        """
        correct = 0
        for i in range(len(features)):
            rec = self.recognize(features[i], top_k=1)
            true_name = concept_names[int(labels[i])] if concept_names else str(int(labels[i]))
            if rec and rec[0][0] == true_name:
                correct += 1
        return correct / len(features) if len(features) > 0 else 0

    @property
    def n_concepts(self):
        return len(self.prototypes)

    def __repr__(self):
        return (f"PrototypeBridge(n_proto={self.n_proto}, "
                f"concepts={self.n_concepts})")
