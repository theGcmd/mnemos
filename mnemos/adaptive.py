"""
Hebbian Adaptive Head — plug into any pre-trained model.

v6: Two scaling fixes from Opus architecture review.
    1. Margin scales with sqrt(n_classes) — reduces noise at 100+ classes
    2. Top-k weighted confusers — each update fights top-5 threats, not just 1

All updates: dw = f(pre, post). No gradients. No loss functions.

Author: Gustav Gausepohl
License: Mnemos Dual License (free research, commercial requires license)
"""

import numpy as np
import time


class AdaptiveHead:
    def __init__(self, n_proto=5, adapt_lr=0.01,
                 contrastive_epochs=10, contrastive_lr=0.01,
                 contrastive_margin=0.7, anti_lr_scale=1.0,
                 confuser_k=1, confuser_beta=3.0,
                 add_refine_epochs=10, add_margin=0.05,
                 add_push_strength=0.5, add_gentle_strength=0.1,
                 max_drift=0.1, n_anchors=5, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_proto = n_proto
        self.adapt_lr = adapt_lr
        self.contrastive_epochs = contrastive_epochs
        self.contrastive_lr = contrastive_lr
        self.contrastive_margin = contrastive_margin
        self.anti_lr_scale = anti_lr_scale
        self.confuser_k = confuser_k        # top-k confusers per update
        self.confuser_beta = confuser_beta  # softmax sharpness
        self.add_refine_epochs = add_refine_epochs
        self.add_margin = add_margin
        self.add_push_strength = add_push_strength
        self.add_gentle_strength = add_gentle_strength
        self.max_drift = max_drift
        self.n_anchors = n_anchors

        self.prototypes = {}
        self.proto_counts = {}
        self.proto_radius = {}
        self.thresholds = {}
        self.win_counts = {}
        self.total_predictions = 0
        self.classes = []
        self.feat_dim = None
        self._fitted = False
        self._ops_count = 0
        self._fit_time = 0
        self._predict_time = 0
        self._n_adapted = 0
        self._all_protos = None
        self._proto_labels = None
        self._anchors = {}

    # ── Spherical geometry ──

    def _spherical_pull(self, n, x, eta):
        cos = float(n @ x)
        tangent = x - cos * n
        n_new = n + eta * tangent
        nrm = np.linalg.norm(n_new)
        return n_new / nrm if nrm > 1e-8 else n

    def _spherical_push(self, n, o, eta):
        cos = float(n @ o)
        if cos < -0.99:
            return n
        tangent = n - cos * o
        if np.linalg.norm(tangent) < 1e-8:
            return n
        n_new = n + eta * tangent
        nrm = np.linalg.norm(n_new)
        return n_new / nrm if nrm > 1e-8 else n

    # ── Internal helpers ──

    def _build_flat(self):
        protos, labels = [], []
        for i, name in enumerate(self.classes):
            for p in self.prototypes[name]:
                protos.append(p)
                labels.append(i)
        self._all_protos = np.array(protos, dtype=np.float32)
        self._proto_labels = np.array(labels, dtype=np.int32)

    def _sync_from_flat(self):
        idx = 0
        for name in self.classes:
            for k in range(len(self.prototypes[name])):
                self.prototypes[name][k] = self._all_protos[idx].copy()
                idx += 1

    def _cluster(self, features, n_proto):
        n = len(features)
        n_proto = min(n_proto, n)
        init_idx = self.rng.choice(n, size=n_proto, replace=False)
        protos = [features[i].copy() for i in init_idx]
        assignments = [[] for _ in range(n_proto)]
        for _ in range(5):
            assignments = [[] for _ in range(n_proto)]
            P = np.array(protos)
            sims = features @ P.T
            best = np.argmax(sims, axis=1)
            for i, k in enumerate(best):
                assignments[k].append(i)
            for k in range(n_proto):
                if assignments[k]:
                    protos[k] = features[assignments[k]].mean(axis=0)
                    nrm = np.linalg.norm(protos[k])
                    if nrm > 0:
                        protos[k] /= nrm
        return protos, assignments

    def _local_radius(self, features, protos, assignments):
        radii = []
        for k in range(len(protos)):
            if assignments[k]:
                dists = np.linalg.norm(features[assignments[k]] - protos[k], axis=1)
                radii.append(float(dists.mean() + 2.0 * dists.std()))
            else:
                radii.append(1.0)
        return radii

    def _build_anchors(self, features, labels):
        for label in sorted(set(labels)):
            name = str(label)
            mask = labels == label
            class_feats = features[mask]
            n = min(self.n_anchors, len(class_feats))
            anchors, _ = self._cluster(class_feats, n)
            self._anchors[name] = np.array(anchors, dtype=np.float32)

    def _all_anchors_flat(self):
        if not self._anchors:
            return np.zeros((0, self.feat_dim or 512), dtype=np.float32), []
        vecs, names = [], []
        for name, ancs in self._anchors.items():
            for a in ancs:
                vecs.append(a)
                names.append(name)
        return np.array(vecs, dtype=np.float32), names

    def _adaptive_margin(self):
        """Margin scales down with class count to reduce noise at 100+ classes."""
        n = max(len(self.classes), 1)
        return self.contrastive_margin / np.sqrt(n / 10.0)

    def _lvq_refine(self, features, labels, verbose):
        self._build_flat()
        P = self._all_protos
        L = self._proto_labels
        eta = self.contrastive_lr
        alpha = self.anti_lr_scale
        k = min(self.confuser_k, len(self.classes) - 1)
        beta = self.confuser_beta
        margin = self._adaptive_margin()
        N = len(features)
        label_to_idx = {name: i for i, name in enumerate(self.classes)}

        if verbose:
            print(f"  Adaptive margin: {margin:.3f} (n_classes={len(self.classes)})")

        for epoch in range(self.contrastive_epochs):
            perm = self.rng.permutation(N)
            n_updates = 0
            for i in perm:
                x = features[i]
                y = label_to_idx[str(labels[i])]
                sims = P @ x
                correct_mask = (L == y)
                wrong_mask   = ~correct_mask

                if not correct_mask.any() or not wrong_mask.any():
                    continue

                winner_idx = int(np.argmax(np.where(correct_mask, sims, -np.inf)))
                d_w = sims[winner_idx]

                # Top-k confusers, softmax-weighted
                wrong_sims = sims.copy()
                wrong_sims[correct_mask] = -np.inf
                n_wrong = int(wrong_mask.sum())
                actual_k = min(k, n_wrong)
                top_k_idx = np.argpartition(wrong_sims, -actual_k)[-actual_k:]
                top_k_sims = wrong_sims[top_k_idx]

                # Only update if any confuser is within the margin window
                best_confuser_sim = top_k_sims.max()
                ratio = min(d_w / best_confuser_sim, best_confuser_sim / d_w)                         if d_w > 0 and best_confuser_sim > 0 else 0.0

                if ratio > margin:
                    # Pull winner toward x (spherical)
                    P[winner_idx] = self._spherical_pull(P[winner_idx], x, eta)

                    # Softmax weights over top-k confusers
                    shifted = top_k_sims - top_k_sims.max()
                    weights = np.exp(beta * shifted)
                    weights /= weights.sum()

                    # Push each confuser, weighted (spherical)
                    for j, w in zip(top_k_idx, weights):
                        P[j] = self._spherical_push(P[j], x, eta * alpha * w)

                    n_updates += 1

            if verbose:
                sample = self.rng.choice(N, size=min(200, N), replace=False)
                correct = sum(
                    self.classes[L[int(np.argmax(P @ features[i]))]] == str(labels[i])
                    for i in sample)
                print(f"  LVQ epoch {epoch+1}/{self.contrastive_epochs}: "
                      f"{n_updates} updates, ~{correct/len(sample):.1%} train acc")

        self._sync_from_flat()

    # ── Public API ──

    def fit(self, features, labels, verbose=True):
        t0 = time.time()
        features = np.asarray(features, dtype=np.float32)
        labels   = np.asarray(labels)
        N, D = features.shape
        self.feat_dim = D
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features_normed = features / norms

        for label in sorted(set(labels)):
            name = str(label)
            mask = labels == label
            class_feats = features_normed[mask]
            if len(class_feats) == 0:
                continue
            protos, assignments = self._cluster(class_feats, self.n_proto)
            radii = self._local_radius(class_feats, protos, assignments)
            self.prototypes[name]   = protos
            self.proto_counts[name] = [float(len(a)) for a in assignments]
            self.proto_radius[name] = radii
            self.thresholds[name]   = 0.0
            self.win_counts[name]   = 0.0
            if name not in self.classes:
                self.classes.append(name)

        if verbose:
            print(f"  Initial clustering: {len(self.classes)} classes, "
                  f"{sum(len(p) for p in self.prototypes.values())} prototypes")

        if self.contrastive_epochs > 0:
            if verbose:
                print(f"  Running LVQ (top-{min(self.confuser_k, len(self.classes)-1)}, "
                      f"{self.contrastive_epochs} epochs)...")
            self._lvq_refine(features_normed, labels, verbose)

        self._build_anchors(features_normed, labels)
        self._fitted = True
        self._fit_time = time.time() - t0
        if verbose:
            print(f"  Fit time: {self._fit_time:.2f}s")

        return {"n_classes": len(self.classes),
                "n_prototypes": sum(len(p) for p in self.prototypes.values()),
                "fit_time": self._fit_time}

    def predict(self, features, top_k=1, update_thresholds=True):
        if not self._fitted:
            raise RuntimeError("Call .fit() first")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms
        if self._all_protos is not None and top_k == 1:
            sims = features @ self._all_protos.T
            best = np.argmax(sims, axis=1)
            return np.array([self.classes[self._proto_labels[i]] for i in best])
        predictions = []
        for x in features:
            class_sims = {name: max(float(x @ p) for p in protos)
                          for name, protos in self.prototypes.items()}
            if top_k == 1:
                predictions.append(max(class_sims, key=class_sims.get))
            else:
                predictions.append(sorted(class_sims.items(),
                                          key=lambda kv: kv[1], reverse=True)[:top_k])
        if top_k == 1:
            return np.array(predictions)
        return predictions

    def accuracy(self, features, labels):
        preds = self.predict(features, update_thresholds=False)
        return float((preds == np.array([str(l) for l in labels])).mean())

    def add_class(self, name, features, verbose=True):
        t0 = time.time()
        name = str(name)
        features = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms
        N = len(features)

        new_protos_list, assignments = self._cluster(features, self.n_proto)
        new_radii = self._local_radius(features, new_protos_list, assignments)
        new_protos = np.array(new_protos_list, dtype=np.float32)

        old_protos_snap = self._all_protos.copy() if self._all_protos is not None                           else np.zeros((0, features.shape[1]), dtype=np.float32)

        anchor_vecs, anchor_labels = self._all_anchors_flat()
        has_anchors = len(anchor_vecs) > 0
        eta    = self.contrastive_lr
        push   = self.add_push_strength
        gentle = self.add_gentle_strength
        marg   = self.add_margin

        if len(old_protos_snap) > 0:
            for epoch in range(self.add_refine_epochs):
                perm = self.rng.permutation(N)
                for i in perm:
                    x = features[i]
                    new_sims = new_protos @ x
                    old_sims = old_protos_snap @ x
                    winner_new  = int(np.argmax(new_sims))
                    nearest_old = int(np.argmax(old_sims))
                    new_protos[winner_new] = self._spherical_pull(
                        new_protos[winner_new], x, eta)
                    margin = float(new_protos[winner_new] @ x) - old_sims[nearest_old]
                    if margin < marg:
                        new_protos[winner_new] = self._spherical_push(
                            new_protos[winner_new], old_protos_snap[nearest_old], eta * push)
                if has_anchors:
                    for a in anchor_vecs[self.rng.permutation(len(anchor_vecs))]:
                        new_sims = new_protos @ a
                        old_sims = old_protos_snap @ a
                        winner_new  = int(np.argmax(new_sims))
                        nearest_old = int(np.argmax(old_sims))
                        if float(new_protos[winner_new] @ a) - old_sims[nearest_old] < marg:
                            new_protos[winner_new] = self._spherical_push(
                                new_protos[winner_new], old_protos_snap[nearest_old], eta * push * 0.5)

        if self._all_protos is not None and len(self._all_protos) > 0:
            drift = np.zeros(len(self._all_protos), dtype=np.float32)
            P = self._all_protos
            label_to_proto_indices = {
                lbl: [j for j in range(len(self._proto_labels)) if self._proto_labels[j] == idx]
                for idx, lbl in enumerate(self.classes)
            }
            for epoch in range(self.add_refine_epochs):
                perm = self.rng.permutation(N)
                for i in perm:
                    x = features[i]
                    old_sims = P @ x
                    new_sims = new_protos @ x
                    nearest_old  = int(np.argmax(old_sims))
                    best_new_sim = float(np.max(new_sims))
                    if old_sims[nearest_old] > best_new_sim - marg:
                        cos = float(P[nearest_old] @ x)
                        tangent = P[nearest_old] - cos * x
                        nrm_t = np.linalg.norm(tangent)
                        if nrm_t > 1e-8:
                            delta = eta * gentle * tangent
                            delta_mag = float(np.linalg.norm(delta))
                            if drift[nearest_old] + delta_mag <= self.max_drift:
                                P[nearest_old] += delta
                                nrm = np.linalg.norm(P[nearest_old])
                                if nrm > 1e-8: P[nearest_old] /= nrm
                                drift[nearest_old] += delta_mag
                if has_anchors:
                    for i, a in enumerate(anchor_vecs):
                        lbl = anchor_labels[i]
                        if lbl not in label_to_proto_indices: continue
                        proto_idxs = label_to_proto_indices[lbl]
                        if not proto_idxs: continue
                        nearest_same = proto_idxs[int(np.argmax([float(P[j] @ a) for j in proto_idxs]))]
                        P[nearest_same] = self._spherical_pull(P[nearest_same], a, eta * gentle * 0.5)
            self._sync_from_flat()

        self.prototypes[name]   = [new_protos[k].copy() for k in range(len(new_protos))]
        self.proto_counts[name] = [float(len(a)) for a in assignments]
        self.proto_radius[name] = new_radii
        self.thresholds[name]   = float(np.mean(list(self.thresholds.values()))) if self.thresholds else 0.0
        self.win_counts[name]   = self.total_predictions / max(len(self.prototypes) + 1, 1)
        if name not in self.classes:
            self.classes.append(name)
        self._anchors[name] = np.array(self._cluster(features, min(self.n_anchors, N))[0], dtype=np.float32)
        self._build_flat()

        add_time = time.time() - t0
        if verbose:
            print(f"  Added class '{name}': {len(new_protos)} prototypes "
                  f"from {N} examples in {add_time:.3f}s")
        return {"time": add_time, "n_proto": len(new_protos)}

    def adapt(self, feature, label, lr=None):
        lr = lr or self.adapt_lr
        label = str(label)
        feature = np.asarray(feature, dtype=np.float32)
        norm = np.linalg.norm(feature)
        if norm > 0: feature = feature / norm
        if label not in self.prototypes:
            self.prototypes[label]   = [feature.copy()]
            self.proto_counts[label] = [1.0]
            self.proto_radius[label] = [1.0]
            self.thresholds[label]   = 0.0
            self.win_counts[label]   = 0.0
            if label not in self.classes:
                self.classes.append(label)
            self._n_adapted += 1
            self._build_flat()
            return
        protos = self.prototypes[label]
        sims   = [float(feature @ p) for p in protos]
        winner = int(np.argmax(sims))
        protos[winner] = self._spherical_pull(protos[winner], feature, lr)
        self.proto_counts[label][winner] += 1
        self._n_adapted += 1
        self._build_flat()

    def forget_class(self, name):
        name = str(name)
        if name in self.prototypes:
            del self.prototypes[name]
            del self.proto_counts[name]
            del self.proto_radius[name]
            self.thresholds.pop(name, None)
            self.win_counts.pop(name, None)
            self._anchors.pop(name, None)
            if name in self.classes:
                self.classes.remove(name)
            self._build_flat()

    def compute_savings(self, backbone_params=None, n_samples=None):
        n_samples    = n_samples or max(self._n_adapted, 1)
        total_protos = sum(len(p) for p in self.prototypes.values())
        feat_dim     = self.feat_dim or 512
        if backbone_params is None:
            backbone_params = feat_dim * 100
        backprop_total  = 4 * backbone_params * n_samples
        hebbian_total   = (total_protos * feat_dim + feat_dim) * n_samples
        backprop_memory = 3 * backbone_params * 4
        hebbian_memory  = total_protos * feat_dim * 4
        return {
            "backprop_ops":          int(backprop_total),
            "hebbian_ops":           int(hebbian_total),
            "compute_ratio":         float(backprop_total / (hebbian_total + 1)),
            "backprop_memory_bytes": int(backprop_memory),
            "hebbian_memory_bytes":  int(hebbian_memory),
            "memory_ratio":          float(backprop_memory / (hebbian_memory + 1)),
            "backbone_params":       backbone_params,
            "n_prototypes":          total_protos,
            "n_samples":             n_samples,
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
                f"contrastive_epochs={self.contrastive_epochs}, "
                f"max_drift={self.max_drift})")

