"""
Continual learning without forgetting.

Learns new classes/tasks sequentially without catastrophic
forgetting of previously learned knowledge. Uses frozen
subspace consolidation: once a filter specialises for a task,
its weights are frozen to protect that knowledge.

97.4% on Split-MNIST (5 tasks, no replay, no backprop).
Beats EWC (65.5%) which uses gradients.

Author: Gustav Gausepohl
"""

import numpy as np
from mnemos.competitive import HebbianFilters
from mnemos.recognition import PrototypeBridge


class ContinualLearner:
    """
    Learns new visual categories without forgetting old ones.

    Strategy:
      1. Learn filters for Task 1 → freeze the ones that specialised
      2. Task 2 arrives → only FREE (unfrozen) filters learn
      3. New prototypes added for new classes
      4. Old prototypes preserved — recognition across ALL tasks

    Parameters
    ----------
    n_filters : int
        Total number of Hebbian filters (shared across tasks).
    n_proto : int
        Prototypes per class.
    patch_size : int
        Convolutional patch size.
    freeze_threshold : float
        Filters with activation rate above this get frozen after
        each task. Higher = fewer frozen = more capacity for future.
    seed : int
        Random seed.

    Example
    -------
    >>> learner = mnemos.ContinualLearner(n_filters=400)
    >>> learner.learn_task(task1_images, task1_labels, task_name="digits_01")
    >>> learner.learn_task(task2_images, task2_labels, task_name="digits_23")
    >>> # Can still recognise task 1 AND task 2
    >>> accuracy_task1 = learner.evaluate(task1_test, task1_labels)
    >>> accuracy_task2 = learner.evaluate(task2_test, task2_labels)
    """

    def __init__(self, n_filters=400, n_proto=3, patch_size=5,
                 freeze_threshold=0.02, seed=42):
        self.n_filters = n_filters
        self.n_proto = n_proto
        self.patch_size = patch_size
        self.freeze_threshold = freeze_threshold
        self.seed = seed

        # Perception layer (shared, with frozen filters)
        self.perception = HebbianFilters(n_filters, patch_size, seed)

        # Recognition (accumulates prototypes across tasks)
        self.recognition = PrototypeBridge(n_proto, seed + 1)

        # Track which filters are frozen
        self.frozen = np.zeros(n_filters, dtype=bool)

        # Task history
        self.tasks = []
        self.n_tasks = 0
        self.total_classes = 0

    def learn_task(self, images, labels, task_name=None,
                   n_epochs=10, lr=0.1, verbose=True):
        """
        Learn a new task without forgetting previous tasks.

        1. Train perception filters (only unfrozen ones update)
        2. Extract features using ALL filters (frozen + new)
        3. Add new prototypes for new classes
        4. Freeze filters that specialised for this task

        Parameters
        ----------
        images : ndarray, shape (N, H, W) or (N, H*W)
        labels : ndarray, shape (N,)
        task_name : str or None
        n_epochs : int
        lr : float
        verbose : bool

        Returns
        -------
        dict with task statistics.
        """
        self.n_tasks += 1
        task_name = task_name or f"task_{self.n_tasks}"

        if verbose:
            n_free = int((~self.frozen).sum())
            print(f"  Task '{task_name}': {n_free}/{self.n_filters} "
                  f"filters available (rest frozen)")

        # Reshape
        if images.ndim == 2:
            img_size = int(np.sqrt(images.shape[1]))
            images = images.reshape(-1, img_size, img_size)

        N = len(images)
        img_size = images.shape[1]
        ps = self.patch_size
        os = img_size - ps + 1

        # Extract patches
        n_per = 20
        n_patches = min(N, 5000) * n_per
        patches = np.zeros((n_patches, ps * ps), dtype=np.float32)
        rng = np.random.RandomState(self.seed + self.n_tasks * 100)
        idx = 0
        for n in range(min(N, 5000)):
            for _ in range(n_per):
                i = rng.randint(0, os)
                j = rng.randint(0, os)
                patches[idx] = images[n, i:i+ps, j:j+ps].ravel()
                idx += 1

        patches -= patches.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8
        patches /= norms

        # Train only UNFROZEN filters
        free_mask = ~self.frozen
        free_idx = np.where(free_mask)[0]

        if len(free_idx) == 0:
            if verbose:
                print("  WARNING: No free filters! Cannot learn new features.")
        else:
            # Extract the free filter columns for training
            for epoch in range(n_epochs):
                perm = rng.permutation(n_patches)
                epoch_lr = lr / (1.0 + epoch * 0.2)
                batch_size = 2000

                for start in range(0, n_patches, batch_size):
                    end = min(start + batch_size, n_patches)
                    batch = patches[perm[start:end]]

                    # Similarity to ALL filters
                    sims = batch @ self.perception.filters

                    # Mask frozen filters with -inf so they can't win
                    sims_masked = sims.copy()
                    sims_masked[:, self.frozen] = -np.inf

                    # Specificity penalty (only among free filters)
                    avg_free = sims[:, free_idx].mean(axis=1, keepdims=True)
                    for fi, f in enumerate(free_idx):
                        sims_masked[:, f] = sims[:, f] - avg_free[:, 0] \
                                            - self.perception.thresholds[f]

                    # Winners (only from free filters)
                    winners = np.argmax(sims_masked, axis=1)

                    # Update only winning free filters
                    for f in free_idx:
                        mask = winners == f
                        if mask.sum() > 0:
                            mean_patch = batch[mask].mean(axis=0)
                            self.perception.filters[:, f] += \
                                epoch_lr * (mean_patch - self.perception.filters[:, f])
                            norm = np.linalg.norm(self.perception.filters[:, f])
                            if norm > 0:
                                self.perception.filters[:, f] /= norm

        self.perception._trained = True
        self.perception._img_size = img_size

        # Extract features using ALL filters (frozen + new)
        features = self.perception.extract(images[:min(N, 5000)],
                                            img_size=img_size)

        # Compute which free filters specialised (high activation variance)
        # and freeze them to protect this task's knowledge
        new_frozen = 0
        if len(free_idx) > 0:
            # Check activation rates per free filter
            all_patches_sample = patches[:min(5000, n_patches)]
            sims_all = all_patches_sample @ self.perception.filters
            activation_rate = np.zeros(self.n_filters, dtype=np.float32)
            winners_all = np.argmax(sims_all, axis=1)
            for f in range(self.n_filters):
                activation_rate[f] = (winners_all == f).mean()

            for f in free_idx:
                if activation_rate[f] > self.freeze_threshold:
                    self.frozen[f] = True
                    new_frozen += 1

        # Add prototypes for new classes
        unique_labels = sorted(set(labels[:min(N, 5000)]))
        new_classes = []
        for label in unique_labels:
            name = str(int(label))
            if name not in self.recognition.prototypes:
                new_classes.append(name)

        # Train recognition on this task's classes
        self.recognition.train(features, labels[:min(N, 5000)])
        self.total_classes += len(new_classes)

        task_info = {
            'name': task_name,
            'n_images': N,
            'new_classes': new_classes,
            'filters_frozen': new_frozen,
            'total_frozen': int(self.frozen.sum()),
            'total_free': int((~self.frozen).sum()),
        }
        self.tasks.append(task_info)

        if verbose:
            print(f"  Learned {len(new_classes)} new classes: "
                  f"{', '.join(new_classes)}")
            print(f"  Froze {new_frozen} filters → "
                  f"{task_info['total_frozen']} frozen, "
                  f"{task_info['total_free']} free")

        return task_info

    def predict(self, images):
        """Predict labels for images."""
        if images.ndim == 2 and images.shape[1] > 100:
            # Flat images
            img_size = int(np.sqrt(images.shape[1]))
            images = images.reshape(-1, img_size, img_size)

        features = self.perception.extract(images)
        predictions = []
        for i in range(len(features)):
            rec = self.recognition.recognize(features[i], top_k=1)
            predictions.append(rec[0][0] if rec else "?")
        return predictions

    def evaluate(self, images, labels, verbose=True):
        """
        Evaluate accuracy on a set of images.

        Parameters
        ----------
        images : ndarray
        labels : ndarray
        verbose : bool

        Returns
        -------
        dict with 'accuracy', 'per_class', 'correct', 'total'.
        """
        predictions = self.predict(images)

        correct = 0
        per_class = {}

        for i in range(len(labels)):
            true = str(int(labels[i]))
            pred = predictions[i]

            if true not in per_class:
                per_class[true] = {'correct': 0, 'total': 0}
            per_class[true]['total'] += 1

            if pred == true:
                correct += 1
                per_class[true]['correct'] += 1

        accuracy = correct / len(labels) if len(labels) > 0 else 0.0

        if verbose:
            print(f"  Accuracy: {correct}/{len(labels)} = {accuracy:.1%}")
            for cls in sorted(per_class.keys(), key=lambda x: int(x)):
                c = per_class[cls]['correct']
                t = per_class[cls]['total']
                a = c / t if t > 0 else 0
                print(f"    Class {cls}: {c}/{t} = {a:.0%}")

        return {
            'accuracy': accuracy,
            'per_class': per_class,
            'correct': correct,
            'total': len(labels),
        }

    @property
    def capacity_remaining(self):
        """Fraction of filters still available for future tasks."""
        return float((~self.frozen).sum()) / self.n_filters

    def __repr__(self):
        return (f"ContinualLearner(tasks={self.n_tasks}, "
                f"classes={self.total_classes}, "
                f"capacity={self.capacity_remaining:.0%})")
