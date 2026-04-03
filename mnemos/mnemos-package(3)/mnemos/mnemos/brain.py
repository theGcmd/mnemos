"""
High-level Brain API combining perception, recognition, and reasoning.

Provides a simple interface:
    brain.learn_features(images)
    brain.learn_concepts(features, labels)
    brain.teach("fire", "produces", "heat")
    result = brain.see(image)

Author: Gustav Gausepohl
"""

import numpy as np
from mnemos.competitive import HebbianFilters
from mnemos.memory import HebbianMemory
from mnemos.recognition import PrototypeBridge


class Brain:
    """
    The Mnemos Brain — perception, recognition, and reasoning
    in a single Hebbian system.

    Parameters
    ----------
    n_filters : int
        Number of Hebbian convolutional filters.
    n_proto : int
        Prototypes per concept in recognition bridge.
    concept_dim : int
        Dimension of concept vectors for reasoning.
    patch_size : int
        Size of convolutional patches.
    seed : int
        Random seed.

    Example
    -------
    >>> import mnemos
    >>> brain = mnemos.Brain(n_filters=200)
    >>>
    >>> # Train perception
    >>> brain.learn_features(train_images)
    >>>
    >>> # Extract features and train recognition
    >>> features = brain.extract(train_images)
    >>> brain.learn_concepts(features, train_labels)
    >>>
    >>> # Teach knowledge
    >>> brain.teach("3", "properties", "odd")
    >>> brain.teach("3", "similar_to", "8")
    >>>
    >>> # See an image → recognise → reason
    >>> result = brain.see(test_image, true_label=3)
    >>> print(result['predicted'], result['reasoning'])
    """

    def __init__(self, n_filters=200, n_proto=3, concept_dim=256,
                 patch_size=5, seed=42):
        self.perception = HebbianFilters(n_filters, patch_size, seed)
        self.recognition = PrototypeBridge(n_proto, seed + 1)
        self.reasoning = HebbianMemory(concept_dim, seed=seed + 2)

        self._concept_dim = concept_dim
        self._rng = np.random.RandomState(seed + 3)

        # Tracking
        self.total_seen = 0
        self.total_correct = 0

    def learn_features(self, images, **kwargs):
        """
        Train Hebbian convolutional filters on images.

        Parameters
        ----------
        images : ndarray, shape (N, H, W) or (N, H*W)
        **kwargs : passed to HebbianFilters.train()

        Returns
        -------
        dict with training statistics.
        """
        return self.perception.train(images, **kwargs)

    def extract(self, images, **kwargs):
        """
        Extract features from images using trained filters.

        Parameters
        ----------
        images : ndarray

        Returns
        -------
        features : ndarray, shape (N, feature_dim)
        """
        return self.perception.extract(images, **kwargs)

    def learn_concepts(self, features, labels, concept_names=None):
        """
        Train the recognition bridge from labeled features.

        Parameters
        ----------
        features : ndarray, shape (N, feat_dim)
        labels : ndarray, shape (N,)
        concept_names : dict or None
        """
        result = self.recognition.train(features, labels, concept_names)

        # Share concept vectors with reasoning layer
        for name in self.recognition._labels:
            self.reasoning.register(name)

        return result

    def teach(self, subject, relation, obj, lr=0.1):
        """
        Teach a fact to the reasoning system.

        Parameters
        ----------
        subject : str
            e.g., "fire"
        relation : str
            e.g., "produces"
        obj : str
            e.g., "heat"
        """
        return self.reasoning.learn(subject, relation, obj, lr)

    def see(self, image, true_label=None):
        """
        THE FULL LOOP: image → features → concept → reasoning.

        Parameters
        ----------
        image : ndarray, shape (H, W) or (H*W,)
            A single image.
        true_label : int or str or None
            Ground truth label for accuracy tracking.

        Returns
        -------
        dict with:
            'predicted': str — predicted concept
            'confidence': float — recognition confidence
            'top3': list — top 3 predictions
            'reasoning': dict — recalled knowledge
            'correct': bool or None
        """
        # 1. Extract features
        if image.ndim == 1:
            img_size = int(np.sqrt(len(image)))
            image = image.reshape(img_size, img_size)
        features = self.perception.extract(image[np.newaxis, :, :])[0]

        # 2. Recognise
        recognized = self.recognition.recognize(features, top_k=3)
        pred = recognized[0][0] if recognized else "?"
        conf = recognized[0][1] if recognized else 0.0

        # 3. Reason
        reasoning = {}
        if pred != "?":
            for rel in self.reasoning.relations:
                recalled = self.reasoning.recall(pred, rel, top_k=2)
                if recalled:
                    reasoning[rel] = recalled

        # 4. Track accuracy
        correct = None
        if true_label is not None:
            true_str = str(int(true_label)) if isinstance(true_label, (int, float, np.integer)) else str(true_label)
            correct = pred == true_str
            self.total_seen += 1
            if correct:
                self.total_correct += 1

        return {
            'predicted': pred,
            'confidence': conf,
            'top3': recognized[:3] if recognized else [],
            'reasoning': reasoning,
            'correct': correct,
        }

    def think(self, concepts, n_steps=10):
        """
        Spreading activation between concepts.

        Parameters
        ----------
        concepts : list of str
            Seed concepts.
        n_steps : int

        Returns
        -------
        list of (step, [(concept, strength), ...])
        """
        return self.reasoning.spread(concepts, n_steps)

    def imagine_without(self, concept, max_depth=3):
        """
        Counterfactual: what if this concept disappeared?

        Returns list of consequence dicts.
        """
        return self.reasoning.counterfactual(concept, max_depth)

    def recall(self, concept, relation, top_k=3):
        """Direct recall from reasoning memory."""
        return self.reasoning.recall(concept, relation, top_k)

    @property
    def accuracy(self):
        """Running accuracy of .see() predictions."""
        if self.total_seen == 0:
            return 0.0
        return self.total_correct / self.total_seen

    def __repr__(self):
        return (f"Brain(filters={self.perception.n_filters}, "
                f"concepts={self.reasoning.n_concepts}, "
                f"knowledge={self.reasoning.total_updates}, "
                f"accuracy={self.accuracy:.1%})")
