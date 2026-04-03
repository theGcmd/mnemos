"""
Hebbian associative memory for structured knowledge storage.

Stores concepts as sparse random vectors and relations between
concepts as weight matrices. Learning is outer-product Hebbian:
    W[relation] += lr * object_vec @ subject_vec.T

Recall is matrix-vector multiplication:
    result = W[relation] @ subject_vec → find nearest concept

Chain reasoning through sequential matrix multiplication:
    W[rel2] @ W[rel1] @ concept_vec

This is how associative memory works in the brain. Each relation
type (produces, requires, causes, etc.) has its own weight matrix,
like different cortical circuits for different knowledge types.

Author: Gustav Gausepohl
"""

import numpy as np


class HebbianMemory:
    """
    Associative knowledge storage using Hebbian weight matrices.

    Parameters
    ----------
    dim : int
        Dimension of concept vectors. Higher = less cross-talk
        between concepts but more memory. 256 recommended.
    relations : list of str
        Relation types to support. Each gets its own weight matrix.
    sparsity : float
        Fraction of active dimensions in concept vectors (0-1).
        Lower = more orthogonal concepts = less cross-talk.
    seed : int
        Random seed.

    Example
    -------
    >>> mem = HebbianMemory(dim=256)
    >>> mem.learn("fire", "produces", "heat")
    >>> mem.learn("fire", "produces", "light")
    >>> mem.recall("fire", "produces")
    [('heat', 0.72), ('light', 0.71)]
    """

    DEFAULT_RELATIONS = [
        'properties', 'produces', 'enables', 'prevents',
        'destroyed_by', 'similar_to', 'opposite_of', 'causes',
        'requires', 'part_of', 'category', 'examples',
        'next_to', 'greater_than', 'affected_by',
    ]

    def __init__(self, dim=256, relations=None, sparsity=0.2, seed=42):
        self.rng = np.random.RandomState(seed)
        self.dim = dim
        self.sparsity = sparsity

        self.relations = relations or self.DEFAULT_RELATIONS

        # Concept vectors: name → sparse random vector
        self.concepts = {}
        self.concept_names = []
        self.concept_matrix = None  # (n_concepts, dim) for fast lookup

        # Weight matrices: one per relation type
        self.W = {rel: np.zeros((dim, dim), dtype=np.float32)
                  for rel in self.relations}
        self.n_updates = {rel: 0 for rel in self.relations}
        self.total_updates = 0

    def _get_or_create(self, name):
        """Get concept vector, creating sparse random one if new."""
        name = str(name).lower().strip()
        if name not in self.concepts:
            vec = np.zeros(self.dim, dtype=np.float32)
            n_active = max(int(self.dim * self.sparsity), 10)
            idx = self.rng.choice(self.dim, size=n_active, replace=False)
            vec[idx] = self.rng.randn(n_active).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            self.concepts[name] = vec
            self.concept_names.append(name)
            self._rebuild_matrix()
        return self.concepts[name]

    def _rebuild_matrix(self):
        if self.concept_names:
            self.concept_matrix = np.array(
                [self.concepts[n] for n in self.concept_names],
                dtype=np.float32)

    def register(self, name, vec=None):
        """
        Register a concept. Optionally provide a specific vector.

        Parameters
        ----------
        name : str
            Concept name.
        vec : ndarray or None
            If provided, use this vector. Otherwise create random.
        """
        name = str(name).lower().strip()
        if vec is not None:
            self.concepts[name] = vec.copy()
            if name not in self.concept_names:
                self.concept_names.append(name)
            self._rebuild_matrix()
        else:
            self._get_or_create(name)
        return self.concepts[name]

    def learn(self, subject, relation, obj, lr=0.1):
        """
        Hebbian learning: strengthen association.

        W[relation] *= (1 - decay)           # synaptic decay
        W[relation] += lr * obj_vec @ subj_vec.T   # outer product

        Parameters
        ----------
        subject : str
            Source concept.
        relation : str
            Relation type (must be in self.relations).
        obj : str
            Target concept.
        lr : float
            Learning rate.

        Returns
        -------
        bool : True if learned successfully.
        """
        if relation not in self.W:
            return False

        subj_vec = self._get_or_create(subject)
        obj_vec = self._get_or_create(obj)

        # Synaptic decay (prevents saturation)
        self.W[relation] *= 0.999
        # Hebbian outer product
        self.W[relation] += lr * np.outer(obj_vec, subj_vec)

        self.n_updates[relation] += 1
        self.total_updates += 1
        return True

    def recall(self, concept, relation, top_k=3):
        """
        Associative recall through matrix-vector multiplication.

        result = W[relation] @ concept_vec → find nearest concepts

        Parameters
        ----------
        concept : str
            Cue concept.
        relation : str
            Relation to recall through.
        top_k : int
            Number of results to return.

        Returns
        -------
        list of (name, similarity) tuples, sorted by similarity.
        """
        concept = str(concept).lower().strip()
        if relation not in self.W or concept not in self.concepts:
            return []
        if self.concept_matrix is None:
            return []

        result = self.W[relation] @ self.concepts[concept]
        norm = np.linalg.norm(result)
        if norm < 1e-8:
            return []

        sims = self.concept_matrix @ result / (norm + 1e-8)
        indices = np.argsort(-sims)

        results = []
        for idx in indices[:top_k * 2]:
            name = self.concept_names[idx]
            if name == concept:
                continue
            sim = float(sims[idx])
            if sim > 0.03:
                results.append((name, sim))
            if len(results) >= top_k:
                break
        return results

    def spread(self, seeds, n_steps=8):
        """
        Spreading activation through all weight matrices.

        Activates seed concepts, then propagates through all
        relation matrices simultaneously. Emergent concepts
        appear as the activation spreads.

        Parameters
        ----------
        seeds : list of str
            Starting concepts.
        n_steps : int
            Number of propagation steps.

        Returns
        -------
        list of (step, [(concept, strength), ...]) pairs.
        """
        activation = np.zeros(self.dim, dtype=np.float32)
        for c in seeds:
            c = str(c).lower().strip()
            if c in self.concepts:
                activation += self.concepts[c]
        norm = np.linalg.norm(activation)
        if norm > 0:
            activation /= norm

        active_rels = [r for r in self.relations if self.n_updates[r] > 0]
        trail = []

        for step in range(n_steps):
            if self.concept_matrix is None:
                break
            norm = np.linalg.norm(activation)
            if norm < 1e-8:
                break

            sims = self.concept_matrix @ activation / (norm + 1e-8)
            indices = np.argsort(-sims)
            focus = [(self.concept_names[i], float(sims[i]))
                     for i in indices[:4] if sims[i] > 0.03]
            if focus:
                trail.append((step, focus))

            spread = np.zeros(self.dim, dtype=np.float32)
            for rel in active_rels:
                spread += 0.15 * (self.W[rel] @ activation)
            activation = 0.5 * activation + spread
            norm = np.linalg.norm(activation)
            if norm > 1e-6:
                activation /= norm
            else:
                break

        return trail

    def counterfactual(self, concept, max_depth=3):
        """
        Counterfactual reasoning: what if this concept disappeared?

        Traces cascading consequences through weight matrices.
        "What produces/enables/causes this?" → those things stop.
        Recurse on stopped things.

        Parameters
        ----------
        concept : str
            Concept to remove.
        max_depth : int
            Maximum cascade depth.

        Returns
        -------
        list of consequence dicts with 'target', 'depth', 'desc'.
        """
        concept = str(concept).lower().strip()
        if concept not in self.concepts:
            return []

        consequences = []
        visited = set()

        def trace(entity, depth, prefix=""):
            if depth > max_depth or entity in visited:
                return
            visited.add(entity)
            for rel in ['produces', 'enables', 'causes']:
                recalled = self.recall(entity, rel, top_k=3)
                for name, sim in recalled:
                    if sim > 0.08 and name not in visited:
                        verb = {'produces': 'stops', 'enables': 'disabled',
                                'causes': 'prevented'}[rel]
                        consequences.append({
                            'target': name, 'depth': depth,
                            'desc': f"{prefix}Without {entity}, "
                                    f"{name} {verb}"
                        })
                        trace(name, depth + 1, prefix + "  ")

        trace(concept, 0)
        return consequences

    @property
    def n_concepts(self):
        return len(self.concept_names)

    def __repr__(self):
        return (f"HebbianMemory(dim={self.dim}, "
                f"concepts={self.n_concepts}, "
                f"updates={self.total_updates})")
