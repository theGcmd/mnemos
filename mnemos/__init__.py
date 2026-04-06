"""
Mnemos — Neuromorphic AI built on Hebbian learning.

A Python library for building biologically plausible AI systems
using purely local Hebbian learning rules. No backpropagation.
No gradients. No loss functions. Every weight update depends only
on pre-synaptic and post-synaptic activity: dw = f(pre, post).

⚠️  ALPHA — Mnemos is under active development. APIs may change.

Core components:
    HebbianFilters      — Competitive feature learning with specificity penalty
    HebbianMemory       — Associative knowledge storage in weight matrices
    PrototypeBridge     — Multi-prototype pattern recognition
    Brain               — High-level API combining all components

Products:
    AnomalyDetector     — Learn normal, flag anomalies. No labels needed.
    ContinualLearner    — Learn new tasks without forgetting old ones.
    StreamProcessor     — Real-time stream monitoring with drift detection.
    AdaptiveHead        — Plug into any PyTorch model. Replace fine-tuning.

Quick start:
    >>> import mnemos
    >>> brain = mnemos.Brain()
    >>> brain.learn_features(images)
    >>> brain.learn_concepts(features, labels)
    >>> brain.teach("fire", "produces", "heat")
    >>> result = brain.see(image)

Author: Gustav Gausepohl (age 14)
License: Mnemos Dual License (free for research, commercial use requires license)
Status: Alpha — under active development
"""

__version__ = "0.2.0-alpha"
__author__ = "Gustav Gausepohl"
__license__ = "Mnemos Dual License"
__status__ = "Alpha"

from mnemos.competitive import HebbianFilters
from mnemos.memory import HebbianMemory
from mnemos.recognition import PrototypeBridge
from mnemos.brain import Brain
from mnemos.anomaly import AnomalyDetector
from mnemos.continual import ContinualLearner
from mnemos.stream import StreamProcessor
from mnemos.adaptive import AdaptiveHead

__all__ = [
    "HebbianFilters",
    "HebbianMemory",
    "PrototypeBridge",
    "Brain",
    "AnomalyDetector",
    "ContinualLearner",
    "StreamProcessor",
    "AdaptiveHead",
]
