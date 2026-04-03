# Mnemos

**Neuromorphic AI built on Hebbian learning.**

> ⚠️ **Alpha** — Mnemos is under active development. APIs may change. Contributions welcome.

Mnemos is a Python library for building biologically plausible AI systems using **purely local Hebbian learning rules**. No backpropagation. No gradients. No loss functions. Every weight update depends only on pre-synaptic and post-synaptic activity.

```python
import mnemos

brain = mnemos.Brain(n_filters=200)
brain.learn_features(train_images)
features = brain.extract(train_images)
brain.learn_concepts(features, labels)
brain.teach("fire", "produces", "heat")

result = brain.see(test_image)
# → {'predicted': '3', 'reasoning': {'properties': [('odd', 0.71), ('prime', 0.68)]}}
```

## Why Mnemos?

Modern deep learning requires backpropagation — global error signals propagated backward through every layer. This works brilliantly but is:

- **Biologically implausible** — real neurons don't have symmetric backward connections
- **Power hungry** — backprop needs stored activations for every layer
- **Cloud dependent** — can't run on tiny edge devices without a GPU

Mnemos uses **competitive Hebbian learning** — the same principle brains use. Neurons that fire together, wire together. This means:

- **Runs on anything** — no GPU required, minimal memory
- **Learns locally** — every weight update is local, no global coordination needed
- **Continual learning** — learns new tasks without forgetting old ones (97.4% on Split-MNIST)
- **Edge-ready** — designed for neuromorphic chips, drones, IoT sensors

## Key Results

| Metric | Mnemos | EWC (backprop) |
|--------|--------|----------------|
| Split-MNIST continual learning | **97.4%** | 65.5% |
| Backpropagation needed | **No** | Yes |
| Gradient computation | **None** | Full |
| Learning rule | Local Hebbian | Global gradient |

## Core Components

### `HebbianFilters` — Competitive Feature Learning

Learns visual features through competition. The **specificity penalty** prevents winner-take-all collapse — the core mechanism that makes everything work.

```python
from mnemos import HebbianFilters

filters = HebbianFilters(n_filters=400, patch_size=5)
filters.train(images, n_epochs=10)
features = filters.extract(test_images)
```

### `HebbianMemory` — Associative Knowledge Storage

Stores concepts as sparse vectors and relations as weight matrices. Learning is outer-product Hebbian. Recall is matrix-vector multiplication.

```python
from mnemos import HebbianMemory

mem = HebbianMemory(dim=256)
mem.learn("fire", "produces", "heat")
mem.learn("fire", "requires", "oxygen")
mem.recall("fire", "produces")  # → [('heat', 0.72)]

# Counterfactual: what if fire disappeared?
mem.counterfactual("fire")  # → cascading consequences

# Spreading activation: think about fire + ice
mem.spread(["fire", "ice"])  # → 'heat' emerges as bridge
```

### `PrototypeBridge` — Multi-Prototype Recognition

Multiple prototypes per concept capture variation. Competitive matching with specificity penalty.

```python
from mnemos import PrototypeBridge

bridge = PrototypeBridge(n_proto=3)
bridge.train(features, labels)
bridge.recognize(new_features)  # → [('3', 0.85), ('8', 0.12)]
```

### `Brain` — The Full Loop

Combines everything into a single system: see an image → extract features → recognise concept → reason about it.

```python
from mnemos import Brain

brain = Brain(n_filters=200, n_proto=3, concept_dim=256)
brain.learn_features(images)
brain.learn_concepts(brain.extract(images), labels)
brain.teach("3", "properties", "odd")

result = brain.see(test_image, true_label=3)
# Perception → Recognition → Reasoning in one call
```

## Installation

```bash
pip install mnemos
```

Or from source:

```bash
git clone https://github.com/theGcmd/mnemos.git
cd mnemos
pip install -e .
```

## The Specificity Penalty

The core mechanism that makes Mnemos work. In competitive learning, filters tend to collapse — all converging to the global mean of the data. The specificity penalty prevents this:

```
effective_similarity = raw_similarity - mean_similarity - threshold
```

Three lines of code. Appears in:
- **Perception** (prevents filter collapse)
- **Recognition** (prevents concept collapse)
- **Every layer** that uses competitive matching

This is the novel contribution. It's simple, effective, and biologically plausible (homeostatic plasticity in real neurons serves the same function).

## Status

Mnemos is in **alpha**. The core algorithms work and are tested. What's coming:

- [ ] GPU acceleration (CuPy backend)
- [ ] Real-time learning (online mode)
- [ ] Neuromorphic chip support (SpiNNaker, Loihi)
- [ ] Pre-trained filter banks
- [ ] More benchmarks (CIFAR-10, Fashion-MNIST)
- [ ] Modular brain architecture (multiple specialized modules)

## License

**Free for research and non-commercial use.** Commercial use requires a license. See [LICENSE](LICENSE) for details.

## Citation

If you use Mnemos in academic work, please cite:

```
@software{mnemos2026,
  author = {Gausepohl, Gustav},
  title = {Mnemos: Neuromorphic AI Built on Hebbian Learning},
  year = {2026},
  url = {https://github.com/theGcmd/mnemos}
}
```

## Author

**Gustav Gausepohl** — Independent AI researcher, age 14.

Built entirely from scratch. No pre-trained models. No borrowed architectures. Every mechanism derived from first principles of Hebbian learning.

---

*"Neurons that fire together, wire together." — Donald Hebb, 1949*
