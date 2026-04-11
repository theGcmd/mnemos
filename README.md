# Mnemos

**Neuromorphic AI built on Hebbian learning. No backprop. No gradients. No loss functions.**

> ⚠️ Alpha — APIs may change. Benchmarks are real and reproducible.

```
pip install mnemos-hebbian
```

---

## Results

### Bearing fault detection — real industrial data

| Metric | Mnemos | Standard 3-sigma | Deep learning (GPU) |
|--------|--------|-----------------|-------------------|
| F1 score | **0.876 ± 0.018** | 0.000 | 0.82–0.88 |
| Recall | **1.000** | 0.000 | ~0.85 |
| Warning time | **6.2 hours** | 0 hours | unknown |
| False alarms/day | **0.0** | — | unknown |
| GPU needed | **No** | No | Yes |
| Labels needed | **No** | No | Yes |

Tested on NASA IMS and CWRU — two standard benchmarks used in published papers.
Same code, no modifications between datasets. CWRU F1: **0.998**.

The standard industrial method (3-sigma RMS threshold) gives zero warning on this
failure mode because the bearing degrades gradually. Mnemos catches it 6.2 hours
early, automatically, with no manual tuning.

```python
detector = mnemos.AnomalyDetector()
detector.fit(normal_vibration_data)    # learns normal in <1s, no labels
scores = detector.score(new_data)      # higher = more anomalous
```

---

### AdaptiveHead — plug into any frozen backbone, replace fine-tuning

| Metric | Mnemos | PyTorch backprop |
|--------|--------|-----------------|
| CIFAR-10 accuracy | **82.2%** | 85.2% |
| Compute cost | **258M ops** | 234,000M ops |
| Compute savings | **905x less** | baseline |
| Add new class | **0.3s, no retraining** | full retrain |
| Forgetting (new classes) | **−18%** | N/A |

```python
head = mnemos.AdaptiveHead(n_proto=10)
head.fit(train_features, train_labels)       # 1.1s, no backprop
head.add_class("new_category", new_features) # 0.3s, no retraining
predictions = head.predict(test_features)
```

---

### AnomalyDetector — beats sklearn IsolationForest on 5/5 benchmarks

| Dataset | Mnemos | IsolationForest |
|---------|--------|----------------|
| Gaussian 10D | **0.840 ± 0.063** | 0.621 |
| High-dim 50D | **1.000 ± 0.000** | 0.893 |
| Noisy 10D | **0.952 ± 0.025** | 0.901 |
| Multi-cluster | **0.977 ± 0.016** | 0.885 |
| Sparse 1% | **0.795 ± 0.055** | 0.512 |

15/15 stress tests passed (drift, adversarial, stability).

---

### ContinualLearner — learns new tasks without forgetting

| Method | Split-MNIST accuracy | Uses gradients |
|--------|---------------------|----------------|
| Mnemos | **97.4%** | No |
| EWC (DeepMind) | 65.5% | Yes |

---

## What is Mnemos?

Every weight update in Mnemos follows one rule:

```
dw = f(pre, post)
```

Pre-synaptic activity times post-synaptic activity. Local only. No global error
signal. No backward pass. This is how biological neurons learn.

The key contribution is a **specificity penalty** that prevents winner-take-all
collapse in competitive Hebbian networks:

```
effective_similarity = raw_similarity - mean_similarity - threshold
```

Without this, competitive learning degenerates — all neurons converge to the same
representation. With it, the network maintains diverse, class-specific prototypes.
This appears in every competitive layer in Mnemos.

**AdaptiveHead** additionally uses LVQ2.1 with spherical tangent geometry for
boundary refinement. The standard push/pull update is geometrically wrong on
normalized vectors — it pushes toward the origin rather than away from the confuser
class. Fixing this was the largest single accuracy improvement.

---

## Installation

```bash
pip install mnemos-hebbian
```

From source:

```bash
git clone https://github.com/theGcmd/mnemos.git
cd mnemos
pip install -e .
```

---

## Modules

| Module | What it does |
|--------|-------------|
| `AnomalyDetector` | Learn normal, flag anomalies. No labels needed. |
| `AdaptiveHead` | Plug into any frozen PyTorch backbone. Replace fine-tuning. |
| `ContinualLearner` | Learn new tasks without forgetting old ones. |
| `HebbianFilters` | Competitive feature learning with specificity penalty. |
| `HebbianMemory` | Associative knowledge storage in weight matrices. |
| `PrototypeBridge` | Multi-prototype pattern recognition. |
| `Brain` | Full perception → recognition → reasoning loop. |
| `StreamProcessor` | Real-time stream monitoring with drift detection. |

---

## How AdaptiveHead works

```
Frozen backbone → 512-dim features → Hebbian clustering → LVQ2.1 refinement → Nearest prototype
```

1. **Hebbian clustering** — competitive k-means using only local updates
2. **LVQ2.1 refinement** — winner pulled toward sample, confuser pushed away, spherical tangent geometry
3. **Prediction** — nearest prototype by cosine similarity

No backward(). No optimizer state. No gradient storage.

---

## Honest limitations

- AdaptiveHead has a 3% accuracy gap vs backprop on CIFAR-10
- `add_class()` has ~18% forgetting when new classes have high feature overlap
  with existing ones. This matches published SOTA for gradient-free incremental learning.
- Bearing fault detection tested on two standard benchmarks — real-world deployment
  may surface additional challenges (variable loading, sensor noise, temperature drift)

---

## License

Free for research and non-commercial use. Commercial use requires a license from
the author. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{mnemos2026,
  author = {Gausepohl, Gustav},
  title  = {Mnemos: Neuromorphic AI Built on Hebbian Learning},
  year   = {2026},
  url    = {https://github.com/theGcmd/mnemos}
}
```

---

## Author

Gustav Gausepohl — independent AI researcher, age 14, UK.
[thegcmd.github.io](https://thegcmd.github.io) · gustavgausepohl@gmail.com

*"Neurons that fire together, wire together." — Donald Hebb, 1949*
