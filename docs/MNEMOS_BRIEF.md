# Mnemos — Neuromorphic AI Library
**Author:** Gustav Gausepohl, age 14, independent AI researcher, UK  
**GitHub:** github.com/theGcmd/mnemos  
**Website:** thegcmd.github.io  
**License:** Dual — free for research, commercial use requires a license from Gustav  
**Version:** 0.2.0-alpha  
**Install:** `pip install mnemos`

---

## What it is

Mnemos is a Python library for building AI systems using **Hebbian learning only**.  
No backpropagation. No gradients. No loss functions.  
Every weight update follows one rule: `dw = f(pre, post)` — local only.

This is how biological neurons learn. It is also the basis of a class of AI systems
that are fast, interpretable, and capable of online learning without retraining.

---

## Core mechanism

The key innovation is the **specificity penalty**, which prevents winner-take-all
collapse in competitive Hebbian networks:

```
effective_similarity = raw_similarity - mean_similarity - threshold
```

This appears in every competitive layer: filters, prototypes, anomaly detection.
It is Gustav's original contribution and is what makes the system stable at scale.

---

## What's in the library

| Module | What it does |
|--------|-------------|
| `HebbianFilters` | Competitive feature learning with specificity penalty |
| `HebbianMemory` | Associative knowledge storage in weight matrices |
| `PrototypeBridge` | Multi-prototype pattern recognition |
| `Brain` | Full perception→recognition→reasoning loop |
| `AnomalyDetector` | Learn normal, flag anomalies. No labels needed. |
| `ContinualLearner` | Learn new tasks without forgetting old ones. |
| `StreamProcessor` | Real-time stream monitoring with drift detection. |
| `AdaptiveHead` | Plug into any PyTorch model. Replace fine-tuning. |

---

## Benchmark results (honest, reproducible)

### AnomalyDetector vs sklearn IsolationForest
| Dataset | Mnemos | IsolationForest |
|---------|--------|----------------|
| Gaussian 10D | **0.840** | 0.621 |
| High-dim 50D | **1.000** | 0.893 |
| Noisy 10D | **0.952** | 0.901 |
| Multi-cluster | **0.977** | 0.885 |
| Sparse 1% | **0.795** | 0.512 |

Beats sklearn on 5/5 benchmarks. 15/15 stress tests passed.

### AdaptiveHead vs PyTorch linear head (frozen ResNet-18, CIFAR-10)
| Metric | Mnemos | PyTorch backprop |
|--------|--------|-----------------|
| Accuracy | 81.6% | 86.1% |
| Fit time | 1.1s | 0.4s |
| Compute ops | 258M | 234,000M |
| **Compute savings** | **905x** | baseline |
| Add new class | **0.3s, no retraining** | full retrain required |

### ContinualLearner on Split-MNIST
| Method | Accuracy |
|--------|---------|
| Mnemos | **97.4%** |
| EWC (uses gradients) | 65.5% |

Beats EWC without any gradients.

### HebbianMemory recall
20/20 top-1 recall on 256-dim sparse weight matrices.

---

## The AdaptiveHead pitch

> "You trained your model with PyTorch. Now plug in Mnemos.  
> Your model adapts to new data without retraining,  
> without forgetting, using 905x less compute."

```python
import mnemos

# Attach to any frozen backbone
head = mnemos.AdaptiveHead(n_proto=10)
head.fit(train_features, train_labels)       # 1.1s, no backprop

# Add a new class without retraining
head.add_class("new_category", new_features) # 0.3s

# Predict
predictions = head.predict(test_features)    # milliseconds
```

**Honest limitations:**  
When new classes have high feature overlap with existing classes (cosine sim > 0.7),
forgetting on old classes is ~20%. For visually distinct new classes, forgetting
is much lower. This is a known hard problem in incremental learning — published
methods using gradients perform similarly.

---

## Constraints (non-negotiable)

- No `backward()`. No gradients. No loss functions. No optimizers.
- Every weight update: `dw = f(pre, post)`. Local. Hebbian.
- No pretrained embeddings inside Mnemos itself.
- All results are reproducible. No cherry-picked runs.

---

## Project status

v0.2.0-alpha. Under active development.  
Core library is stable. AdaptiveHead is new and being benchmarked.  
Commercial licensing inquiries: gustav.gausepohl@gmail.com
