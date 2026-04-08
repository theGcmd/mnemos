"""
Mnemos AdaptiveHead vs PyTorch on REAL CIFAR-10
Uses a real ResNet-18 backbone (pre-trained), frozen.
Compares Hebbian adaptation vs backprop fine-tuning.
"""
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import mnemos

print("Loading CIFAR-10...")
transform = T.Compose([
    T.Resize(224),  # ResNet expects 224x224
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_ds = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_ds = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

# Use subsets to keep it fast
train_subset = torch.utils.data.Subset(train_ds, range(5000))
test_subset = torch.utils.data.Subset(test_ds, range(1000))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64)

print("Loading frozen ResNet-18 backbone...")
backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
backbone.fc = nn.Identity()  # remove final layer, get 512-dim features
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

print("Extracting features (this is slow on CPU, ~5 min)...")
def extract(loader):
    feats, labs = [], []
    with torch.no_grad():
        for images, labels in loader:
            f = backbone(images)
            feats.append(f.numpy())
            labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)

train_feats, train_labels = extract(train_loader)
test_feats, test_labels = extract(test_loader)
print(f"Features: {train_feats.shape}")

# ── Mnemos AdaptiveHead ──
print("\nMnemos AdaptiveHead:")
head = mnemos.AdaptiveHead(n_proto=10, seed=42)
t0 = time.time()
head.fit(train_feats, train_labels, verbose=False)
fit_time = time.time() - t0
acc = head.accuracy(test_feats, test_labels)
print(f"  Accuracy: {acc:.1%}")
print(f"  Fit time: {fit_time:.2f}s")

# ── PyTorch backprop linear head ──
print("\nPyTorch linear head (backprop):")
linear = nn.Linear(512, 10)
optim = torch.optim.Adam(linear.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()
train_t = torch.tensor(train_feats)
labels_t = torch.tensor(train_labels, dtype=torch.long)

t0 = time.time()
for epoch in range(10):
    perm = torch.randperm(len(train_t))
    for start in range(0, len(train_t), 64):
        idx = perm[start:start+64]
        optim.zero_grad()
        loss = crit(linear(train_t[idx]), labels_t[idx])
        loss.backward()
        optim.step()
bp_time = time.time() - t0

linear.eval()
with torch.no_grad():
    bp_preds = linear(torch.tensor(test_feats)).argmax(dim=1).numpy()
bp_acc = float((bp_preds == test_labels).mean())
print(f"  Accuracy: {bp_acc:.1%}")
print(f"  Fit time: {bp_time:.2f}s")

# ── Compute savings ──
print("\nCompute savings (ResNet-18 backbone, 11.7M params):")
savings = head.compute_savings(backbone_params=11_700_000, n_samples=5000)
print(f"  Backprop ops: {savings['backprop_ops']:,}")
print(f"  Mnemos ops:   {savings['hebbian_ops']:,}")
print(f"  Savings:      {savings['compute_ratio']:.0f}x")

# ── Add new class test ──
print("\nAdd-new-class test:")
mask8 = train_labels < 8
head8 = mnemos.AdaptiveHead(n_proto=10, seed=42)
head8.fit(train_feats[mask8], train_labels[mask8], verbose=False)
acc_old = head8.accuracy(test_feats[test_labels < 8], test_labels[test_labels < 8])
print(f"  Trained on classes 0-7: {acc_old:.1%}")

t0 = time.time()
head8.add_class("8", train_feats[train_labels == 8])
head8.add_class("9", train_feats[train_labels == 9])
add_time = time.time() - t0
acc_new = head8.accuracy(test_feats, test_labels)
acc_old_after = head8.accuracy(test_feats[test_labels < 8], test_labels[test_labels < 8])
print(f"  Added classes 8,9 in {add_time:.3f}s")
print(f"  All 10 accuracy: {acc_new:.1%}")
print(f"  Old classes: {acc_old_after:.1%} (forgetting: {acc_old_after - acc_old:+.1%})")
print(f"\n══ HONEST RESULTS ══")
print(f"  Mnemos: {acc:.1%}  |  PyTorch: {bp_acc:.1%}  |  Gap: {bp_acc - acc:+.1%}")
print(f"  Compute savings: {savings['compute_ratio']:.0f}x")
print(f"  New class addition: {add_time:.3f}s, forgetting: {acc_old - acc_old_after:+.1%}")
