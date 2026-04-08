import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import mnemos

transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_ds = torchvision.datasets.CIFAR10('../data', train=True, download=False, transform=transform)
test_ds  = torchvision.datasets.CIFAR10('../data', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_ds, range(5000)), batch_size=64, shuffle=False)
test_loader  = torch.utils.data.DataLoader(torch.utils.data.Subset(test_ds,  range(1000)), batch_size=64)

print("Extracting features...")
backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
backbone.fc = nn.Identity()
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

def extract(loader):
    feats, labs = [], []
    with torch.no_grad():
        for images, labels in loader:
            feats.append(backbone(images).numpy())
            labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)

train_feats, train_labels = extract(train_loader)
test_feats,  test_labels  = extract(test_loader)
print(f"Done. {train_feats.shape}\n")

# Normalize
train_n = train_feats / (np.linalg.norm(train_feats, axis=1, keepdims=True) + 1e-8)
test_n  = test_feats  / (np.linalg.norm(test_feats,  axis=1, keepdims=True) + 1e-8)

def predict_raw(head, features):
    """Nearest prototype, no specificity penalty."""
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms
    preds = []
    for x in features:
        best_class, best_sim = None, -np.inf
        for name, protos in head.prototypes.items():
            sim = max(float(x @ p) for p in protos)
            if sim > best_sim:
                best_sim = sim
                best_class = name
        preds.append(best_class)
    return np.array(preds)

print(f"{'n_proto':<10} {'with specificity':<20} {'without specificity'}")
print("-" * 50)
for n_proto in [5, 10, 20, 50]:
    head = mnemos.AdaptiveHead(n_proto=n_proto, seed=42)
    head.fit(train_feats, train_labels, verbose=False)

    acc_with    = head.accuracy(test_feats, test_labels)
    preds_raw   = predict_raw(head, test_feats)
    labels_str  = np.array([str(l) for l in test_labels])
    acc_without = float((preds_raw == labels_str).mean())

    print(f"{n_proto:<10} {acc_with:<20.1%} {acc_without:.1%}")
