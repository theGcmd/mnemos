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
train_subset = torch.utils.data.Subset(train_ds, range(5000))
test_subset  = torch.utils.data.Subset(test_ds,  range(1000))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_subset,  batch_size=64)

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

print(f"{'n_proto':<10} {'accuracy':<12} {'n_prototypes'}")
print("-" * 35)
for n_proto in [5, 10, 20, 50, 100]:
    head = mnemos.AdaptiveHead(n_proto=n_proto, seed=42)
    head.fit(train_feats, train_labels, verbose=False)
    acc = head.accuracy(test_feats, test_labels)
    total_protos = head.n_prototypes
    print(f"{n_proto:<10} {acc:.1%}        {total_protos}")
