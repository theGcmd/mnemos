"""
Example: MNIST digit recognition with the Mnemos Brain.

Shows the full pipeline:
  1. Train Hebbian convolutional filters
  2. Extract features
  3. Train multi-prototype recognition
  4. Teach number knowledge
  5. See digits → recognise → reason

Requires: torchvision (for MNIST download only)
"""

import numpy as np
import mnemos


def load_mnist():
    """Load MNIST using torchvision."""
    import torchvision
    import torchvision.transforms as T
    tr = torchvision.datasets.MNIST('./data', train=True, download=True,
                                     transform=T.ToTensor())
    te = torchvision.datasets.MNIST('./data', train=False, download=True,
                                     transform=T.ToTensor())
    trX = np.array([tr[i][0].numpy().squeeze() for i in range(len(tr))],
                   dtype=np.float32)
    trY = np.array([tr[i][1] for i in range(len(tr))])
    teX = np.array([te[i][0].numpy().squeeze() for i in range(len(te))],
                   dtype=np.float32)
    teY = np.array([te[i][1] for i in range(len(te))])
    return trX, trY, teX, teY


def main():
    print("Mnemos MNIST Example")
    print("=" * 50)
    print()

    # Load data
    print("Loading MNIST...", end="", flush=True)
    trX, trY, teX, teY = load_mnist()
    print(f" done ({len(trX)} train, {len(teX)} test)")

    # Create brain
    brain = mnemos.Brain(n_filters=200, n_proto=3, concept_dim=256)
    print(f"Brain: {brain}")
    print()

    # 1. Train perception
    print("Training Hebbian filters...")
    brain.learn_features(trX[:5000], n_epochs=8)
    print()

    # 2. Extract features
    print("Extracting features...", end="", flush=True)
    train_feats = brain.extract(trX[:5000])
    test_feats = brain.extract(teX[:500])
    print(f" done (shape: {train_feats.shape})")

    # 3. Train recognition
    print("Training recognition bridge...")
    brain.learn_concepts(train_feats, trY[:5000])
    print()

    # 4. Teach number knowledge
    knowledge = [
        ("0", "properties", "even"), ("0", "properties", "round"),
        ("1", "properties", "odd"), ("1", "properties", "straight"),
        ("2", "properties", "even"), ("2", "properties", "prime"),
        ("3", "properties", "odd"), ("3", "properties", "prime"),
        ("4", "properties", "even"), ("4", "properties", "straight"),
        ("5", "properties", "odd"), ("5", "properties", "prime"),
        ("6", "properties", "even"), ("6", "properties", "round"),
        ("7", "properties", "odd"), ("7", "properties", "prime"),
        ("8", "properties", "even"), ("8", "properties", "round"),
        ("9", "properties", "odd"), ("9", "properties", "round"),
        ("3", "similar_to", "8"), ("8", "similar_to", "3"),
        ("1", "similar_to", "7"), ("7", "similar_to", "1"),
    ]
    for s, r, o in knowledge:
        brain.teach(s, r, o)
    print(f"Taught {len(knowledge)} facts")
    print()

    # 5. Test accuracy
    print("Testing recognition accuracy...")
    correct = 0
    for i in range(len(test_feats)):
        rec = brain.recognition.recognize(test_feats[i], top_k=1)
        if rec and rec[0][0] == str(int(teY[i])):
            correct += 1
    acc = correct / len(test_feats) * 100
    print(f"Accuracy: {correct}/{len(test_feats)} = {acc:.1f}%")
    print()

    # 6. Full loop demos
    print("Full loop — see, recognise, reason:")
    print("-" * 50)
    for i in range(5):
        result = brain.see(teX[i], true_label=teY[i])
        mark = "✓" if result['correct'] else "✗"
        print(f"  [{mark}] True={int(teY[i])} → Predicted={result['predicted']}")
        if result['reasoning']:
            for rel, items in list(result['reasoning'].items())[:2]:
                vals = ", ".join(n for n, _ in items)
                print(f"      {result['predicted']} {rel}: {vals}")
    print()

    # 7. Thinking
    print("Thinking about 3 and 8:")
    trail = brain.think(["3", "8"], n_steps=5)
    for step, focus in trail:
        state = " | ".join(f"{n}({s:.2f})" for n, s in focus[:3])
        print(f"  Step {step}: [{state}]")
    print()

    print(f"Final: {brain}")


if __name__ == "__main__":
    main()
