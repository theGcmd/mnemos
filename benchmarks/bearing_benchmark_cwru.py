"""
Mnemos AnomalyDetector on CWRU Bearing Dataset
Same pipeline as NASA IMS — tests generalization.

Data: https://engineering.case.edu/bearingdatacenter/download-data-file
Place normal .mat files in: data/bearing/cwru/normal/
Place fault  .mat files in: data/bearing/cwru/fault/
"""

import numpy as np
import sys
from pathlib import Path
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).parent.parent))
import mnemos

NORMAL_DIR  = Path("data/bearing/cwru/normal")
FAULT_DIR   = Path("data/bearing/cwru/fault")
SAMPLE_RATE = 12000

def extract_features(sig):
    sig   = sig.astype(np.float32).flatten()
    chunk = 1024
    feats = []
    for i in range(len(sig) // chunk):
        s = sig[i*chunk:(i+1)*chunk]
        tf = [
            float(np.mean(s)), float(np.std(s)),
            float(np.max(np.abs(s))),
            float(np.sqrt(np.mean(s**2))),
            float(np.mean(np.abs(s))),
            float(np.max(np.abs(s))/(np.sqrt(np.mean(s**2))+1e-8)),
            float(np.mean(s**4)/(np.mean(s**2)**2+1e-8)),
        ]
        fft_mag = np.abs(rfft(s))
        freqs   = rfftfreq(len(s), d=1.0/SAMPLE_RATE)
        total   = fft_mag.sum() + 1e-8
        ff = []
        for lo,hi in [(0,500),(500,1000),(1000,2000),(2000,3000),
                      (3000,4000),(4000,5000),(5000,6000),(6000,6000)]:
            mask = (freqs>=lo)&(freqs<hi)
            ff.append(float(fft_mag[mask].sum()/total))
        peak = np.argmax(fft_mag)
        ff  += [float(freqs[peak]), float(fft_mag[peak]/total)]
        feats.append(np.array(tf + ff, dtype=np.float32))
    return np.array(feats)

def load_mat(path):
    try:
        mat = loadmat(str(path))
    except Exception as e:
        print(f"  Skipping {path.name}: {e}")
        return None
    for key in mat:
        if 'DE_time' in key:
            return mat[key].flatten()
    for key in mat:
        if not key.startswith('_'):
            arr = mat[key].flatten()
            if len(arr) > 1000:
                return arr
    return None

def run():
    if not NORMAL_DIR.exists() or not FAULT_DIR.exists():
        print("Data not found. See script header for download instructions.")
        return

    print("Loading normal data...")
    normal_feats = []
    for f in sorted(NORMAL_DIR.glob("*.mat")):
        sig = load_mat(f)
        if sig is not None:
            chunks = extract_features(sig)
            normal_feats.append(chunks)
            print(f"  {f.name}: {len(chunks)} windows")

    print("Loading fault data...")
    fault_feats = []
    for f in sorted(FAULT_DIR.glob("*.mat")):
        sig = load_mat(f)
        if sig is not None:
            chunks = extract_features(sig)
            fault_feats.append(chunks)
            print(f"  {f.name}: {len(chunks)} windows")

    if not normal_feats or not fault_feats:
        print("No data loaded.")
        return

    normal_all = np.concatenate(normal_feats)
    fault_all  = np.concatenate(fault_feats)

    train_end   = int(len(normal_all) * 0.7)
    train_feats = normal_all[:train_end]
    test_normal = normal_all[train_end:]
    test_feats  = np.concatenate([test_normal, fault_all])
    test_labels = np.array([0]*len(test_normal) + [1]*len(fault_all))

    print(f"\nTrain: {len(train_feats)} windows (normal)")
    print(f"Test:  {len(test_feats)} windows ({test_labels.sum()} fault)")

    det = mnemos.AnomalyDetector()
    det.fit(train_feats, verbose=False)

    train_scores = det.score(train_feats)
    test_scores  = det.score(test_feats)
    thresh = np.percentile(train_scores, 99)
    preds  = (test_scores > thresh).astype(int)

    tp = ((preds==1)&(test_labels==1)).sum()
    fp = ((preds==1)&(test_labels==0)).sum()
    fn = ((preds==0)&(test_labels==1)).sum()
    p  = tp/(tp+fp+1e-8)
    r  = tp/(tp+fn+1e-8)
    f1 = 2*p*r/(p+r+1e-8)

    print(f"\n══ CWRU RESULTS ══")
    print(f"  F1:        {f1:.3f}")
    print(f"  Precision: {p:.3f}")
    print(f"  Recall:    {r:.3f}")
    print(f"  No GPU:    True")
    print(f"  No labels: True")

    print(f"\n── Comparison ──")
    print(f"  NASA IMS (previous): F1 0.860, warning 6.2h, 0 FA/day")
    print(f"  CWRU (this run):     F1 {f1:.3f}")

if __name__ == "__main__":
    run()
