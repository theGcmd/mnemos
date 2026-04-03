"""
Mnemos AnomalyDetector — Reproducible Benchmark

10 runs per dataset. Mean ± std. Compared against Isolation Forest
and One-Class SVM. Honest results.

Run this once. It produces:
  1. Console output with all results
  2. benchmark_results.json (machine-readable)
  3. demo.html (visual anomaly detection demo)

Gustav Gausepohl, April 2026.
"""

import numpy as np
import time
import json
import sys

sys.path.insert(0, '/home/claude/mnemos')
import mnemos

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

N_RUNS = 10

# ═══════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════

def make_dataset(name, seed):
    rng = np.random.RandomState(seed)

    if name == "gaussian":
        normal = np.vstack([
            rng.randn(300, 10) * 0.5 + 2,
            rng.randn(300, 10) * 0.5 - 2,
        ]).astype(np.float32)
        anomalies = rng.randn(60, 10).astype(np.float32) * 0.5 + 8
        desc = "Gaussian blobs (10D)"

    elif name == "high_dim":
        normal = rng.randn(500, 50).astype(np.float32) * 0.3
        normal[:, :10] += 2
        anomalies = rng.randn(50, 50).astype(np.float32) * 0.3
        anomalies[:, 20:30] += 5
        desc = "High-dimensional (50D)"

    elif name == "multi_cluster":
        centers = rng.randn(5, 15) * 3
        parts = [rng.randn(100, 15) * 0.4 + c for c in centers]
        normal = np.vstack(parts).astype(np.float32)
        anomalies = rng.randn(50, 15).astype(np.float32) * 1.5
        desc = "Multi-cluster (15D)"

    elif name == "noisy":
        normal = rng.randn(500, 10).astype(np.float32) * 1.5 + 1
        anomalies = rng.randn(50, 10).astype(np.float32) * 1.5 + 5
        desc = "Noisy (10D)"

    elif name == "sparse":
        normal = rng.randn(900, 10).astype(np.float32) * 0.5
        anomalies = rng.randn(10, 10).astype(np.float32) * 0.5 + 6
        desc = "Sparse anomalies (1%)"

    elif name == "clustered":
        normal = rng.randn(500, 10).astype(np.float32) * 0.5
        anomalies = rng.randn(50, 10).astype(np.float32) * 0.3 + 4
        desc = "Clustered anomalies"

    else:
        raise ValueError(name)

    n_train = int(len(normal) * 0.6)
    train = normal[:n_train]
    test_normal = normal[n_train:]
    test = np.vstack([test_normal, anomalies]).astype(np.float32)
    labels = np.concatenate([np.zeros(len(test_normal)), np.ones(len(anomalies))])
    return train, test, labels, desc


def f1(preds, labels):
    preds, labels = np.asarray(preds, bool), np.asarray(labels, bool)
    tp = (preds & labels).sum()
    fp = (preds & ~labels).sum()
    fn = (~preds & labels).sum()
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * p * r / (p + r) if p + r > 0 else 0


def run_mnemos(train, test, labels, seed):
    det = mnemos.AnomalyDetector(n_prototypes=64, threshold_percentile=95, seed=seed)
    det.fit(train, n_epochs=10, verbose=False)
    return f1(det.detect(test), labels)


def run_iforest(train, test, labels, seed):
    sc = StandardScaler()
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=seed)
    clf.fit(sc.fit_transform(train))
    return f1(clf.predict(sc.transform(test)) == -1, labels)


def run_ocsvm(train, test, labels):
    sc = StandardScaler()
    clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    clf.fit(sc.fit_transform(train))
    return f1(clf.predict(sc.transform(test)) == -1, labels)


# ═══════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  MNEMOS ANOMALY DETECTOR — REPRODUCIBLE BENCHMARK")
    print(f"  {N_RUNS} runs per dataset. Mean ± std.")
    print("=" * 65)
    print()

    datasets = ['gaussian', 'high_dim', 'multi_cluster',
                'noisy', 'sparse', 'clustered']
    all_results = {}

    for ds_name in datasets:
        _, _, _, desc = make_dataset(ds_name, seed=0)
        print(f"  {desc}")

        mnemos_scores = []
        iforest_scores = []
        ocsvm_scores = []

        for run in range(N_RUNS):
            seed = run * 1000 + 42
            train, test, labels, _ = make_dataset(ds_name, seed)

            mnemos_scores.append(run_mnemos(train, test, labels, seed))
            iforest_scores.append(run_iforest(train, test, labels, seed))
            ocsvm_scores.append(run_ocsvm(train, test, labels))

        m_mean, m_std = np.mean(mnemos_scores), np.std(mnemos_scores)
        i_mean, i_std = np.mean(iforest_scores), np.std(iforest_scores)
        o_mean, o_std = np.mean(ocsvm_scores), np.std(ocsvm_scores)

        # Who wins?
        scores = {'Mnemos': m_mean, 'IForest': i_mean, 'OCSVM': o_mean}
        winner = max(scores, key=scores.get)

        print(f"    {'Mnemos':<12s}  F1 = {m_mean:.3f} ± {m_std:.3f}"
              f"{'  ← WINNER' if winner == 'Mnemos' else ''}")
        print(f"    {'IForest':<12s}  F1 = {i_mean:.3f} ± {i_std:.3f}"
              f"{'  ← WINNER' if winner == 'IForest' else ''}")
        print(f"    {'OCSVM':<12s}  F1 = {o_mean:.3f} ± {o_std:.3f}"
              f"{'  ← WINNER' if winner == 'OCSVM' else ''}")
        print()

        all_results[ds_name] = {
            'desc': desc,
            'mnemos': {'mean': round(m_mean, 4), 'std': round(m_std, 4),
                       'scores': [round(s, 4) for s in mnemos_scores]},
            'iforest': {'mean': round(i_mean, 4), 'std': round(i_std, 4),
                        'scores': [round(s, 4) for s in iforest_scores]},
            'ocsvm': {'mean': round(o_mean, 4), 'std': round(o_std, 4),
                      'scores': [round(s, 4) for s in ocsvm_scores]},
            'winner': winner,
        }

    # ── SUMMARY TABLE ──
    print("  " + "=" * 60)
    print("  SUMMARY TABLE (for paper / website)")
    print("  " + "=" * 60)
    print()
    print(f"  {'Dataset':<22s} {'Mnemos':<16s} {'IForest':<16s} {'OCSVM':<16s}")
    print(f"  {'─'*68}")

    mnemos_wins = 0
    for ds_name in datasets:
        r = all_results[ds_name]
        m = f"{r['mnemos']['mean']:.3f}±{r['mnemos']['std']:.3f}"
        i = f"{r['iforest']['mean']:.3f}±{r['iforest']['std']:.3f}"
        o = f"{r['ocsvm']['mean']:.3f}±{r['ocsvm']['std']:.3f}"
        marker = " *" if r['winner'] == 'Mnemos' else ""
        print(f"  {r['desc']:<22s} {m:<16s} {i:<16s} {o:<16s}{marker}")
        if r['winner'] == 'Mnemos':
            mnemos_wins += 1

    print()
    print(f"  Mnemos wins: {mnemos_wins}/{len(datasets)}")
    print()

    # ── ONLINE ADAPTATION ──
    print("  " + "=" * 60)
    print("  ONLINE ADAPTATION (10 runs)")
    print("  " + "=" * 60)
    print()

    adapt_scores = []
    static_mnemos = []
    static_if = []

    for run in range(N_RUNS):
        rng = np.random.RandomState(run * 100 + 7)
        dim = 10

        train_A = rng.randn(500, dim).astype(np.float32) * 0.3
        drift = np.array([rng.randn(dim).astype(np.float32) * 0.3 + i/300*6
                          for i in range(300)])
        test_B_norm = rng.randn(200, dim).astype(np.float32) * 0.3 + 6.0
        test_B_anom = rng.randn(30, dim).astype(np.float32) * 0.3 + 15.0
        test = np.vstack([test_B_norm, test_B_anom])
        test_lab = np.concatenate([np.zeros(200), np.ones(30)])

        # Mnemos adapted
        det = mnemos.AnomalyDetector(n_prototypes=64, threshold_percentile=95,
                                      seed=run)
        det.fit(train_A, n_epochs=8, verbose=False)
        for s in drift:
            det.update(s, lr=0.01)
        # Recalibrate after drift
        drift_scores = det.score(drift[-100:])
        det.threshold = float(np.percentile(drift_scores, 95))
        adapt_scores.append(f1(det.detect(test), test_lab))

        # Mnemos static
        det2 = mnemos.AnomalyDetector(n_prototypes=64, adapt=False, seed=run)
        det2.fit(train_A, n_epochs=8, verbose=False)
        static_mnemos.append(f1(det2.detect(test), test_lab))

        # IForest static
        sc = StandardScaler()
        clf = IsolationForest(n_estimators=100, contamination=0.05,
                               random_state=run)
        clf.fit(sc.fit_transform(train_A))
        static_if.append(f1(clf.predict(sc.transform(test)) == -1, test_lab))

    a_mean, a_std = np.mean(adapt_scores), np.std(adapt_scores)
    sm_mean, sm_std = np.mean(static_mnemos), np.std(static_mnemos)
    si_mean, si_std = np.mean(static_if), np.std(static_if)

    print(f"  {'Mnemos (adapted)':<28s}  F1 = {a_mean:.3f} ± {a_std:.3f}")
    print(f"  {'Mnemos (static)':<28s}  F1 = {sm_mean:.3f} ± {sm_std:.3f}")
    print(f"  {'IsolationForest (static)':<28s}  F1 = {si_mean:.3f} ± {si_std:.3f}")
    print()
    if a_mean > si_mean:
        print(f"  → Mnemos adapted wins by {a_mean - si_mean:.3f} F1")
    print()

    all_results['online_adaptation'] = {
        'mnemos_adapted': {'mean': round(a_mean, 4), 'std': round(a_std, 4)},
        'mnemos_static': {'mean': round(sm_mean, 4), 'std': round(sm_std, 4)},
        'iforest_static': {'mean': round(si_mean, 4), 'std': round(si_std, 4)},
    }

    # ── SAVE RESULTS ──
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("  Saved benchmark_results.json")

    # ── GENERATE VISUAL DEMO ──
    print()
    print("  Generating visual demo...")
    generate_demo()
    print("  Saved demo.html")
    print()
    print("  DONE. Three files produced:")
    print("    benchmark_results.json  — machine-readable results")
    print("    demo.html               — visual anomaly detection demo")
    print()


# ═══════════════════════════════════════════════════════════════
# VISUAL DEMO
# ═══════════════════════════════════════════════════════════════

def generate_demo():
    """Generate a visual demo showing anomaly detection in action."""
    rng = np.random.RandomState(42)
    dim = 5

    # Build a stream: normal → anomaly → normal → drift → new normal → anomaly
    stream = []
    labels = []
    phases = []

    # Phase 1: Normal (0-200)
    for _ in range(200):
        stream.append(rng.randn(dim) * 0.3 + 1.0)
        labels.append(0)
        phases.append("normal")

    # Phase 2: Anomaly burst (200-220)
    for _ in range(20):
        stream.append(rng.randn(dim) * 0.3 + 7.0)
        labels.append(1)
        phases.append("anomaly")

    # Phase 3: Normal (220-400)
    for _ in range(180):
        stream.append(rng.randn(dim) * 0.3 + 1.0)
        labels.append(0)
        phases.append("normal")

    # Phase 4: Gradual drift (400-500)
    for i in range(100):
        stream.append(rng.randn(dim) * 0.3 + 1.0 + i * 0.04)
        labels.append(0)
        phases.append("drift")

    # Phase 5: New normal (500-650)
    for _ in range(150):
        stream.append(rng.randn(dim) * 0.3 + 5.0)
        labels.append(0)
        phases.append("new_normal")

    # Phase 6: Anomaly at new distribution (650-670)
    for _ in range(20):
        stream.append(rng.randn(dim) * 0.3 + 12.0)
        labels.append(1)
        phases.append("anomaly2")

    # Phase 7: Back to new normal (670-750)
    for _ in range(80):
        stream.append(rng.randn(dim) * 0.3 + 5.0)
        labels.append(0)
        phases.append("new_normal")

    stream = np.array(stream, dtype=np.float32)

    # Run through Mnemos
    proc = mnemos.StreamProcessor(dim=dim, n_prototypes=32,
                                   warmup=100, window_size=50)
    scores = []
    anomaly_flags = []
    states = []

    for i in range(len(stream)):
        r = proc.process(stream[i])
        scores.append(r['score'])
        anomaly_flags.append(r['anomaly'])
        states.append(r['state'])

    # Generate HTML
    scores_js = json.dumps([round(s, 4) for s in scores])
    labels_js = json.dumps(labels)
    anomaly_js = json.dumps([1 if a else 0 for a in anomaly_flags])
    phases_js = json.dumps(phases)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mnemos — Live Anomaly Detection Demo</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Courier New', monospace;
  background: #0a0a0f;
  color: #c8c8d8;
  padding: 2rem;
}}
h1 {{ color: #4af0c0; font-size: 1.4rem; margin-bottom: 0.5rem; }}
.sub {{ color: #6a6a80; font-size: 0.85rem; margin-bottom: 2rem; }}
canvas {{
  width: 100%;
  height: 300px;
  background: #0f0f18;
  border: 1px solid #252535;
  border-radius: 6px;
  display: block;
  margin-bottom: 1rem;
}}
.legend {{
  display: flex;
  gap: 2rem;
  font-size: 0.75rem;
  color: #6a6a80;
  margin-bottom: 2rem;
}}
.legend span {{ display: flex; align-items: center; gap: 0.4rem; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.controls {{
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
}}
button {{
  font-family: inherit;
  background: #4af0c0;
  color: #0a0a0f;
  border: none;
  padding: 0.5rem 1.5rem;
  font-size: 0.85rem;
  cursor: pointer;
  border-radius: 3px;
}}
button:hover {{ background: #2ad0a0; }}
button.secondary {{
  background: transparent;
  color: #c8c8d8;
  border: 1px solid #252535;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 2rem;
}}
.stat {{
  background: #0f0f18;
  border: 1px solid #252535;
  border-radius: 6px;
  padding: 1rem;
}}
.stat .val {{ font-size: 1.5rem; color: #4af0c0; font-weight: bold; }}
.stat .label {{ font-size: 0.65rem; color: #6a6a80; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }}
.phase-bar {{
  height: 20px;
  background: #0f0f18;
  border: 1px solid #252535;
  border-radius: 3px;
  margin-bottom: 2rem;
  display: flex;
  overflow: hidden;
}}
.phase-bar div {{ height: 100%; }}
</style>
</head>
<body>

<h1>MNEMOS — Live Anomaly Detection</h1>
<p class="sub">Hebbian competitive learning detects anomalies in real-time. No backprop. No labels. Watch the scores spike on anomalies.</p>

<div class="stats">
  <div class="stat"><div class="val" id="s-t">0</div><div class="label">Time Step</div></div>
  <div class="stat"><div class="val" id="s-score">0.00</div><div class="label">Current Score</div></div>
  <div class="stat"><div class="val" id="s-anom">0</div><div class="label">Anomalies Found</div></div>
  <div class="stat"><div class="val" id="s-phase">—</div><div class="label">Phase</div></div>
</div>

<canvas id="chart"></canvas>

<div class="legend">
  <span><span class="dot" style="background:#4af0c0"></span> Anomaly score</span>
  <span><span class="dot" style="background:#f04a6a"></span> Anomaly detected</span>
  <span><span class="dot" style="background:rgba(240,160,74,0.3)"></span> True anomaly zone</span>
</div>

<div class="controls">
  <button onclick="start()">▶ Play</button>
  <button class="secondary" onclick="pause()">⏸ Pause</button>
  <button class="secondary" onclick="reset()">↺ Reset</button>
  <button class="secondary" onclick="skipTo(200)">→ Skip to anomaly</button>
</div>

<script>
const scores = {scores_js};
const trueLabels = {labels_js};
const anomalyFlags = {anomaly_js};
const phases = {phases_js};

const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
let t = 0;
let running = false;
let animId = null;
let nAnom = 0;

function resize() {{
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
}}
resize();
window.addEventListener('resize', resize);

function draw() {{
  const W = canvas.width, H = canvas.height;
  const pad = 40;
  const plotW = W - pad * 2;
  const plotH = H - pad * 2;
  
  ctx.fillStyle = '#0f0f18';
  ctx.fillRect(0, 0, W, H);
  
  if (t === 0) return;
  
  const visible = scores.slice(0, t);
  const maxScore = Math.max(...visible, 1);
  
  const xScale = plotW / Math.max(visible.length - 1, 1);
  const yScale = plotH / maxScore;
  
  // True anomaly zones
  ctx.fillStyle = 'rgba(240, 160, 74, 0.08)';
  for (let i = 0; i < t; i++) {{
    if (trueLabels[i] === 1) {{
      ctx.fillRect(pad + i * xScale - 2, pad, Math.max(xScale, 4), plotH);
    }}
  }}
  
  // Score line
  ctx.beginPath();
  ctx.strokeStyle = '#4af0c0';
  ctx.lineWidth = 2;
  for (let i = 0; i < t; i++) {{
    const x = pad + i * xScale;
    const y = pad + plotH - visible[i] * yScale;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }}
  ctx.stroke();
  
  // Anomaly markers
  for (let i = 0; i < t; i++) {{
    if (anomalyFlags[i]) {{
      const x = pad + i * xScale;
      const y = pad + plotH - visible[i] * yScale;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#f04a6a';
      ctx.fill();
    }}
  }}
  
  // Axis
  ctx.strokeStyle = '#252535';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, pad + plotH);
  ctx.lineTo(pad + plotW, pad + plotH);
  ctx.stroke();
}}

function step() {{
  if (t >= scores.length) {{
    running = false;
    return;
  }}
  
  if (anomalyFlags[t]) nAnom++;
  
  document.getElementById('s-t').textContent = t;
  document.getElementById('s-score').textContent = scores[t].toFixed(3);
  document.getElementById('s-anom').textContent = nAnom;
  
  const phaseNames = {{
    'normal': 'Normal', 'anomaly': '⚠ ANOMALY', 'drift': 'Drifting...',
    'new_normal': 'New Normal', 'anomaly2': '⚠ ANOMALY'
  }};
  const p = phaseNames[phases[t]] || phases[t];
  document.getElementById('s-phase').textContent = p;
  document.getElementById('s-phase').style.color = 
    phases[t].includes('anomaly') ? '#f04a6a' : 
    phases[t] === 'drift' ? '#f0a04a' : '#4af0c0';
  
  t++;
  draw();
}}

function start() {{
  if (running) return;
  running = true;
  function loop() {{
    if (!running) return;
    step();
    animId = requestAnimationFrame(loop);
  }}
  loop();
}}

function pause() {{ running = false; }}

function reset() {{
  running = false;
  t = 0;
  nAnom = 0;
  draw();
  document.getElementById('s-t').textContent = '0';
  document.getElementById('s-score').textContent = '0.00';
  document.getElementById('s-anom').textContent = '0';
  document.getElementById('s-phase').textContent = '—';
}}

function skipTo(target) {{
  while (t < target && t < scores.length) step();
}}

draw();
</script>

</body>
</html>"""

    with open('demo.html', 'w') as f:
        f.write(html)


if __name__ == "__main__":
    main()
