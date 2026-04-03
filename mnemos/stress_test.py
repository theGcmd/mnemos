"""
Mnemos AnomalyDetector — Stress Test Suite

Tests the system under real-world conditions:

  1. DRIFT TESTS      — gradual shift, sudden shift, oscillating
  2. ADVERSARIAL       — noise injection, overlapping clusters, edge cases
  3. STABILITY         — threshold drift, confidence consistency, long runs
  4. LOGGING           — full event log for debugging and analysis

Run: python stress_test.py

Gustav Gausepohl, April 2026.
"""

import numpy as np
import sys
import json
from collections import defaultdict

sys.path.insert(0, '.')
import mnemos


def metrics(preds, labels):
    p, l = np.asarray(preds, bool), np.asarray(labels, bool)
    tp = int((p & l).sum())
    fp = int((p & ~l).sum())
    fn = int((~p & l).sum())
    tn = int((~p & ~l).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    return {'f1': f1, 'precision': prec, 'recall': rec,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


class TestLog:
    """Collects all test results and events for analysis."""
    def __init__(self):
        self.tests = []
        self.events = []

    def add(self, name, result, passed):
        self.tests.append({'name': name, 'result': result, 'passed': passed})
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, float):
                    print(f"         {k}: {v:.3f}")

    def event(self, msg):
        self.events.append(msg)

    def summary(self):
        passed = sum(1 for t in self.tests if t['passed'])
        total = len(self.tests)
        print(f"\n  {'=' * 55}")
        print(f"  RESULTS: {passed}/{total} tests passed")
        print(f"  {'=' * 55}")
        if passed < total:
            print(f"\n  FAILURES:")
            for t in self.tests:
                if not t['passed']:
                    print(f"    ✗ {t['name']}")
        return passed, total


def main():
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║     MNEMOS ANOMALY DETECTOR — STRESS TEST SUITE     ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    log = TestLog()

    # ══════════════════════════════════════════════════════════
    # 1. DRIFT TESTS
    # ══════════════════════════════════════════════════════════
    print("  ── 1. DRIFT TESTS ──────────────────────────────────")
    print()

    # 1a. Gradual drift — distribution slowly shifts
    rng = np.random.RandomState(42)
    dim = 10
    train = rng.randn(300, dim).astype(np.float32) * 0.3

    det = mnemos.AnomalyDetector(n_prototypes=32, smooth_window=5, seed=42)
    det.fit(train, n_epochs=10, verbose=False)

    # Gradual shift: mean moves from 0 to 4 over 400 samples
    drift_normal, drift_anomaly = [], []
    for i in range(400):
        shift = i / 400.0 * 4.0
        sample = rng.randn(dim).astype(np.float32) * 0.3 + shift
        r = det.process(sample)
        drift_normal.append(r)

    # After drift, test: can it tell B-normal from B-anomaly?
    post_drift_scores_normal = []
    post_drift_scores_anomaly = []
    for _ in range(100):
        s = rng.randn(dim).astype(np.float32) * 0.3 + 4.0
        r = det.process(s)
        post_drift_scores_normal.append(r['confidence'])
    for _ in range(20):
        s = rng.randn(dim).astype(np.float32) * 0.3 + 12.0
        r = det.process(s)
        post_drift_scores_anomaly.append(r['confidence'])

    normal_conf = np.mean(post_drift_scores_normal)
    anom_conf = np.mean(post_drift_scores_anomaly)
    separation = anom_conf - normal_conf

    log.add("Gradual drift: confidence separates after adaptation",
            {'normal_conf': normal_conf, 'anomaly_conf': anom_conf,
             'separation': separation},
            separation > 0.3)

    # 1b. Sudden shift — distribution jumps instantly
    det2 = mnemos.AnomalyDetector(n_prototypes=32, smooth_window=5, seed=42)
    det2.fit(train, n_epochs=10, verbose=False)

    # Sudden jump to mean=5
    sudden_false_alarms = 0
    for i in range(200):
        s = rng.randn(dim).astype(np.float32) * 0.3 + 5.0
        r = det2.process(s)
        if i > 50 and r['is_anomaly']:  # after 50 adaptation samples
            sudden_false_alarms += 1

    fa_rate = sudden_false_alarms / 150
    log.add("Sudden shift: false alarm rate after 50-sample adaptation",
            {'false_alarm_rate': fa_rate, 'false_alarms': sudden_false_alarms},
            fa_rate < 0.3)

    # 1c. Oscillating — distribution switches back and forth
    det3 = mnemos.AnomalyDetector(n_prototypes=32, smooth_window=5, seed=42)
    det3.fit(train, n_epochs=10, verbose=False)

    oscillation_results = []
    for cycle in range(4):
        mean = 0.0 if cycle % 2 == 0 else 3.0
        for _ in range(100):
            s = rng.randn(dim).astype(np.float32) * 0.3 + mean
            r = det3.process(s)
            oscillation_results.append(r['confidence'])

    # Confidence should stay reasonable (not explode or collapse)
    osc_std = np.std(oscillation_results)
    log.add("Oscillating drift: confidence stays bounded",
            {'confidence_std': osc_std,
             'max_conf': max(oscillation_results),
             'min_conf': min(oscillation_results)},
            osc_std < 1.0)

    print()

    # ══════════════════════════════════════════════════════════
    # 2. ADVERSARIAL / ATTACK TESTS
    # ══════════════════════════════════════════════════════════
    print("  ── 2. ADVERSARIAL TESTS ────────────────────────────")
    print()

    # 2a. Pure noise — random Gaussian noise as input
    det_noise = mnemos.AnomalyDetector(n_prototypes=64, seed=42)
    clean_train = rng.randn(500, 10).astype(np.float32) * 0.3
    det_noise.fit(clean_train, n_epochs=10, verbose=False)

    noise_input = rng.randn(100, 10).astype(np.float32) * 10.0  # 33x normal std
    noise_scores = det_noise.score(noise_input)
    noise_detected = det_noise.detect(noise_input).mean()

    log.add("Pure noise (33x normal std): detection rate",
            {'detection_rate': noise_detected,
             'mean_score': float(noise_scores.mean())},
            noise_detected > 0.8)

    # 2b. Overlapping clusters — anomalies partially overlap normal
    normal_overlap = rng.randn(500, 10).astype(np.float32) * 0.5
    # Anomalies start at 1.5 std — partial overlap
    anom_overlap = rng.randn(50, 10).astype(np.float32) * 0.5 + 1.5

    det_overlap = mnemos.AnomalyDetector(n_prototypes=64, seed=42)
    det_overlap.fit(normal_overlap[:300], n_epochs=10, verbose=False)
    test_ol = np.vstack([normal_overlap[300:], anom_overlap])
    labels_ol = np.concatenate([np.zeros(200), np.ones(50)])
    m_ol = metrics(det_overlap.detect(test_ol), labels_ol)

    log.add("Overlapping clusters (1.5σ separation): F1",
            m_ol, m_ol['f1'] > 0.3)  # this IS hard, >0.3 is reasonable

    # 2c. Very sparse anomalies (0.5%)
    normal_sparse = rng.randn(1000, 10).astype(np.float32) * 0.3
    anom_sparse = rng.randn(5, 10).astype(np.float32) * 0.3 + 5.0

    det_sparse = mnemos.AnomalyDetector(n_prototypes=64, seed=42)
    det_sparse.fit(normal_sparse[:600], n_epochs=10, verbose=False)
    test_sp = np.vstack([normal_sparse[600:], anom_sparse])
    labels_sp = np.concatenate([np.zeros(400), np.ones(5)])
    m_sp = metrics(det_sparse.detect(test_sp), labels_sp)

    log.add("Very sparse anomalies (0.5%): recall",
            m_sp, m_sp['recall'] > 0.6)

    # 2d. Identical anomaly repeated — same vector 50 times
    single_anomaly = np.ones(10, dtype=np.float32) * 8.0
    repeated = np.tile(single_anomaly, (50, 1))
    repeated_scores = det_noise.score(repeated)
    all_same = np.std(repeated_scores) < 0.01

    log.add("Identical anomaly repeated 50x: consistent scoring",
            {'score_std': float(np.std(repeated_scores)),
             'score_mean': float(np.mean(repeated_scores))},
            all_same)

    # 2e. Zero vector — edge case
    zero_vec = np.zeros((1, 10), dtype=np.float32)
    try:
        zero_score = det_noise.score(zero_vec)
        log.add("Zero vector input: no crash",
                {'score': float(zero_score[0])}, True)
    except Exception as e:
        log.add("Zero vector input: no crash",
                {'error': str(e)}, False)

    # 2f. Huge values — numerical stability
    huge = np.ones((1, 10), dtype=np.float32) * 1e6
    try:
        huge_score = det_noise.score(huge)
        is_finite = np.isfinite(huge_score).all()
        log.add("Huge values (1e6): finite output",
                {'score': float(huge_score[0]), 'finite': is_finite},
                is_finite)
    except Exception as e:
        log.add("Huge values (1e6): finite output",
                {'error': str(e)}, False)

    print()

    # ══════════════════════════════════════════════════════════
    # 3. STABILITY TESTS
    # ══════════════════════════════════════════════════════════
    print("  ── 3. STABILITY TESTS ──────────────────────────────")
    print()

    # 3a. Long run — 5000 samples, track threshold drift
    det_long = mnemos.AnomalyDetector(n_prototypes=32, smooth_window=5, seed=42)
    long_train = rng.randn(300, 10).astype(np.float32) * 0.3
    det_long.fit(long_train, n_epochs=10, verbose=False)

    thresholds_over_time = []
    confs_over_time = []
    initial_threshold = det_long._adaptive_threshold

    for i in range(5000):
        s = rng.randn(10).astype(np.float32) * 0.3  # all normal
        r = det_long.process(s)
        if i % 100 == 0:
            thresholds_over_time.append(det_long._adaptive_threshold)
            confs_over_time.append(r['confidence'])

    final_threshold = det_long._adaptive_threshold
    threshold_drift = abs(final_threshold - initial_threshold) / (initial_threshold + 1e-8)

    log.add("5000-sample stability: threshold drift < 20%",
            {'initial': initial_threshold, 'final': final_threshold,
             'drift_pct': threshold_drift * 100},
            threshold_drift < 0.2)

    # 3b. Confidence consistency — same input should get same score
    test_point = rng.randn(10).astype(np.float32) * 0.3
    scores_repeated = [float(det_long.score(test_point.reshape(1, -1))[0])
                       for _ in range(10)]
    score_var = np.var(scores_repeated)

    log.add("Same input → same score (deterministic)",
            {'score_variance': score_var},
            score_var < 1e-10)

    # 3c. Prototype movement — do prototypes drift during normal operation?
    proto_before = det_long.prototypes.copy()
    for _ in range(100):
        s = rng.randn(10).astype(np.float32) * 0.3
        det_long.process(s)
    proto_after = det_long.prototypes
    proto_movement = np.linalg.norm(proto_after - proto_before, axis=1).mean()

    log.add("Prototype movement during normal operation: small",
            {'mean_movement': float(proto_movement)},
            proto_movement < 0.5)

    # 3d. No anomalies in purely normal stream
    det_pure = mnemos.AnomalyDetector(n_prototypes=32, smooth_window=5, seed=42)
    det_pure.fit(rng.randn(300, 10).astype(np.float32) * 0.3,
                  n_epochs=10, verbose=False)

    false_alarms_pure = 0
    for _ in range(1000):
        s = rng.randn(10).astype(np.float32) * 0.3
        r = det_pure.process(s)
        if r['is_anomaly']:
            false_alarms_pure += 1

    fa_rate_pure = false_alarms_pure / 1000
    log.add("1000 normal samples: false alarm rate < 5%",
            {'false_alarms': false_alarms_pure, 'rate': fa_rate_pure},
            fa_rate_pure < 0.05)

    print()

    # ══════════════════════════════════════════════════════════
    # 4. LOGGING / DIAGNOSTICS
    # ══════════════════════════════════════════════════════════
    print("  ── 4. LOGGING & DIAGNOSTICS ────────────────────────")
    print()

    # Full event trace on a stream with anomalies
    det_log = mnemos.AnomalyDetector(n_prototypes=32, smooth_window=5, seed=42)
    det_log.fit(rng.randn(200, 10).astype(np.float32) * 0.3,
                 n_epochs=8, verbose=False)

    event_log = []
    stream_data = []

    # 100 normal, 10 anomalies, 100 normal
    for i in range(100):
        stream_data.append(('normal', rng.randn(10).astype(np.float32) * 0.3))
    for i in range(10):
        stream_data.append(('anomaly', rng.randn(10).astype(np.float32) * 0.3 + 5.0))
    for i in range(100):
        stream_data.append(('normal', rng.randn(10).astype(np.float32) * 0.3))

    for t, (label, sample) in enumerate(stream_data):
        r = det_log.process(sample)
        event_log.append({
            't': t,
            'true_label': label,
            'score': round(r['score'], 4),
            'confidence': round(r['confidence'], 4),
            'smoothed': round(r['smoothed_confidence'], 4),
            'is_anomaly': r['is_anomaly'],
            'streak': r['streak'],
            'threshold': round(det_log._adaptive_threshold, 4),
        })

    # Print key events
    print("  Event trace (key moments):")
    for e in event_log:
        if e['is_anomaly'] or e['t'] in [0, 99, 100, 109, 110, 209]:
            flag = "⚠ ANOMALY" if e['is_anomaly'] else "  normal "
            print(f"    t={e['t']:3d} [{e['true_label']:7s}] {flag} "
                  f"score={e['score']:.3f} conf={e['confidence']:.3f} "
                  f"smooth={e['smoothed']:.3f} streak={e['streak']}")

    # Check logging captures the burst correctly
    burst_detected = sum(1 for e in event_log[100:110] if e['is_anomaly'])
    post_burst_fa = sum(1 for e in event_log[115:] if e['is_anomaly'])

    log.add("Logging: anomaly burst detected in event trace",
            {'burst_detected': burst_detected, 'out_of': 10},
            burst_detected >= 5)

    log.add("Logging: false alarms stop after burst ends",
            {'post_burst_false_alarms': post_burst_fa},
            post_burst_fa < 5)

    # Save full log
    # Save full log
    serializable_log = []
    for e in event_log:
        se = {}
        for k, v in e.items():
            if isinstance(v, (np.bool_, np.integer)):
                se[k] = int(v) if isinstance(v, np.integer) else bool(v)
            else:
                se[k] = v
        serializable_log.append(se)

    with open('stress_test_log.json', 'w') as f:
        json.dump({
            'event_trace': serializable_log,
            'tests': [{'name': t['name'], 'passed': bool(t['passed'])}
                      for t in log.tests],
        }, f, indent=2)
    print(f"\n  Full event log saved to stress_test_log.json")

    print()

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    passed, total = log.summary()

    # Honest assessment
    print()
    failures = [t for t in log.tests if not t['passed']]
    if failures:
        print("  HONEST ASSESSMENT:")
        for f in failures:
            print(f"    The system struggles with: {f['name']}")
            if isinstance(f['result'], dict):
                for k, v in f['result'].items():
                    if isinstance(v, float):
                        print(f"      {k} = {v:.3f}")
    else:
        print("  All tests passed. System is robust under stress.")

    return passed, total


if __name__ == "__main__":
    main()
