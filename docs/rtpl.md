# Robot Traffic Privacy Layer (RTPL)

> **Threat Mitigation**: Traffic-analysis fingerprinting of encrypted robot-control traffic (arXiv:2312.06802)

---

## Overview

RTPL is a subsystem within TensorGuardFlow that protects robot control traffic from passive network observers who can infer robot actions from encrypted traffic metadata (packet sizes, timing patterns).

### What RTPL Protects Against
- **Traffic fingerprinting** via signal processing (convolution/correlation)
- **Action identification** from encrypted command/feedback patterns
- **Behavioral profiling** of robot workflows

### What RTPL Does NOT Protect Against
- Endpoint compromise (malware on controller/robot)
- Physical side-channels (audio, vibration, EMI)
- Active network attacks (MITM, injection)

---

## Quick Start

```bash
# Install dependencies
pip install xgboost scikit-learn scipy

# Generate synthetic dataset and evaluate attack baseline
tgflow rtpl attack-eval --in ./traces --report report.html --synthetic

# Reproduce paper results (arXiv:2312.06802)
tgflow rtpl reproduce-paper --dataset synthetic --out ./results
```

---

## Defense Modes

### 1. FRONT (Recommended)
**Zero latency, ~33% overhead**

Injects random dummy packets at the *front* of each action using Rayleigh distribution. Disrupts the fingerprintable features concentrated in the initial traffic burst.

```yaml
rtpl:
  enabled: true
  mode: front
  front_max_dummies: 1000
```

### 2. WTF-PAD
**Zero latency, ~77% overhead**

Adaptive padding that learns traffic patterns and fills "gaps" with dummy packets. Best for continuous control streams.

```yaml
rtpl:
  enabled: true
  mode: wtf_pad
```

### 3. Padding Only (Baseline)
**Zero latency, 100-700% overhead**

Simple bucket-based padding. Not recommended due to high overhead for meaningful protection.

```yaml
rtpl:
  enabled: true
  mode: padding_only
  pad_bucket_bytes: 800
```

---

## Configuration Reference

```yaml
rtpl:
  # Master switch
  enabled: true
  mode: front  # off | padding_only | wtf_pad | front
  
  # FRONT settings
  front_max_dummies: 1000
  front_window_min_s: 1.0
  front_window_max_s: 5.0
  
  # WTF-PAD settings
  wtf_pad_gap_threshold_s: 0.1
  
  # Padding settings
  pad_bucket_bytes: 800
  
  # Safety
  determinism_guard:
    profile: collaborative  # surgical | collaborative | warehouse | lab
    on_violation: fallback_padding_only
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `tgflow rtpl capture` | Capture robot traffic to PCAP |
| `tgflow rtpl attack-eval` | Evaluate attack accuracy on traces |
| `tgflow rtpl defend` | Apply defense to traces |
| `tgflow rtpl reproduce-paper` | Reproduce arXiv:2312.06802 results |

---

## Tuning Guide

### Choosing a Defense Mode

| Robot Type | Recommended Mode | Expected Accuracy |
|------------|-----------------|-------------------|
| Surgical (1kHz) | FRONT | ~50% (vs 97% baseline) |
| Collaborative (100Hz) | FRONT or WTF-PAD | ~30-50% |
| Warehouse (10Hz) | WTF-PAD | ~20-40% |

### Determinism Profiles

| Profile | Max Latency | Max Jitter | Use Case |
|---------|-------------|------------|----------|
| `surgical` | 0.5ms | 0.1ms | Medical robotics |
| `collaborative` | 2ms | 1ms | Kinova, UR |
| `warehouse` | 10ms | 5ms | AMRs, AGVs |
| `lab` | 50ms | 20ms | Research |

---

## Architecture

```
src/tensorguard/rtpl/
├── attack/              # Attack reproduction pipeline
│   ├── convolution_detector.py
│   ├── correlation_detector.py
│   ├── feature_extractor.py
│   └── classifier.py
├── defense/             # Zero-delay defenses
│   ├── front.py         # Rayleigh front-loading
│   ├── wtf_pad.py       # Adaptive padding
│   └── padding.py       # Baseline
├── safety/              # Determinism controls
│   └── determinism_guard.py
├── data/                # Data loading
│   ├── trace_loader.py
│   └── synthetic.py
├── config.py            # Settings
└── cli.py               # CLI commands
```

---

## References

- arXiv:2312.06802 - "On the Feasibility of Fingerprinting Collaborative Robot Traffic"
- USENIX Security 2020 - FRONT/GLUE zero-delay defenses
- ESORICS 2016 - WTF-PAD adaptive padding
