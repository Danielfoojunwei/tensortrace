"""
Benchmark: Fingerprinting Collaborative Robot Network Traffic
Reference: "On the Feasibility of Fingerprinting Collaborative Robot Network Traffic" (Cheng Tang et al.)

This script replicates the privacy attack evaluation described in the paper.
It generates synthetic traffic traces for common robotic actions (Move, Pick, Place),
features them using signal processing techniques (IAT/Size Stats), and trains a 
classifier to identify the action.

It then applies TensorGuard's defenses (Padding) and measures the drop in Attack Accuracy.
"""

import numpy as np
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter
import logging

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Benchmark")

# === 1. Traffic Simulation ===

@dataclass
class Packet:
    timestamp: float
    size: int
    direction: str # 'up', 'down'

class TrafficGenerator:
    """Generates synthetic traces for robotic actions."""
    
    def generate_trace(self, action: str, duration: float = 2.0) -> List[Packet]:
        packets = []
        t = 0.0
        
        while t < duration:
            # Simulation Logic based on typical ROS 2 patterns
            # Control loop: 50Hz (20ms interval), small packets
            # Video stream: 30Hz (33ms interval), large packets (fragmented)
            
            if action == "IDLE":
                # Just heartbeat
                t += np.random.normal(0.1, 0.01) # 10Hz
                packets.append(Packet(t, 64, 'up'))
                
            elif action == "MOVE_J":
                # High frequency control updates
                dt = np.random.normal(0.01, 0.001) # 100Hz
                t += dt
                packets.append(Packet(t, 256 + int(np.random.normal(0, 10)), 'up'))
                if random.random() < 0.1: # Occasional feedback
                     packets.append(Packet(t+0.001, 512, 'down'))

            elif action == "PICK":
                # Burst of control + Vision spike
                dt = np.random.normal(0.02, 0.005) # 50Hz
                t += dt
                packets.append(Packet(t, 128, 'up'))
                
                # Camera burst
                if random.random() < 0.3: # ~15fps effective
                    packets.append(Packet(t + 0.002, 1400, 'up'))
                    packets.append(Packet(t + 0.003, 1400, 'up'))
                    packets.append(Packet(t + 0.004, 800, 'up'))

            elif action == "PLACE":
                # Precision control, lower freq vision
                dt = np.random.normal(0.02, 0.001) # 50Hz strict
                t += dt
                packets.append(Packet(t, 128, 'up'))
                packets.append(Packet(t + 0.005, 64, 'down'))
        
        return packets

# === 2. Defense Simulation ===

class PaddingDefense:
    """Simulates PaddingOnly defense."""
    def __init__(self, bucket_size: int = 512):
        self.bucket_size = bucket_size
        
    def apply(self, trace: List[Packet]) -> List[Packet]:
        defended_trace = []
        for p in trace:
            # Pad size to next multiple of bucket_size
            new_size = ((p.size + self.bucket_size - 1) // self.bucket_size) * self.bucket_size
            # Add small jitter to timestamp
            new_time = p.timestamp + random.uniform(0, 0.005)
            defended_trace.append(Packet(new_time, new_size, p.direction))
        return defended_trace

class FrontDefense:
    """Simulates FRONT (Dummy packets) defense with ROBOTICS ENHANCEMENT."""
    
    def __init__(self):
        self.total_delay = 0.0
        self.packet_count = 0

    def apply(self, trace: List[Packet]) -> List[Packet]:
        defended = list(trace) # Copy
        
        # ROBOTICS MODE: Strict CBR (Constant Bit Rate) Traffic Shaping
        # Instead of just adding dummies, we enforce a strict output schedule.
        # This aligns with a "Buffered Shaper" defense.
        
        target_rate = 500.0 # Hz (2ms interval)
        interval = 1.0 / target_rate
        
        if not trace: return []
        
        start_t = trace[0].timestamp
        end_t = trace[-1].timestamp
        
        # Queue of real packets
        real_queue = sorted(trace, key=lambda x: x.timestamp)
        queue_idx = 0
        
        defended_trace = []
        current_t = start_t
        
        # Run until buffer empty or time elapsed (with some tail)
        while current_t < end_t + 0.2 or queue_idx < len(real_queue):
            # 1. Enqueue packets that have "arrived" by current_t
            # (In simulation, we just check timestamp)
            
            # 2. Dequeue ONE packet for this slot
            # If real packet available (and arrived), send it (padded)
            # Else, send dummy
            
            has_real = False
            if queue_idx < len(real_queue):
                p = real_queue[queue_idx]
                if p.timestamp <= current_t:
                    # Packet is ready to send
                    defended_trace.append(Packet(current_t, 1400, p.direction))
                    
                    # TRACK LATENCY
                    delay = (current_t - p.timestamp) * 1000.0 # ms
                    self.total_delay += delay
                    self.packet_count += 1
                    
                    queue_idx += 1
                    has_real = True
            
            if not has_real:
                # Send Dummy
                defended_trace.append(Packet(current_t, 1400, 'up'))
                
            current_t += interval
            
        # Metrics: Latency Calculation
        # Map original timestamps to new timestamps for real packets
        # latencies = []
        # real_defended = [p for p in defended_trace if p.size > 0] # Simplified check (dummy logic handled by size/flag in Packet class?)
        # Actually in this script Packet doesn't have 'dummy' flag, we just infer.
        # But we know real packets are the ones we dequeued.
        
        # Let's just do a rough check: Time of last real packet out - Time of last real packet in?
        # Better: Average delay.
        # Since we re-ordered, we track indices?
        # Simplify: We know the defense enforces `current_t`.
        # The delay for a packet arriving at `p.timestamp` sent at `current_t` is `current_t - p.timestamp`.
        
        return defended_trace
    
    # NOTE: To avoid breaking the signature `apply(trace) -> trace` expected by the loop, 
    # we will implement a separate measure method or just print it inside apply if we want simplicity.
    # Let's stick to the signature `apply` but print inside is messy.
    # Let's modify the Loop in run_benchmark to handle the return if we change it.
    
    # Actually, simpler: Just modify the Loop to expect a tuple/extra info only if we want specific reporting.
    # Or just print it inside 'apply' for this ad-hoc request.
    
    pass 


# === 3. Feature Extraction (The "Signal Processing" Approach) ===

def extract_features(trace: List[Packet]) -> List[float]:
    """
    Extracts statistical features from a trace:
    - Packet Size: Mean, Std, Max, Min
    - Inter-Arrival Time (IAT): Mean, Std, Min
    - Throughput (bytes/sec)
    - Burstiness (Count of >1000 byte packets)
    """
    if not trace: return [0]*8
    
    sizes = [p.size for p in trace]
    timestamps = [p.timestamp for p in trace]
    iats = np.diff(timestamps) if len(timestamps) > 1 else [0]
    
    features = [
        np.mean(sizes),
        np.std(sizes),
        np.max(sizes),
        np.min(sizes),
        np.mean(iats),
        np.std(iats),
        sum(sizes) / (timestamps[-1] - timestamps[0] + 1e-9), # Throughput
        sum(1 for s in sizes if s > 1000) # Burst count
    ]
    return features

# === 4. Benchmark Runner ===

def run_benchmark():
    print("="*60)
    print("   TensorGuard: Robot Traffic Fingerprinting Benchmark")
    print("="*60)
    
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not found. Install it to run benchmark.")
        print("pip install scikit-learn")
        return

    gen = TrafficGenerator()
    actions = ["IDLE", "MOVE_J", "PICK", "PLACE"]
    SAMPLES_PER_ACTION = 50
    
    print(f"Generating {SAMPLES_PER_ACTION} traces per action ({len(actions)} actions)...")
    
    # 1. Generate Dataset (Baseline)
    X = []
    y = []
    
    raw_traces = [] # Keep for defense step
    
    for action in actions:
        for _ in range(SAMPLES_PER_ACTION):
            trace = gen.generate_trace(action)
            raw_traces.append((action, trace))
            feats = extract_features(trace)
            X.append(feats)
            y.append(action)
            
    # 2. Train baseline classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    baseline_acc = accuracy_score(y_test, preds)
    
    print(f"\n[BASELINE] Attack Accuracy (No Defense): {baseline_acc*100:.1f}%")
    print("-" * 30)
    print(classification_report(y_test, preds))
    
    # 3. Apply Padding Defense
    pad_defense = PaddingDefense(bucket_size=512)
    
    X_def = []
    y_def = []
    
    for action, trace in raw_traces:
        def_trace = pad_defense.apply(trace)
        feats = extract_features(def_trace)
        X_def.append(feats)
        y_def.append(action)
        
    X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_def, y_def, test_size=0.3, random_state=42)
    
    # Retrain attacker on DEFENDED traffic (Adaptive Attacker)
    # The paper highlights that attackers adapt.
    clf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_def.fit(X_train_def, y_train_def)
    
    preds_def = clf_def.predict(X_test_def)
    def_acc = accuracy_score(y_test_def, preds_def)
    
    print(f"\n[DEFENSE: PADDING] Attack Accuracy: {def_acc*100:.1f}%")
    drop = baseline_acc - def_acc
    print(f"Privacy Gain: +{drop*100:.1f}% reduction in identification rate.")
    
    # 4. Apply FRONT Defense (Noise)
    front_defense = FrontDefense()
    
    X_front = []
    y_front = []
    
    for action, trace in raw_traces:
        def_trace = front_defense.apply(trace)
        feats = extract_features(def_trace)
        X_front.append(feats)
        y_front.append(action)
        
    X_train_front, X_test_front, y_train_front, y_test_front = train_test_split(X_front, y_front, test_size=0.3, random_state=42)
    
    clf_front = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_front.fit(X_train_front, y_train_front)
    
    preds_front = clf_front.predict(X_test_front)
    front_acc = accuracy_score(y_test_front, preds_front)
    
    print(f"\n[DEFENSE: FRONT/CBR] Attack Accuracy: {front_acc*100:.1f}%")
    drop_front = baseline_acc - front_acc
    print(f"Privacy Gain: +{drop_front*100:.1f}% reduction in identification rate.")
    
    avg_latency = front_defense.total_delay / max(front_defense.packet_count, 1)
    print(f"Average Network Latency Overhead: {avg_latency:.2f} ms")

    
    print("\nSummary:")
    print(f"Baseline: {baseline_acc:.2%}")
    print(f"Padding:  {def_acc:.2%}")
    print(f"FRONT:    {front_acc:.2%}")

if __name__ == "__main__":
    run_benchmark()
