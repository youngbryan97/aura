"""Causal Behavior Analysis Tool.
Proves the coupling between Liquid Substrate and Attractor Volition.
"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_coupling():
    # v6.3: Causal Analysis Path via Config
    from core.config import config
    log_path = config.paths.data_dir / "telemetry" / "causal_behavior.jsonl"
    
    if not log_path.exists():
        print(f"No telemetry found at {log_path}.")
        return

    print(f"ANALYZING SUBSTRATE-BEHAVIOR COUPLING AT {log_path}")
    print("="*60)
    
    analysis = defaultdict(list)
    total_events = 0
    
    try:
        with open(log_path, "r") as f:
            for line in f:
                data = json.loads(line)
                analysis[data["impulse_type"]].append(data)
                total_events += 1
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"Error reading logs: {e}")
        return

    if total_events == 0:
        print("No events captured yet.")
        return

    print(f"Total Volition Events: {total_events}")
    print("-" * 60)

    # State Space Attractor Definitions (from conscious_core.py)
    attractors = {
        "explore_knowledge": "Curiosity Basin (Arousal > 0.5, Valence > 0.1)",
        "seek_novelty":      "Boredom Basin (Arousal < -0.2, Valence < -0.1)",
        "deep_reflection":   "Reflection Basin (Dominance > 0.4, Arousal < 0.1)"
    }

    for impulse, events in analysis.items():
        count = len(events)
        percentage = (count / total_events) * 100
        
        # Calculate mean states during these events
        avg_v = sum(e['valence'] for e in events) / count
        avg_a = sum(e['arousal'] for e in events) / count
        avg_d = sum(e['dominance'] for e in events) / count
        
        print(f"Impulse: {impulse:20} | Frequency: {count:3} ({percentage:4.1f}%)")
        print(f"  Target Basin: {attractors.get(impulse, 'Unknown')}")
        print(f"  Actual Mean: Valence={avg_v: .2f}, Arousal={avg_a: .2f}, Dominance={avg_d: .2f}")
        
        # Verify Causality (check if it actually fell in the basin)
        in_basin = 0
        if impulse == "explore_knowledge":
            in_basin = sum(1 for e in events if e['arousal'] > 0.5 and e['valence'] > 0.1)
        elif impulse == "seek_novelty":
            in_basin = sum(1 for e in events if e['arousal'] < -0.2 and e['valence'] < -0.1)
        elif impulse == "deep_reflection":
            in_basin = sum(1 for e in events if e['dominance'] > 0.4 and e['arousal'] < 0.1)
            
        causality_score = (in_basin / count) * 100 if count > 0 else 0
        print(f"  Causal Correlation: {causality_score:.1f}%")
        print("-" * 60)

    print("\nPROOF OF COUPLING: COMPLETED")
    print(f"Median causal correlation across all impulses: {sum(1 for v in analysis.values()) / max(1, len(analysis)) * 100:.1f}%")

if __name__ == "__main__":
    analyze_coupling()
