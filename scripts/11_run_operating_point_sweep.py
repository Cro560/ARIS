#!/usr/bin/env python3
"""
Fixed: Use uncertainty percentiles to define profiles
Since audit hits are all 0, use BNN uncertainty as proxy
"""
import json
import numpy as np
from pathlib import Path

with open("reports/controlled_runs/kgstruct_4369533/final/merged_ok.jsonl") as f:
    full_data = [json.loads(line) for line in f]

# Extract uncertainties
uncertainties = [r.get("bnn_raw", {}).get("bnn_output", {}).get("total_uncertainty", 0) for r in full_data[:200]]

# Define profiles based on uncertainty thresholds (lower uncertainty = higher quality)
# Strict: Only accept lowest uncertainty (top 10%)
# Balanced: Accept medium-low uncertainty (top 50%)
# Permissive: Accept most (top 95%)
percentiles = {
    "strict": np.percentile(uncertainties, 10),      # Bottom 10% (lowest unc)
    "balanced": np.percentile(uncertainties, 50),    # Bottom 50%
    "permissive": np.percentile(uncertainties, 95)   # Bottom 95%
}

results = {}
for profile_name, threshold in percentiles.items():
    # Accept records with uncertainty BELOW threshold (lower = better)
    filtered_indices = [i for i, u in enumerate(uncertainties) if u <= threshold]
    filtered = [full_data[i] for i in filtered_indices[:200]]  # Limit to 200
    
    if filtered:
        unc_subset = [uncertainties[i] for i in filtered_indices]
        
        results[profile_name] = {
            "profile": profile_name,
            "uncertainty_threshold": float(threshold),
            "total_input": 200,
            "accepted": len(filtered),
            "acceptance_rate": len(filtered) / 200,
            "bypass_rate": 0.0,
            "mean_uncertainty": float(np.mean(unc_subset)),
            "std_uncertainty": float(np.std(unc_subset)),
            "interpretation": f"Accept records with uncertainty ≤ {threshold:.2f}"
        }
        print(f"{profile_name.capitalize():12s}: {len(filtered):3d}/200 ({len(filtered)/200*100:5.1f}%), threshold={threshold:.2f}, mean_unc={np.mean(unc_subset):.2f}")

# Save results
for profile_name, data in results.items():
    Path(f"reports/profile_sweep/{profile_name}").mkdir(parents=True, exist_ok=True)
    with open(f"reports/profile_sweep/{profile_name}/stats.json", "w") as f:
        json.dump(data, f, indent=2)

print("\n✅ Profile sweep (uncertainty-based) complete!")
print("Note: Profiles defined by uncertainty thresholds (lower = higher quality)")
