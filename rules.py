# ===========================================================================
# Label-pure segment rules: four segments whose label is constant across all
# 668,665 training rows (found by the community, verified here fold by fold).
# Post-processing pushes each segment to the top (+1) or bottom (-1) of the
# ranking while keeping the order inside the segment.
# ===========================================================================
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def rule_masks(frame):
    income = frame["Annual_Income_USD"].to_numpy(np.float64)
    commute = frame["Daily_Commute_km"].to_numpy(np.float64)
    no_subsidy = frame["Subsidy_Available"].astype(str).to_numpy() == "No"
    concern_1 = frame["Environmental_Concern_Level"].to_numpy() == 1
    anxious = np.isin(frame["Range_Anxiety_Level"].astype(str).to_numpy(), ["Medium", "High"])
    return [
        ("income >= 170,537 -> always buys", income >= 170537, +1),
        ("income 31,004-41,970 -> never buys", (income >= 31004) & (income <= 41970), -1),
        ("commute >= 83 km -> never buys", commute >= 83, -1),
        ("30k, no subsidy, concern 1 or anxious", (income == 30000) & no_subsidy & (concern_1 | anxious), -1),
    ]

def apply_rules(score, frame):
    r = rankdata(score) / len(score)
    for _, mask, sign in rule_masks(frame):
        r[mask] += 2.0 * sign
    return rankdata(r) / len(r)
