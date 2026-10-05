# Apply the segment rules to a saved ensemble prediction file.
#   usage: python finish_blend.py submissions/blend_hc.csv blend
#   writes submissions/<tag>_rules.csv
import sys
import pandas as pd
from rules import apply_rules

src, tag = sys.argv[1], sys.argv[2]
ss = pd.read_csv("data/sample_submission.csv")
te = pd.read_csv("data/test.csv").set_index("id").loc[ss.id].reset_index()
b = pd.read_csv(src)
assert b.shape == (len(ss), 2) and (b.id.values == ss.id.values).all() and b.Will_Buy_EV.notna().all()
out = ss[["id"]].copy()
out["Will_Buy_EV"] = apply_rules(b.Will_Buy_EV.values, te)
out.to_csv(f"submissions/{tag}_rules.csv", index=False)
print("saved", f"submissions/{tag}_rules.csv")
