# Builds the technical report PDF: report/EV_Purchase_Prediction_Technical_Report.pdf
#   usage: .venv/bin/python report/build_report.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether, NextPageTemplate)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(HERE, "EV_Purchase_Prediction_Technical_Report.pdf")

# ---- palette (reference palette, light mode) -------------------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e5e4df"
BLUE, BLUE_D, BLUE_L, RED, NEUTRAL = "#2a78d6", "#1c5cab", "#cde2fb", "#e34948", "#f0efec"

# ---- fonts -------------------------------------------------------------------------
SUP = "/System/Library/Fonts/Supplemental"
if os.path.exists(f"{SUP}/Arial.ttf"):                       # macOS
    FONT_FILES = {"": "Arial.ttf", "-Bold": "Arial Bold.ttf", "-Italic": "Arial Italic.ttf",
                  "-BoldItalic": "Arial Bold Italic.ttf"}
    FONT_DIR, MPL_FONT = SUP, "Arial"
else:                                                       # elsewhere: fonts bundled with matplotlib
    FONT_DIR, MPL_FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf"), "DejaVu Sans"
    FONT_FILES = {"": "DejaVuSans.ttf", "-Bold": "DejaVuSans-Bold.ttf", "-Italic": "DejaVuSans-Oblique.ttf",
                  "-BoldItalic": "DejaVuSans-BoldOblique.ttf"}
for suffix, fname in FONT_FILES.items():
    pdfmetrics.registerFont(TTFont(f"Arial{suffix}", os.path.join(FONT_DIR, fname)))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")
plt.rcParams.update({"font.family": MPL_FONT, "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})

# ======================================================================================
# Figures
# ======================================================================================
def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(7.2, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    boxes = [
        (2, 22, "Training data", "668,665 labelled rows\n13 input columns"),
        (27, 22, "Feature engineering", "cross-fitted encodings,\nwindows, digits"),
        (52, 22, "Candidate models", "~70 tree & neural\nmodels (own + external)"),
        (77, 22, "Out-of-fold scores", "10-fold CV: each row\nscored out-of-fold"),
        (77, 2, "Ensemble", "rank-space hill climbing,\nweights from OOF only"),
        (52, 2, "Segment rules", "4 label-pure segments\npushed to the extremes"),
        (27, 2, "Final predictions", "286,571 test rows\nranked by propensity"),
    ]
    for x, y, title, sub in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 21, 15, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#f7f7f5", ec=GRID, lw=1))
        ax.text(x + 10.5, y + 10.6, title, ha="center", va="center", fontsize=8.6, weight="bold", color=INK)
        ax.text(x + 10.5, y + 5.0, sub, ha="center", va="center", fontsize=6.6, color=INK2, linespacing=1.35)
    arrow = dict(arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=10)
    for x0, x1 in [(23.5, 26.5), (48.5, 51.5), (73.5, 76.5)]:
        ax.annotate("", xy=(x1, 29.5), xytext=(x0, 29.5), arrowprops=arrow)
    ax.annotate("", xy=(87.5, 17.8), xytext=(87.5, 21.5), arrowprops=arrow)      # OOF scores -> ensemble
    ax.annotate("", xy=(73.5, 9.5), xytext=(76.5, 9.5), arrowprops=arrow)        # ensemble -> segment rules
    ax.annotate("", xy=(48.5, 9.5), xytext=(51.5, 9.5), arrowprops=arrow)        # rules -> final predictions
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

def fig_feature_checks(path):
    items = [("Dropping 4 low-signal columns", -47), ("Nearest-neighbour labels (reference data)", -18),
             ("All reference-data features together", -15), ("Expected range-anxiety features", -6),
             ("Reference-data target rates", -5), ("Business-rule purchase score", -2),
             ("Score + range-anxiety features", -2), ("Reference-data frequency / lift", -1)]
    labels = [a for a, _ in items][::-1]; vals = [b for _, b in items][::-1]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.barh(labels, vals, color=RED, height=0.56, edgecolor="white", linewidth=1)
    ax.axvline(0, color=INK2, lw=1)
    for i, v in enumerate(vals):
        ax.text(v - 0.8, i, f"{v:+d}", va="center", ha="right", fontsize=8, color=INK2)
    ax.set_xlim(-56, 4); ax.set_xlabel("Change in 5-fold CV ROC AUC vs baseline ($\\times 10^{-5}$)   ·   baseline = 0.94586")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0, labelsize=8.4)
    ax.spines["left"].set_visible(False)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

STAGES = ["Starting\nnotebook", "First\nensemble", "+ own\nmodels", "+ larger\npool", "+ new engines\n+ rules", "+ extra\nseeds"]
OOF = [0.94597, 0.94642, 0.94644, 0.94645, 0.94648, 0.94650]
PRIV = [0.94509, 0.94542, 0.94542, 0.94544, 0.94545, 0.94546]

def fig_progression(path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, vals, title in [(axes[0], OOF, "Validation ROC AUC (10-fold out-of-fold)"),
                            (axes[1], PRIV, "Final hold-out ROC AUC (80% of test set)")]:
        x = range(len(vals))
        ax.plot(x, vals, color=BLUE, lw=2, marker="o", ms=5, mfc=BLUE, mec="white", mew=1.2, zorder=3)
        ax.set_xticks(list(x)); ax.set_xticklabels(STAGES, fontsize=6.8)
        ax.set_title(title, fontsize=8.8, color=INK, loc="left", pad=8)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
        ax.annotate(f"{vals[0]:.5f}", (0, vals[0]), textcoords="offset points", xytext=(6, -2), fontsize=7.6, color=INK2)
        ax.annotate(f"{vals[-1]:.5f}", (len(vals) - 1, vals[-1]), textcoords="offset points", xytext=(-4, 7),
                    fontsize=7.6, color=INK, ha="right", weight="bold")
        lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.18
        ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

def fig_weights(path):
    items = [("Gradient boosting with hierarchical income encodings\n(XGBoost, 3 seeds; 2 trained by me on cloud GPU)", 0.375),
             ("XGBoost on triple target-encoded features", 0.125),
             ("Tuned LightGBM on encoded features", 0.125),
             ("LightGBM with lift and window features", 0.125),
             ("RealMLP neural network (10-fold, 3 seeds)", 0.125),
             ("Residual-stacked gradient boosting", 0.125)]
    labels = [a for a, _ in items][::-1]; vals = [b for _, b in items][::-1]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.barh(labels, vals, color=BLUE, height=0.56, edgecolor="white", linewidth=1)
    for i, v in enumerate(vals):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 0.44); ax.set_xlabel("Ensemble weight (rank space, sums to 1)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0, labelsize=7.8)
    ax.spines["left"].set_visible(False)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

fig_pipeline(f"{FIG}/pipeline.png"); fig_feature_checks(f"{FIG}/feature_checks.png")
fig_progression(f"{FIG}/progression.png"); fig_weights(f"{FIG}/weights.png")

# ======================================================================================
# Document
# ======================================================================================
W, H = A4
M = 20 * mm
st = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=27, leading=32, textColor=INK, spaceAfter=8),
    "subtitle": ParagraphStyle("subtitle", fontName="Arial", fontSize=13, leading=18, textColor=INK2, spaceAfter=16),
    "meta": ParagraphStyle("meta", fontName="Arial", fontSize=9.5, leading=13, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=14.5, leading=19, textColor=INK, spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11, leading=15, textColor=INK, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Arial", fontSize=9.6, leading=14, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Arial", fontSize=9.6, leading=14, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=2),
    "caption": ParagraphStyle("caption", fontName="Arial-Italic", fontSize=8.3, leading=11, textColor=INK2, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.3, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.3, leading=11, textColor=INK),
    "kpi_v": ParagraphStyle("kpi_v", fontName="Arial-Bold", fontSize=17, leading=21, textColor=INK, alignment=TA_LEFT),
    "kpi_l": ParagraphStyle("kpi_l", fontName="Arial", fontSize=8.3, leading=11, textColor=INK2),
}
P = lambda t, s="body": Paragraph(t, st[s])
def bullets(items):
    return [Paragraph(t, st["bullet"], bulletText="•") for t in items]
def fig(path, width_mm, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize(); w = width_mm * mm
    return KeepTogether([Image(path, width=w, height=w * ih / iw), P(caption, "caption")])
def table(rows, col_mm, header=True, zebra=True, align_right_cols=()):
    data = [[Paragraph(str(c), st["cellb"] if (header and i == 0) else st["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[c * mm for c in col_mm], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID))]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f2ef")),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#c9c7c0"))]
    t.setStyle(TableStyle(style)); return t

def kpis(items):
    cells = [[P(v, "kpi_v"), ] for v, _ in items]
    row_v = [P(v, "kpi_v") for v, _ in items]; row_l = [P(l, "kpi_l") for _, l in items]
    t = Table([row_v, row_l], colWidths=[(W - 2 * M) / len(items)] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f7f5")),
                           ("LINEAFTER", (0, 0), (-2, -1), 0.6, colors.HexColor(GRID)),
                           ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t

story = []
# ---------------------------------------------------------------- title page
story += [Spacer(1, 18 * mm),
          P("Data-Driven Targeting of Likely Electric Vehicle Buyers", "title"),
          P("An Ensemble Approach with Gradient-Boosted and Neural Network Models", "subtitle"),
          P("Technical report · Artur Markosyan · September 2026", "meta"), Spacer(1, 10 * mm),
          kpis([("Top 10%", "final rank 307 of ~3,300 teams"),
                ("0.9455", "ROC AUC on the final hold-out set"),
                ("0.9465", "ROC AUC in 10-fold cross-validation"),
                ("~70", "candidate models evaluated")]),
          Spacer(1, 9 * mm),
          P("Summary", "h1"),
          P("The goal was to rank people by how likely they are to buy an electric vehicle, so that incentives and "
            "marketing can be focused on the most likely buyers. The model is an ensemble of gradient-boosted "
            "decision trees and neural networks trained on 668,665 labelled records with 13 inputs (age, income, "
            "commute, charging access, environmental concern, subsidy, range anxiety and others). Every modelling "
            "decision was made on out-of-fold predictions from 10-fold cross-validation, never on the leaderboard."),
          P("The work was benchmarked in an open machine-learning competition with about 3,300 "
            "participating teams. The final solution scored a ROC AUC of 0.94546 on the hidden final test set "
            "(rank 307, top 10%), up from 0.94509 for the starting single-notebook model. A notable finding: "
            "submissions tuned to the public leaderboard scored higher there (up to 0.94661) but lower on the "
            "hidden final set than the cross-validated ensemble, which confirmed the validation-first approach."),
          P("What the report covers", "h2")]
story += bullets(["How the problem and evaluation were set up (sections 1–2)",
                  "Features and models, and how the ensemble combines them (sections 3–5)",
                  "Every idea that was tested, with the result and the decision to accept or reject it (section 6)",
                  "Results, lessons and limitations (sections 7–8); credits and reproducibility (appendix)"])
story.append(PageBreak())

# ---------------------------------------------------------------- 1. problem
story += [P("1. Problem and data", "h1"),
          P("<b>Task.</b> Binary classification: for each person, predict the probability that they will buy an "
            "electric vehicle (<i>Will_Buy_EV</i>, yes/no). Only 17.5% of training records are buyers, so the "
            "classes are imbalanced."),
          P("<b>Data.</b> 668,665 labelled training records and 286,571 unlabelled test records. A smaller 10,000-row "
            "reference dataset from the same domain was also available and was tested as an extra source of features."),
          table([["Group", "Input columns"],
                 ["Demographics", "Age, Gender, Annual income (USD)"],
                 ["Mobility", "Daily commute (km), Number of cars owned, Current car type, City type"],
                 ["Charging access", "Charging stations near home, Charging stations near work, Home charging possible"],
                 ["Attitude & policy", "Environmental concern level (1–5), Subsidy available, Range anxiety level"]],
                [38, 132]),
          Spacer(1, 8),
          P("<b>Metric.</b> ROC AUC: the probability that a randomly chosen buyer is ranked above a randomly chosen "
            "non-buyer (0.5 = random, 1.0 = perfect). AUC depends only on the <i>order</i> of predictions, which "
            "is used heavily later: models are combined by their ranks, and post-processing only moves rows up or "
            "down the ranking."),
          P("<b>Evaluation.</b> Test predictions were scored on two hidden splits: a public 20% shown during the "
            "competition, and a final 80% revealed only at the end, which determines the rank. The public split "
            "is small enough (about 57,000 rows) that differences of 0.0001 AUC are within noise."),
          P("2. Validation strategy", "h1"),
          P("Every model was evaluated with <b>10-fold stratified cross-validation</b> (fixed seed, identical folds "
            "across all of my models). For each fold, the model trains on 90% of the rows and predicts the held-out "
            "10%. Stitching the held-out predictions together gives an <b>out-of-fold (OOF)</b> prediction for every "
            "training row from a model that never saw that row's label. The OOF AUC over 668,665 rows is a far more "
            "stable estimate than the 57,000-row public split, and the OOF predictions are also the inputs for "
            "choosing ensemble weights."),
          P("Safeguards", "h2")]
story += bullets([
    "<b>Leakage-safe encodings.</b> Any feature computed from labels (target encodings, window rates) is "
    "cross-fitted: training rows get values from an inner 5-fold split, so no row ever sees its own label.",
    "<b>Honest ensemble estimates.</b> Ensemble weights were fitted on one half of the rows and scored on the "
    "other half (4 random splits, both directions), so the reported ensemble gain is not optimistic.",
    "<b>Nested validation for post-processing.</b> Rules and corrections were accepted only if they improved "
    "AUC on every fold of an independent 5-fold split.",
    "<b>Leakage screening of inputs.</b> One external candidate reported an OOF AUC of exactly 1.000, which is "
    "impossible for an honest out-of-fold file; it was excluded.",
    "<b>One decision rule.</b> Nothing was accepted because of a public-leaderboard score."])
story.append(PageBreak())

# ---------------------------------------------------------------- 3. features
story += [P("3. Feature engineering", "h1"),
          P("Tree models already handle raw columns well, so most of the gain comes from encodings that expose the "
            "<i>exact</i> value of income and commute. Analysis showed that individual income values carry real, "
            "value-specific purchase effects of about ±4 percentage points beyond what a smooth income curve explains, "
            "and the features below are different ways of letting a model use that information without overfitting."),
          table([["Feature family", "What it captures", "How it is computed"],
                 ["Raw inputs", "The 13 original columns", "Categoricals as native categories"],
                 ["Digit features", "Fine-grained structure of income and commute values",
                  "Individual digits (ones, tens, …) of income and commute"],
                 ["Frequency encoding", "How common each income / commute value is", "Count of the value in train + test"],
                 ["Target encoding", "Purchase rate for each value of a column",
                  "Smoothed mean label per value, cross-fitted; several smoothing strengths (m = 5, 20, 100, 200)"],
                 ["Multi-resolution keys", "Purchase rate at coarser income / commute levels",
                  "Target encoding of income rounded to 10 / 100 / 1,000 and commute to 1 / 5 km"],
                 ["Window encodings", "Purchase rate among people with <i>similar</i> income or commute",
                  "Smoothed rate within ±2 … ±1,000 USD and ±0.1 … ±10 km, via prefix sums; cross-fitted"],
                 ["Interaction encodings", "Combined effects (e.g. subsidy × range anxiety, income band × subsidy)",
                  "Target encoding of 12 column combinations"]],
                [36, 58, 76]),
          Spacer(1, 6),
          P("Feature ideas tested and rejected", "h2"),
          P("Before building full models, candidate feature sets were screened with a fast 5-fold LightGBM against a "
            "fixed baseline (AUC 0.94586). Every idea below lowered the score and was rejected. The clearest lesson: "
            "removing columns that look irrelevant (age, gender, number of cars, car type) cost 0.00047, because "
            "they still carry signal the trees use."),
          fig(f"{FIG}/feature_checks.png", 168, "Figure 1. Fast screening of feature ideas (5-fold LightGBM). All "
              "ideas lowered validation AUC relative to the baseline and were rejected.")]
story.append(PageBreak())

# ---------------------------------------------------------------- 4. models
story += [P("4. Models", "h1"),
          P("Two model families were used. <b>Gradient-boosted decision trees</b> (LightGBM, XGBoost, CatBoost) "
            "build thousands of small trees in sequence, each correcting the errors of the previous ones; they are "
            "the strongest single models for tabular data. <b>Neural networks</b> (multi-layer perceptrons, "
            "RealMLP, TabM) learn smooth functions of the inputs; they are slightly weaker alone but make "
            "<i>different</i> errors, which is what an ensemble needs."),
          P("Models I trained", "h2"),
          table([["Model", "What is distinctive", "OOF AUC", "Outcome"],
                 ["LightGBM (starting notebook, 3 seeds)", "Nested target encodings, digits, frequencies", "0.94611", "Pool member"],
                 ["CatBoost (starting notebook)", "Native categorical handling", "0.94602", "Pool member"],
                 ["LightGBM + window encodings", "Similar-income / similar-commute purchase rates", "0.94612", "Pool member"],
                 ["LightGBM + interaction encodings", "12 column-combination encodings", "0.94606", "Pool member"],
                 ["Depth-2 LightGBM", "Near-additive model (pairwise interactions only)", "0.94603", "Pool member"],
                 ["MLP + periodic embeddings (3 seeds)", "Learned sin/cos embeddings of income & commute", "0.94593", "Pool member"],
                 ["MLP + interaction encodings (3 seeds)", "Neural view of the combination features", "0.94596", "Pool member"],
                 ["RealMLP (re-implemented, 3 seeds)", "Modern tabular network, run on a local GPU", "≈0.9460 / seed", "Pool member"],
                 ["XGBoost, hierarchical encodings (2 new seeds)", "Re-runs of the strongest external model on cloud GPU",
                  "0.94640 (3-seed avg)", "<b>Top ensemble member</b>"],
                 ["XGBoost, window encodings (1 new seed)", "Re-run of an external model on cloud GPU", "0.94628 (2-seed avg)", "Pool member"],
                 ["LightGBM with pseudo-labels", "Test rows added with ensemble probabilities as soft labels", "0.94601", "Rejected"],
                 ["TabM (cloud GPU)", "Parameter-efficient ensemble network", "0.94015", "Rejected"]],
                [52, 62, 28, 28]),
          Spacer(1, 6),
          P("External candidate models", "h2"),
          P("Other competition participants published out-of-fold and test predictions for their models. Because "
            "out-of-fold predictions are honest by construction, these could be evaluated and weighted with exactly "
            "the same procedure as my own models. About 60 external candidates from 15 sources were added to the "
            "pool (individual OOF AUC 0.938–0.9463), including XGBoost and LightGBM models with different encoding "
            "schemes, several RealMLP variants, CatBoost models, and simpler linear, spline (GAM) and random-forest "
            "models. Sources are credited in the appendix. Files were aligned by row id and checked for leakage "
            "before use.")]
story.append(PageBreak())

story += [P("5. How the ensemble works", "h1"),
          fig(f"{FIG}/pipeline.png", 168, "Figure 2. End-to-end pipeline. All selection decisions use out-of-fold "
              "predictions; the test set is only scored at the end."),
          P("<b>Rank space.</b> Models output probabilities on different scales (some even raw scores). Because "
            "AUC depends only on order, every prediction is converted to its rank (0–1) before combining, which "
            "makes all models directly comparable."),
          P("<b>Hill climbing (Caruana ensemble selection).</b> Start from the best single model; repeatedly add "
            "(with replacement) whichever model most improves the OOF AUC of the running average; stop when no "
            "addition helps. Weights are therefore multiples of 1/8 and the method automatically ignores redundant "
            "models. With ~70 highly correlated candidates (pairwise correlation 0.99+), this simple, low-parameter "
            "method generalised better than every more flexible alternative tested:"),
          table([["Combiner", "OOF AUC", "Held-out gain over best single model", "Decision"],
                 ["Hill climbing (rank space)", "0.94650", "+0.00008 (±0.00000)", "<b>Used</b>"],
                 ["Logistic-regression stacking", "0.94649", "+0.00005 (±0.00002)", "Rejected"],
                 ["Non-negative logistic stacking", "0.94641–0.94643", "+0.00008–0.00009 (earlier pool)", "Rejected"],
                 ["LightGBM level-2 stacker", "0.94634", "below logistic stacking", "Rejected"],
                 ["Equal-weight average of all models", "—", "−0.00007", "Rejected"]],
                [55, 30, 55, 30]),
          Spacer(1, 6),
          fig(f"{FIG}/weights.png", 168, "Figure 3. Final ensemble: six members selected by hill climbing from ~70 "
              "candidates. Selection is purely data-driven: my own models were competitive individually but were "
              "superseded by external models carrying the same information more strongly. The top member combines "
              "the original run with two seeds I trained on cloud GPU."),
          KeepTogether([P("Segment rules (post-processing)", "h2"),
          P("Four segments of the input space have a constant label across all 668,665 training rows. Tree models "
            "almost learn them, but averaging many models smooths the edges, so a few rows inside a "
            "\"never buys\" segment were ranked above 70% of the data. Enforcing the segments moves those rows to "
            "the very top or bottom of the ranking while keeping the order inside each segment."),
          table([["Segment", "Training rows", "Buy rate", "Test rows"],
                 ["Income ≥ 170,537 USD", "393", "100%", "156"],
                 ["Income 31,004–41,970 USD", "1,257", "0%", "494"],
                 ["Daily commute ≥ 83 km", "186", "0%", "81"],
                 ["Income exactly 30,000, no subsidy, and (concern level 1 or medium/high anxiety)", "7,157", "0%", "3,002"]],
                [92, 26, 22, 30]),
          Spacer(1, 4),
          P("Validated on an independent 5-fold split, the rules improved AUC on <b>5 of 5 folds</b> (mean "
            "+0.0000056). The gain is small but consistent, so the rules were accepted.")])]

# ---------------------------------------------------------------- 6. experiment log
story += [P("6. Experiment log: what was accepted and what was rejected", "h1"),
          P("Every idea was judged by the same rule: does it improve out-of-fold AUC (and, for post-processing, on "
            "every fold)? Changes are relative to the configuration at the time of the test."),
          P("Accepted", "h2"),
          table([["Idea", "Effect", "Why it worked"],
                 ["Rank-space hill-climbing ensemble", "+0.00045 OOF over the starting notebook (0.94597 → 0.94642)",
                  "Combines models that make different errors"],
                 ["Expanding the candidate pool with external OOF predictions", "+0.00006 OOF (0.94644 → 0.94650 with seeds)",
                  "New feature views and model types add diversity"],
                 ["Within-fold rank normalisation for neural nets", "+0.00008 on the neural-net member",
                  "Neural-net calibration drifted between folds"],
                 ["Seed averaging of the top members (cloud GPU)", "+0.00009 on the top member, +0.00002 on the ensemble",
                  "Reduces model variance where the weight is largest"],
                 ["Label-pure segment rules", "+0.0000056 OOF, 5 of 5 folds",
                  "Restores hard edges that averaging smooths away"]],
                [55, 62, 53]),
          Spacer(1, 6),
          P("Rejected", "h2"),
          table([["Idea", "Result", "Reason"],
                 ["Business-rule purchase score as a feature", "−0.00002 CV", "Trees already learn the same rule"],
                 ["Expected range-anxiety features", "−0.00006 CV", "Redundant with existing columns"],
                 ["Reference-dataset frequency, lift and target-rate features", "−0.00001 to −0.00005 CV",
                  "Reference data too small to add signal"],
                 ["Nearest-neighbour labels from the reference dataset", "−0.00018 CV", "Noisy; adds variance"],
                 ["Dropping 4 low-signal columns", "−0.00047 CV", "Those columns still carry signal"],
                 ["Linking each record to a specific reference row", "≤ +0.00003 (measured by others)",
                  "Too small to justify the complexity"],
                 ["Hypothesis: labels follow a tree fitted on the reference data", "Not supported by the data",
                  "Same segments are not pure there (47%, 10%, 15% buy rates)"],
                 ["Pseudo-labelling (soft labels on test rows)", "0.94601 vs 0.94612 without",
                  "Worse on 8 of 10 folds"],
                 ["TabM network (cloud GPU)", "0.94015; lowers the ensemble at every weight", "Too weak in this setting"],
                 ["Residual target encoding on the ensemble", "0.000000 on all folds", "Ensemble already captures per-value effects"],
                 ["Level-2 LightGBM / non-negative logistic stacking", "0.94634 / 0.94641–0.94643",
                  "Overfit with highly correlated inputs"],
                 ["Blending in files tuned to the public leaderboard", "Public 0.94661, final 0.94544",
                  "Higher public score, lower final score than the CV ensemble"]],
                [62, 50, 58])]

# ---------------------------------------------------------------- 7. results
story += [P("7. Results", "h1"),
          fig(f"{FIG}/progression.png", 170, "Figure 4. Validation AUC (left) and final hold-out AUC (right) across the "
              "main stages. Both improve together, which shows the cross-validation tracked the hidden test set."),
          table([["Submission", "OOF AUC", "Public (20%)", "Final (80%)"],
                 ["Starting notebook (single ensemble)", "0.94597", "0.94609", "0.94509"],
                 ["First rank-space ensemble", "0.94642", "0.94647", "0.94542"],
                 ["Larger candidate pool", "0.94645", "0.94649", "0.94544"],
                 ["+ new external models + segment rules", "0.94648", "0.94649", "0.94545"],
                 ["<b>Final ensemble (+ seed averaging)</b>", "<b>0.94650</b>", "0.94647", "<b>0.94546</b>"],
                 ["For comparison: leaderboard-tuned public file", "n/a", "0.94661", "0.94544"],
                 ["Competition winner", "n/a", "n/a", "0.94602"]],
                [78, 28, 32, 32]),
          Spacer(1, 6),
          P("<b>Final rank: 307 of about 3,300 teams (top 10%).</b> The final ensemble improved the hidden-test AUC "
            "by 0.00037 over the starting notebook. On the public split its score stayed around 0.94647–0.94649 "
            "while validation kept improving; the final split confirmed the validation ranking, not the public one."),
          P("8. Lessons and limitations", "h1")]
story += bullets([
    "<b>Validate on the largest honest sample available.</b> 668,665 out-of-fold predictions were a better guide "
    "than a 57,000-row public split; files tuned to the public split lost their advantage on the final set.",
    "<b>Diversity beats individual strength in ensembles.</b> Models with a different view of the data earned "
    "weight even when slightly weaker; strong but near-duplicate models earned none.",
    "<b>Simple combiners generalise best with correlated inputs.</b> Hill climbing beat logistic, "
    "non-negative and tree-based stackers.",
    "<b>Gains were small by the end.</b> After the first ensemble, further work added +0.00008 OOF in total. The "
    "winning solution (0.94602) was 0.0006 ahead, which suggests a structural insight about the data rather than "
    "more models.",
    "<b>Limitations.</b> The final ensemble relies largely on externally published models; my contribution is the "
    "validation and selection pipeline, my own candidate models, the extra seed runs and the segment rules. "
    "Compute was limited to a laptop plus free cloud GPUs (two concurrent sessions), which ruled out large "
    "hyper-parameter searches."])
story.append(Spacer(1, 6))

# ---------------------------------------------------------------- appendix
story += [P("Appendix A. Reproducibility", "h1"),
          table([["File", "Purpose"],
                 ["quick_checks.py", "Fast 5-fold screening of feature ideas against a fixed baseline"],
                 ["view_model.py", "10-fold LightGBM / XGBoost / CatBoost on configurable feature views (windows, interactions, depth)"],
                 ["mlp_model.py", "10-fold PyTorch MLP with optional periodic embeddings"],
                 ["realmlp_local.py", "Local re-implementation run of a public RealMLP architecture"],
                 ["pool.py", "Loads, aligns (by row id) and rank-normalises all OOF / test prediction pairs"],
                 ["blend.py", "Hill-climbing and logistic stacking with honest half-split estimates"],
                 ["stack_l2.py, blend_nn.py", "Level-2 LightGBM and non-negative logistic stackers (rejected)"],
                 ["rules.py, finish_blend.py", "Segment rules and final file assembly"],
                 ["Cloud-GPU notebooks", "Notebook forks run on cloud GPUs (TabM, extra seeds) and the orchestration script"]],
                [45, 125]),
          P("Appendix B. Credits", "h1"),
          P("External out-of-fold predictions and published model code were used from the following competition "
            "participants (usernames): megayak (feature-view libraries), najiama (OOF hub), lucifer19 "
            "(fusion components, segment rules write-up), heuljax (hierarchical-encoding XGBoost), blamerx and medvax "
            "(window-encoding XGBoost), yekenot (RealMLP), chinzorigtganbat (LightGBM, CatBoost, TabM), "
            "mizushimatoshihiko (LightGBM), tamerlanomralinov (fingerprint models), legtarrr (residual stack), "
            "dariushafshar (model library), marcmaldonado (row-level analysis), aryankaisth (HistGBM), "
            "evgendvorkin and kirill0212 (XGBoost / LightGBM baselines), udaken10 and stephentarter (CatBoost / "
            "HistGBM), and Chris Deotte (exploratory analysis of the reference data). The segment rules were found "
            "by the community and independently verified here.")]

# ---------------------------------------------------------------- page templates
def on_page(c, doc):
    c.saveState()
    c.setFont("Arial", 7.8); c.setFillColor(colors.HexColor(MUTED))
    c.drawString(M, 12 * mm, "Data-Driven Targeting of Likely Electric Vehicle Buyers · Technical report")
    c.drawRightString(W - M, 12 * mm, f"{doc.page}")
    c.setStrokeColor(colors.HexColor(GRID)); c.setLineWidth(0.5); c.line(M, 15 * mm, W - M, 15 * mm)
    c.restoreState()
def on_first(c, doc):
    c.saveState(); c.setFillColor(colors.HexColor(BLUE)); c.rect(0, H - 9 * mm, W, 9 * mm, stroke=0, fill=1); c.restoreState()

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=20 * mm,
                      title="Data-Driven Targeting of Likely Electric Vehicle Buyers: An Ensemble Approach with Gradient-Boosted and Neural Network Models", author="Artur Markosyan",
                      subject="Technical report: EV purchase propensity model")
frame = Frame(M, 20 * mm, W - 2 * M, H - M - 20 * mm, id="f")
doc.addPageTemplates([PageTemplate(id="first", frames=[frame], onPage=on_first),
                      PageTemplate(id="rest", frames=[frame], onPage=on_page)])
story.insert(0, NextPageTemplate("rest"))
doc.build(story)
print("wrote", OUT)
