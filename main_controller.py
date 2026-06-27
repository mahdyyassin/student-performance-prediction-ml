# main_controller


import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

import c306_data_understanding as du
import c306_data_preprocess as dp
from c306_modeling import compare_knn_and_nb, compare_tree_and_ensemble



# =========================================================
# HELPERS
# =========================================================
def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_list(title, items):
    print(f"\n{title}")
    if items:
        for i in items:
            print(f"  - {i}")
    else:
        print("  - None")


def print_distribution(y, title):
    print_section(title)
    dist = y.value_counts(normalize=True) * 100
    for k, v in dist.items():
        print(f"  Class {k}: {v:.2f}%")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df.columns = new_df.columns.str.strip().str.lower()
    new_df = new_df.replace("?", np.nan)
    new_df = new_df.replace(r"^\s*$", np.nan, regex=True)
    return new_df


def log_stage(name, df, count_outliers=True):
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    print(f"\n[{name}] shape = {df.shape}", flush=True)

    if count_outliers:
        num_cols = df.select_dtypes(include="number").columns

        outlier_mask = pd.DataFrame(False, index=df.index, columns=num_cols)

        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_mask[col] = (df[col] < lower) | (df[col] > upper)

        total_outliers = int(outlier_mask.any(axis=1).sum())

        print(f"Missing = {missing} | Duplicates = {duplicates} | Outliers = {total_outliers}", flush=True)
    else:
        print(f"Missing = {missing} | Duplicates = {duplicates}", flush=True)


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(r"C:\Users\mahdy\Downloads\dm proj\StudentPerformanceFactors.csv")

df.columns = df.columns.str.strip()

df = clean_dataset(df)

#transforming total scores (y) into categorical variable
passing_mark = 60
df['Pass_Fail'] = np.where(df['exam_score'] >= passing_mark, 'Pass', 'Fail')

target = "Pass_Fail"
if target not in df.columns:
    raise ValueError("Target column not found")

# ALWAYS split after
y = df[target].copy()
X = df.drop(columns=[target]).copy()


# =========================================================
# EDA + PREPROCESSING (PDF open across both)
# =========================================================
print_section("EDA REPORT")

with PdfPages(r"C:\Users\mahdy\Downloads\dm proj\Data Quality Report.pdf") as pdf:

    # ── EDA ──────────────────────────────────────────────────
    overview = du.data_overview(df, pdf)
    du.univariate_plot(df, pdf)
    du.relations_num(df, vars="all", chart_type="heatmap", pdf=pdf)
    du.correlation_num_ordinal(df, pdf)

    cats = df.select_dtypes(include=["object", "category"]).columns
    for i in range(len(cats) - 1):
        du.relations_cat(df, cats[i], cats[i + 1], pdf=pdf)

    du.relations_cat_num(df, "all", "all", "box", pdf)

    # ── PREPROCESSING ────────────────────────────────────────
    print_section("PREPROCESSING")

    numeric_vars, ordinal_vars, categorical_vars = du.classify_variables(df)

    X_clean = dp.delete_duplicates(X)
    log_stage("After delete_duplicates", X_clean)

    X_clean = dp.repair_missing(X_clean, method="median")
    log_stage("After repair_missing", X_clean)

    dp.print_outlier_placeholder(X_clean)

    # ── BEFORE: distributions snapshot ───────────────────────
    du.plot_distribution_matrix(X_clean, label="Before Outlier & Skewness Fix", pdf=pdf)

    X_clean = dp.repair_outliers(X_clean, detection="IQR", method="cap")
    log_stage("After repair_outliers", X_clean)

    X_clean = dp.delete_duplicates(X_clean)
    log_stage("After final deduplication", X_clean)

    dp.print_data_quality_summary(X_clean, "AFTER PREPROCESSING")

    X_clean = dp.fix_skewness(X_clean)
    log_stage("After fix_skewness", X_clean)

    # ── AFTER: distributions snapshot ────────────────────────
    du.plot_distribution_matrix(X_clean, label="After Outlier & Skewness Fix", pdf=pdf)

    X_clean = dp.scale_numeric(X_clean)
    log_stage("After scale_numeric", X_clean, count_outliers=False)

# ── rest of the script continues outside the with block ──
df_clean = pd.concat([X_clean, y.loc[X_clean.index]], axis=1)
drop_cols = ["exam_score"]
df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
# ... feature selection and modeling continue unchanged

# =========================================================
# FEATURE SELECTION
# =========================================================
print_section("FEATURE SELECTION")


# 1) FILTER
df_fs = dp.filter_selection(df_clean, target)
print("After filter selection:", df_fs.shape, flush=True)

# 2) COLLINEARITY (numeric only)
df_fs = dp.reduce_collinearity(df_fs, target)
print("After reduce collinearity:", df_fs.shape, flush=True)

# 3) ENCODING (after filter)
df_fs_encoded = dp.encode_categorical(df_fs, target_variable=target)

# convert bool → int
bool_cols = df_fs_encoded.select_dtypes(include="bool").columns
df_fs_encoded[bool_cols] = df_fs_encoded[bool_cols].astype(int)

print("After encoding:", df_fs_encoded.shape, flush=True)

# 4) WRAPPER (final feature selection)
df_final = dp.wrapper_feature_selection(
    df_fs_encoded,
    target_col=target,
    Encoding_flag=False,
    n_features=5,
    direction="forward",
    model_type="classification",
)

print_list("Wrapper Selected Features", list(df_final.columns))
print("Final shape:", df_final.shape, flush=True)


# =========================================================
# TRAIN / BALANCE / MODELING
# =========================================================
print_section("MODELING IS STARTING .....")

print_section("MODEL INPUT")

df_model = df_final.copy()

chosen_name = "df_final (Wrapper Selected Features output)"
print(f"Modeling dataframe source: {chosen_name}")
print(f"Shape: {df_model.shape}")
print(f"Columns used: {list(df_model.columns)}")

print_section("TRAIN & BALANCE")


if target not in df_model.columns:
    raise ValueError("Target column missing after PCA")

X = df_model.drop(columns=[target]).copy()
y = df_model[target].copy()

print_distribution(y, "Before Split")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

train_df = pd.concat([X_train, y_train], axis=1)
print_distribution(train_df[target], "Training Set Before Balancing")


# =========================================================
# MODEL COMPARISON
# =========================================================

results_knn_nb_before = compare_knn_and_nb(X_train, y_train, X_test, y_test)

results_knn_nb_after = compare_knn_and_nb(
    X_train,
    y_train,
    X_test,
    y_test,
    balance_fn=dp.balance_dataset,
    target_col=target,
    balance_kwargs={"method": "oversampling", "minor_per": 30},
    show_before=False,
)

results_tree_before = compare_tree_and_ensemble(
    X_train,
    y_train,
    X_test,
    y_test,
    show_before=True
)

results_tree_after = compare_tree_and_ensemble(
    X_train,
    y_train,
    X_test,
    y_test,
    balance_fn=dp.balance_dataset,
    target_col=target,
    balance_kwargs={"method": "oversampling", "minor_per": 30},
    show_before=False
)


def add_group(df, group_name):
    out = df.reset_index().copy()
    if "Model" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Model"})
    out.insert(0, "Group", group_name)
    return out

before_table = pd.concat([
    add_group(results_knn_nb_before, "KNN + Naive Bayes"),
    add_group(results_tree_before, "Decision Tree + Ensembles"),
], ignore_index=True)

after_table = pd.concat([
    add_group(results_knn_nb_after, "KNN + Naive Bayes"),
    add_group(results_tree_after, "Decision Tree + Ensembles"),
], ignore_index=True)

before_table = before_table[
    ["Group", "Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score"]
].sort_values(by="F1-score", ascending=False)

after_table = after_table[
    ["Group", "Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score"]
].sort_values(by="F1-score", ascending=False)

print("\n==================== FINAL TABLE: BEFORE BALANCING ====================")
print(before_table.round(4).to_string(index=False))

print("\n==================== FINAL TABLE: AFTER BALANCING ====================")
print(after_table.round(4).to_string(index=False))

print("\nDone.")


"""
plot_results.py
---------------
Generates:
  1. A styled comparison table (Before & After balancing) saved as a figure.
  2. A combined ROC curve plot for all 5 models on the same axes.

Paste the actual ROC curve data (fpr / tpr arrays) from your modeling run,
OR just run this file as-is to reproduce the figures from the output log values.

How to use
----------
Option A – Quick reproduction from logged metrics (no re-training needed):
    Just run this file.  The ROC curves are built from the real AUC values
    already captured in your console output.

Option B – Plug in real fpr/tpr arrays from c306_modeling.py:
    In compare_knn_and_nb / compare_tree_and_ensemble, after each
    evaluate_model() call, add:
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        roc_data["ModelName"] = (fpr, tpr, auc_value)
    then pass roc_data into this script.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# =============================================================================
# 1.  METRIC DATA  (from your console output)
# =============================================================================

before_data = {
    "Model":            ["k-NN",    "Naive Bayes", "Decision Tree", "Bagging", "Random Forest"],
    "Accuracy":         [0.9894,    0.9743,        0.9879,          0.9887,    0.9887],
    "Recall":           [0.2143,    0.3571,        0.0714,          0.3571,    0.2857],
    "Precision":        [0.5000,    0.1667,        0.2500,          0.4545,    0.4444],
    "Specificity":      [0.9977,    0.9809,        0.9977,          0.9954,    0.9962],
    "F1-Score":         [0.3000,    0.2273,        0.1111,          0.4000,    0.3478],
    "ROC-AUC":          [0.9189,    0.9728,        0.9609,          0.9141,    0.9822],
}

after_data = {
    "Model":            ["k-NN",    "Naive Bayes", "Decision Tree", "Bagging", "Random Forest"],
    "Accuracy":         [0.9682,    0.9206,        0.9455,          0.9750,    0.9758],
    "Recall":           [0.4286,    1.0000,        0.9286,          0.3571,    0.4286],
    "Precision":        [0.1500,    0.1176,        0.1548,          0.1724,    0.2000],
    "Specificity":      [0.9740,    0.9197,        0.9457,          0.9817,    0.9817],
    "F1-Score":         [0.2222,    0.2105,        0.2653,          0.2326,    0.2727],
    "ROC-AUC":          [0.8405,    0.9727,        0.9480,          0.9115,    0.9459],
}

df_before = pd.DataFrame(before_data)
df_after  = pd.DataFrame(after_data)

# Model display colours (consistent across both figures)
MODEL_COLORS = {
    "k-NN":          "#1d6fa5",
    "Naive Bayes":   "#8b3a8b",
    "Decision Tree": "#b85c00",
    "Bagging":       "#1a7a4a",
    "Random Forest": "#8b1a1a",
}

MODEL_DASHES = {
    "k-NN":          (None, None),
    "Naive Bayes":   (6, 2),
    "Decision Tree": (3, 3),
    "Bagging":       (8, 3),
    "Random Forest": (4, 2, 1, 2),
}

COLS = ["Model", "Accuracy", "Recall", "Precision", "Specificity", "F1-Score", "ROC-AUC"]


# =============================================================================
# 2.  HELPER – approximate ROC curve from AUC (convex power-law)
#     Replace fpr/tpr arrays here if you have the real values from roc_curve()
# =============================================================================

def approx_roc(auc, n=200):
    """Generate a smooth approximate ROC curve that integrates to `auc`."""
    fpr = np.linspace(0, 1, n)
    # Power-law shape: tpr = fpr^alpha, then solve alpha so AUC matches
    # AUC of fpr^alpha = 1/(alpha+1)  =>  alpha = 1/AUC - 1
    alpha = max(0.01, 1.0 / auc - 1.0)
    tpr = np.power(fpr, alpha)
    return fpr, tpr


# =============================================================================
# 3.  FIGURE 1 – COMPARISON TABLE
# =============================================================================

def make_table_figure(df, title, filename):
    display_df = df[COLS].copy()

    # Highlight best value per metric column (green bg)
    metric_cols = COLS[1:]

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")

    cell_text = []
    for _, row in display_df.iterrows():
        cell_text.append([row["Model"]] + [f"{row[c]:.4f}" for c in metric_cols])

    col_labels = COLS
    col_widths  = [0.18] + [0.12] * len(metric_cols)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)

    # Header row styling
    n_cols = len(COLS)
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2c2c2a")
        cell.set_text_props(color="white", fontweight="bold")

    # Model name column – colour by model
    for i, model_name in enumerate(display_df["Model"], start=1):
        cell = tbl[i, 0]
        cell.set_text_props(color=MODEL_COLORS.get(model_name, "black"), fontweight="bold")
        cell.set_facecolor("#f5f5f3")

    # Highlight best metric per column
    for j, col in enumerate(metric_cols, start=1):
        best_row = display_df[col].idxmax() + 1   # +1 because row 0 is header
        tbl[best_row, j].set_facecolor("#d4edda")
        tbl[best_row, j].set_text_props(fontweight="bold")

    # Zebra striping for data rows
    for i in range(1, len(display_df) + 1):
        for j in range(1, n_cols):
            if tbl[i, j].get_facecolor() == (1.0, 1.0, 1.0, 1.0):   # not already highlighted
                tbl[i, j].set_facecolor("#ffffff" if i % 2 else "#f9f9f7")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")


make_table_figure(df_before, "Model Comparison – Before Balancing", "table_before.png")
make_table_figure(df_after,  "Model Comparison – After Balancing",  "table_after.png")


# =============================================================================
# 4.  FIGURE 2 – COMBINED ROC CURVE (all 5 models, before & after)
# =============================================================================

def make_roc_figure(df, title, filename):
    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in df.iterrows():
        model = row["Model"]
        auc   = row["ROC-AUC"]
        fpr, tpr = approx_roc(auc)

        color  = MODEL_COLORS.get(model, "gray")
        dashes = MODEL_DASHES.get(model, (None, None))

        if dashes[0] is None:
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{model}  (AUC = {auc:.4f})")
        else:
            ax.plot(fpr, tpr, color=color, linewidth=2, dashes=dashes,
                    label=f"{model}  (AUC = {auc:.4f})")

    # Random-chance diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            linewidth=1, label="Random chance")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")


make_roc_figure(df_before, "ROC Curves – Before Balancing", "roc_before.png")
make_roc_figure(df_after,  "ROC Curves – After Balancing",  "roc_after.png")


# =============================================================================
# 5.  COMBINED single figure (table on top, ROC below) – optional
# =============================================================================

def make_combined_figure(df, title_prefix, filename):
    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(title_prefix, fontsize=14, fontweight="bold", y=1.01)

    # --- TABLE (top half) ---
    ax_tbl = fig.add_axes([0, 0.52, 1, 0.48])
    ax_tbl.axis("off")

    display_df = df[COLS].copy()
    metric_cols = COLS[1:]

    cell_text = []
    for _, row in display_df.iterrows():
        cell_text.append([row["Model"]] + [f"{row[c]:.4f}" for c in metric_cols])

    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=COLS,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 2.2)

    for j in range(len(COLS)):
        tbl[0, j].set_facecolor("#2c2c2a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, model_name in enumerate(display_df["Model"], start=1):
        tbl[i, 0].set_text_props(color=MODEL_COLORS.get(model_name, "black"), fontweight="bold")
        tbl[i, 0].set_facecolor("#f5f5f3")

    for j, col in enumerate(metric_cols, start=1):
        best_row = display_df[col].idxmax() + 1
        tbl[best_row, j].set_facecolor("#d4edda")
        tbl[best_row, j].set_text_props(fontweight="bold")

    for i in range(1, len(display_df) + 1):
        for j in range(1, len(COLS)):
            fc = tbl[i, j].get_facecolor()
            if fc == (1.0, 1.0, 1.0, 1.0):
                tbl[i, j].set_facecolor("#ffffff" if i % 2 else "#f9f9f7")

    # --- ROC (bottom half) ---
    ax_roc = fig.add_axes([0.1, 0.04, 0.55, 0.44])

    for _, row in display_df.iterrows():
        model = row["Model"]
        auc   = row["ROC-AUC"]
        fpr, tpr = approx_roc(auc)
        color  = MODEL_COLORS.get(model, "gray")
        dashes = MODEL_DASHES.get(model, (None, None))
        if dashes[0] is None:
            ax_roc.plot(fpr, tpr, color=color, linewidth=2)
        else:
            ax_roc.plot(fpr, tpr, color=color, linewidth=2, dashes=dashes)

    ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=11)
    ax_roc.set_title("ROC Curve Comparison", fontsize=11, fontweight="bold")
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
    ax_roc.grid(True, linestyle="--", alpha=0.4)
    ax_roc.set_aspect("equal")

    # Legend to the right of ROC
    handles = [
        Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2,
               dashes=MODEL_DASHES[m] if MODEL_DASHES[m][0] else [],
               label=f"{m}  (AUC={df.loc[df['Model']==m,'ROC-AUC'].values[0]:.4f})")
        for m in display_df["Model"]
    ]
    handles.append(Line2D([0], [0], color="gray", linestyle="--",
                          linewidth=1, label="Random chance"))
    ax_roc.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")


make_combined_figure(df_before, "Model Comparison – Before Balancing", "combined_before.png")
make_combined_figure(df_after,  "Model Comparison – After Balancing",  "combined_after.png")
