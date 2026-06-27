"""
c306_data_understanding

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
# ---------------------------------------------------
# VARIABLE CLASSIFICATION
def classify_variables(df):
    """
    Classify dataset variables into numeric, ordinal, and categorical groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    tuple
        numeric_vars : list
            List of numeric column names.
        ordinal_vars : list
            List of ordinal column names.
        categorical_vars : list
            List of categorical column names.

    Notes
    -----
    Only variables that exist in the dataframe are returned.
    """

    if df is None or df.empty:
        return [], [], []

    numeric_vars = [
         'hours_studied', 'attendance', 'sleep_hours', 'previous_scores', 
                'tutoring_sessions', 'physical_activity'
    ]

    ordinal_vars = [
        'parental_involvement', 'access_to_resources', 
    'motivation_level', 'family_income', 'teacher_quality', 
    'parental_education_level', 'distance_from_home', 'peer_influence' 
    ]

    categorical_vars = [
        'extracurricular_activities', 
    'internet_access',  
    'school_type', 'learning_disabilities', 
    'gender',"pass_fail"
    ]


    numeric_vars = [c for c in numeric_vars if c in df.columns]
    ordinal_vars = [c for c in ordinal_vars if c in df.columns]
    categorical_vars = [c for c in categorical_vars if c in df.columns]

# keep only categorical variables that have more than one category
    categorical_vars = [
        c for c in categorical_vars
        if c in df.columns and df[c].nunique() > 1
        ]
    
    return numeric_vars, ordinal_vars, categorical_vars

# ---------------------------------------------------
# HELPER FUNCTION FOR TABLES

def save_table_to_pdf(df_table, title, pdf, rows_per_page=20):
    """
    Save large tables across multiple pages in the PDF report.

    Parameters
    ----------
    df_table : pandas.DataFrame
        Table to be exported to PDF.
    title : str
        Title displayed on the PDF page.
    pdf : matplotlib.backends.backend_pdf.PdfPages
        PDF object used to store the figures.
    rows_per_page : int, optional
        Maximum number of rows per page (default is 20).

    Returns
    -------
    None
    """

    if df_table is None or df_table.empty:
        return

    df_table = df_table.T
    total = len(df_table)

    for start in range(0, total, rows_per_page):

        subset = df_table.iloc[start:start+rows_per_page]

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        table = ax.table(
            cellText=subset.round(3).values,
            colLabels=subset.columns,
            rowLabels=subset.index,
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        ax.set_title(title)

        plt.tight_layout()

        pdf.savefig(fig)
        plt.close()


# ---------------------------------------------------
# DATA OVERVIEW

def data_overview(df, pdf=None):
    """
    Retrieve general information about the dataset.

    Computes:
    - Number of rows and columns
    - Total missing values
    - Number of duplicate rows
    - Descriptive statistics for numeric and categorical variables

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    pdf : PdfPages object
        PDF file where results will be saved.

    Returns
    -------
    dict
        Dictionary containing dataset summary information.
    """

    if df is None or df.empty:
        print("Empty DataFrame.")
        return {}

    rows, cols = df.shape
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    numeric_vars, ordinal_vars, categorical_vars = classify_variables(df)

    numeric_stats = df[numeric_vars].describe() if numeric_vars else pd.DataFrame()

    cat_ord = categorical_vars + ordinal_vars
    categorical_df = df[cat_ord].copy().sort_index()

    for col in ordinal_vars:
        categorical_df[col] = categorical_df[col].astype("category")

    categorical_stats = categorical_df.describe() if not categorical_df.empty else pd.DataFrame()

    print("Rows:", rows)
    print("Columns:", cols)
    print("Missing:", missing)
    print("Duplicates:", duplicates)

    if pdf:

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        text = f"""
Dataset Overview

Rows: {rows}
Columns: {cols}

Missing values: {missing}
Duplicate rows: {duplicates}
"""
        ax.text(0.1, 0.7, text, fontsize=14)

        pdf.savefig(fig)
        plt.close()

        save_table_to_pdf(numeric_stats, "Numeric Statistics", pdf)
        save_table_to_pdf(categorical_stats, "Categorical Statistics", pdf)

    return {
        "rows": rows,
        "columns": cols,
        "missing_values": missing,
        "duplicates": duplicates,
        "numeric_summary": numeric_stats,
        "categorical_summary": categorical_stats
    }


# ---------------------------------------------------
# UNIVARIATE ANALYSIS

def univariate_plot(df, pdf=None):
    """
    Generate univariate visualizations for all dataset variables.

    Numeric variables:
        - Histogram
        - Boxplot

    Categorical and ordinal variables:
        - Pareto chart (bar chart with cumulative percentage line)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    pdf : PdfPages object
        PDF file where plots will be saved.

    Returns
    -------
    None
    """

    if df is None or df.empty:
        return

    numeric_vars, ordinal_vars, categorical_vars = classify_variables(df)

    for col in numeric_vars:

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        df[col].hist(ax=ax[0])
        ax[0].set_title(f"{col} Histogram")
        ax[0].set_xlabel(col)

        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title(f"{col} Boxplot")

        plt.tight_layout()

        if pdf:
            pdf.savefig(fig)
            plt.close()

    for col in categorical_vars + ordinal_vars:

        counts = df[col].value_counts().sort_values(ascending=False)
        cumulative = counts.cumsum() / counts.sum() * 100

        fig, ax1 = plt.subplots(figsize=(8, 4))

        ax1.bar(counts.index.astype(str), counts.values)
        ax1.set_title(f"{col} Pareto Chart")
        ax1.set_ylabel("Frequency")

        ax2 = ax1.twinx()
        ax2.plot(counts.index.astype(str), cumulative, marker="o", color="red")
        ax2.set_ylabel("Cumulative %")

        plt.xticks(rotation=45)
        plt.tight_layout()

        if pdf:
            pdf.savefig(fig)
            plt.close()



def plot_distribution_matrix(df, label="", pdf=None):
    """
    Plot a matrix of histograms and a matrix of boxplots
    for all numeric variables. Designed to be called before
    and after outlier/skewness fixing.

    Parameters
    ----------
    df : pandas.DataFrame
    label : str
        Label shown in the title, e.g. "Before" or "After".
    pdf : PdfPages object
    """
    if df is None or df.empty:
        return

    numeric_vars, _, _ = classify_variables(df)
    num_cols = [c for c in numeric_vars if c in df.columns]

    if not num_cols:
        return

    n = len(num_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    # ── HISTOGRAM MATRIX ──────────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    fig.suptitle(f"Histogram Matrix — {label}", fontsize=14, y=1.01)
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="black")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):   # hide unused subplots
        axes[j].set_visible(False)

    plt.tight_layout()
    if pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── BOXPLOT MATRIX ────────────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    fig.suptitle(f"Boxplot Matrix — {label}", fontsize=14, y=1.01)
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------
# NUMERIC RELATIONSHIPS

def relations_num(df, vars="all", chart_type="heatmap", pdf=None):
    """
    Demonstrate relationships between numeric variables.

    Parameters
    ----------
    df : pandas.DataFrame
    vars : list or "all"
        Numeric variables to include.
    chart_type : str
        'pair' for pairplot or 'heatmap' for correlation heatmap.
    pdf : PdfPages object
        PDF file to store the plot.

    Returns
    -------
    None
    """

    numeric_vars, _, _ = classify_variables(df)

    if not numeric_vars:
        return

    if vars != "all":
        numeric_vars = [v for v in vars if v in numeric_vars]

    num_df = df[numeric_vars]

    if chart_type == "pair":

        g = sns.pairplot(num_df)

        if pdf:
            pdf.savefig(g.fig)
            plt.close()

    else:

        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size":7},
            ax=ax
        )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if pdf:
            pdf.savefig(fig)
            plt.close()

        return corr

# ---------------------------------------------------
def correlation_num_ordinal(df, pdf=None):
    """
    Compute Spearman correlation between numeric variables and ordinal variable 'rad'.
    """

    numeric_vars, ordinal_vars, _ = classify_variables(df)

    if "rad" not in ordinal_vars:
        print("No ordinal variable 'rad' found.")
        return

    # Encode rad as ordered values
    rad_map = {"Low": 0, "Medium": 1, "High": 2}
    rad_encoded = df["rad"].map(rad_map)

    results = {}

    for col in numeric_vars:
        corr = df[col].corr(rad_encoded, method="spearman")
        results[col] = corr

    corr_df = pd.DataFrame.from_dict(results, orient="index", columns=["Spearman_corr_with_rad"])

    print("\n--- CORRELATION: NUMERIC vs RAD (Spearman) ---")
    print(corr_df)

    # Save to PDF
    if pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        table = ax.table(
            cellText=corr_df.round(3).values,
            rowLabels=corr_df.index,
            colLabels=corr_df.columns,
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        ax.set_title("Spearman Correlation: Numeric vs rad")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    return corr_df

# ---------------------------------------------------
# CATEGORICAL RELATIONSHIPS

def relations_cat(df, var1, var2, chart_type="bar", pdf=None):
    """
    Analyze relationship between two categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame
    var1 : str
        First categorical variable.
    var2 : str
        Second categorical variable.
    chart_type : str
        Type of plot (default 'bar').
    pdf : PdfPages object

    Returns
    -------
    None
    """

    if (
        var1 not in df.columns
        or var2 not in df.columns
        or df[var1].nunique() <= 1
        or df[var2].nunique() <= 1
    ):
        return

    cross = pd.crosstab(df[var1], df[var2]).sort_index()
    cross = cross.sort_index(axis=1)
    print(cross)

    fig, ax = plt.subplots()

    if chart_type == "bar":
        cross.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title(f"{var1} vs {var2}")
    ax.set_xlabel(var1)
    ax.set_ylabel("Count")

    plt.tight_layout()

    if pdf:
        pdf.savefig(fig)
        plt.close()


# ---------------------------------------------------
# CATEGORICAL vs NUMERIC


def relations_cat_num(df, grouping_vars="all", num_vars="all", chart_type="box", pdf=None):
    """
    Analyze relationships between categorical and numeric variables.

    Parameters
    ----------
    df : pandas.DataFrame
    grouping_vars : list or "all"
        Categorical variables used for grouping.
    num_vars : list or "all"
        Numeric variables analyzed.
    chart_type : str
        Plot type ("box" or "violin").
    pdf : PdfPages object
    """

    if df is None or df.empty:
        return

    numeric_vars, ordinal_vars, categorical_vars = classify_variables(df)

    if grouping_vars == "all":
        grouping_vars = categorical_vars + ordinal_vars

    if num_vars == "all":
        num_vars = numeric_vars

    max_plots = 30
    plot_count = 0

    for cat in grouping_vars:

        if cat not in df.columns or df[cat].nunique() <= 1:
            continue

        for num in num_vars:

            if plot_count >= max_plots:
                return

            if num not in df.columns or df[num].nunique() <= 1:
                continue

# ----- Aggregated statistics -----
            stats = df.groupby(cat)[num].agg(["mean", "median", "count"])

            print(f"\nStatistics for {num} by {cat}")
            print(stats)

            order = sorted(df[cat].dropna().unique())

            fig, ax = plt.subplots(figsize=(8, 4))

 # ----- Visualization -----
            if chart_type == "violin":

                sns.violinplot(
                    x=cat,
                    y=num,
                    data=df,
                    order=order,
                    inner="quartile",
                    ax=ax
                )

            else:  # default boxplot

                sns.boxplot(
                    x=cat,
                    y=num,
                    data=df,
                    order=order,
                    ax=ax
                )

            ax.set_title(f"{num} by {cat}")
            ax.set_xlabel(cat)
            ax.set_ylabel(num)

            plt.xticks(rotation=45)
            plt.tight_layout()

            if pdf:
                pdf.savefig(fig)

            plt.close(fig)
            plt.cla()
            plt.clf()

            plot_count += 1
