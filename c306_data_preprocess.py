"""
c306_data_preprocess
Data preprocessing module 

"""

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from c306_data_understanding import classify_variables
from scipy.stats import zscore


from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from statsmodels.stats.outliers_influence import variance_inflation_factor



# -------------------------------------------------------
# 1. DELETE DUPLICATES
# -------------------------------------------------------

def delete_duplicates(df):
    """
    Remove duplicate rows from the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        New DataFrame with duplicate rows removed.
    """

    if df.empty:
        return df.copy()

    return df.drop_duplicates().copy()


# -------------------------------------------------------
# 2. HANDLE MISSING VALUES
# -------------------------------------------------------

def repair_missing(df, method="impute"):
    """
    Detect and repair missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    method : str
        Method used for numerical columns:
        "mean", "median", or "mode".

    Returns
    -------
    pandas.DataFrame
        New DataFrame with missing values handled.
    """

    new_df = df.copy()

    if new_df.empty:
        return new_df

    for col in new_df.columns:

        if new_df[col].isna().sum() == 0:
            continue

        # categorical / text columns
        if (
            pd.api.types.is_object_dtype(new_df[col])
            or pd.api.types.is_string_dtype(new_df[col])
            or pd.api.types.is_categorical_dtype(new_df[col])
        ):

            if new_df[col].mode().empty:
                continue

            new_df[col] = new_df[col].fillna(new_df[col].mode()[0])

        # numerical columns
        else:

            if method == "mean":
                new_df[col] = new_df[col].fillna(new_df[col].mean())

            elif method == "median":
                new_df[col] = new_df[col].fillna(new_df[col].median())

            else: # default "impute"
                new_df[col] = new_df[col].fillna(new_df[col].median())

    return new_df


# -------------------------------------------------------
# 3. HANDLE OUTLIERS
# -------------------------------------------------------

def repair_outliers(df, detection="IQR", method="cap", z_thresh=3.0):
    """
    Detect and repair outliers using IQR or Z-score.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    detection : str
        "IQR" or "Z-SCORE".
    method : str
        "cap", "remove", or "impute".
    z_thresh : float
        Z-score threshold when detection="Z-SCORE".

    Returns
    -------
    pandas.DataFrame
        New DataFrame with outliers handled.
    """
    new_df = df.copy()

    numeric_vars, _, _ = classify_variables(new_df)
    num_cols = [c for c in numeric_vars if c in new_df.columns]

    if len(num_cols) == 0:
        return new_df

    remove_mask = pd.Series(False, index=new_df.index)

    for col in num_cols:
        s = new_df[col]

        if detection.upper() == "IQR":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (s < lower) | (s > upper)

        elif detection.upper() == "Z-SCORE":
            z = np.abs(zscore(s, nan_policy="omit"))
            mask = pd.Series(z > z_thresh, index=new_df.index)

            lower = s.mean() - z_thresh * s.std()
            upper = s.mean() + z_thresh * s.std()

        else:
            raise ValueError("detection must be 'IQR' or 'Z-SCORE'")

        if method == "cap":
            new_df[col] = np.clip(s, lower, upper)

        elif method == "impute":
            new_df.loc[mask, col] = s.median()

        elif method == "remove":
            remove_mask = remove_mask | mask

        else:
            raise ValueError("method must be 'cap', 'remove', or 'impute'")

    if method == "remove":
        new_df = new_df.loc[~remove_mask].copy()

    return new_df


# -------------------------------------------------------
# 4. FIX SKEWNESS
# -------------------------------------------------------
    
#     Symmetric distribution → skew = 0
#     Right skew → skew > 0
#     Left skew → skew < 0


def fix_skewness(df, skew_threshold=1.0):
    new_df = df.copy()
    numeric_vars, _, _ = classify_variables(new_df)
    num_cols = [c for c in numeric_vars if c in new_df.columns]


    print("\n--- SKEWNESS STEP ---")
    print(f"Skewness threshold: |skew| > {skew_threshold}\n")

    transformed_cols = []
    unchanged_cols = []

    for col in num_cols:
        s = pd.to_numeric(new_df[col], errors="coerce").dropna()

        if s.empty:
            print(f"{col}: skipped (no valid numeric values)")
            continue

        before_skew = s.skew()

        if abs(before_skew) > skew_threshold:
            print(f"{col}: skewness BEFORE = {before_skew:.4f}")

            if (s > 0).all():
                try:
                    transformed, _ = boxcox(s)
                    new_df[col] = transformed
                    after_skew = pd.Series(transformed).skew()
                    print(f"{col}: skewness AFTER  = {after_skew:.4f}")
                    print("      -> transformed with Box-Cox")
                except Exception:
                    pt = PowerTransformer(method="yeo-johnson")
                    transformed = pt.fit_transform(new_df[[col]]).flatten()
                    new_df[col] = transformed
                    after_skew = pd.Series(transformed).skew()
                    print(f"{col}: skewness AFTER  = {after_skew:.4f}")
                    print("      -> transformed with Yeo-Johnson fallback")
            else:
                pt = PowerTransformer(method="yeo-johnson")
                transformed = pt.fit_transform(new_df[[col]]).flatten()
                new_df[col] = transformed
                after_skew = pd.Series(transformed).skew()
                print(f"{col}: skewness AFTER  = {after_skew:.4f}")
                print("      -> transformed with Yeo-Johnson")

            transformed_cols.append(col)

        else:
            print(f"{col}: skewness = {before_skew:.4f}")
            print("      -> already within threshold after previous preprocessing; no transformation applied")
            unchanged_cols.append(col)

        print()

    print("========== SKEWNESS SUMMARY ==========")
    print("Transformed columns:", transformed_cols if transformed_cols else "None")
    print("Already within threshold:", unchanged_cols if unchanged_cols else "None")
    print("======================================\n")

    return new_df


def _outlier_stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((s < lower) | (s > upper)).sum()

    return q1, q3, iqr, lower, upper, n_outliers


def print_outlier_placeholder(df=None):
    print("\n==================== OUTLIER ANSWERS ====================\n")

    if df is None or df.empty:
        print("Q1 = —")
        print("Q3 = —")
        print("IQR = —")
        print("Lower fence = —")
        print("Upper fence = —")
        print("Number of outliers = —\n")
        return

    from c306_data_understanding import classify_variables

    numeric_vars, ordinal_vars, _ = classify_variables(df)

    # ONLY true numeric (continuous) variables
    num_cols = [c for c in numeric_vars if c in df.columns]

    for col in num_cols:
        stats = _outlier_stats(df[col])
        if stats is None:
            continue

        q1, q3, iqr, lower, upper, n_outliers = stats

        print(f"{col}:")
        print(f"Q1 = {q1:.4f}")
        print(f"Q3 = {q3:.4f}")
        print(f"IQR = {iqr:.4f}")
        print(f"Lower fence = {lower:.4f}")
        print(f"Upper fence = {upper:.4f}")
        print(f"Number of outliers = {n_outliers}\n")

# -------------------------------------------------------
# 5. SCALE NUMERIC VARIABLES
# -------------------------------------------------------

def scale_numeric(df, method="standard"):
    """
    Scale numerical variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    method : str
        Scaling method:
        "standard", "minmax", or "robust".

    Returns
    -------
    pandas.DataFrame
        New DataFrame with scaled numeric variables.
    """

    new_df = df.copy()

    numeric_vars, _, _ = classify_variables(new_df)
    num_cols = [c for c in numeric_vars if c in new_df.columns]

    if len(num_cols) == 0:
        return new_df

    if method == "standard":        # z = (x − mean) / std; mean = 0, std = 1
        scaler = StandardScaler()

    elif method == "minmax":        # (x − min) / (max − min); values range between 0 and 1
        scaler = MinMaxScaler()

    elif method == "robust":        # Uses median and IQR --> less sensitive to outliers
        scaler = RobustScaler()

    else:
        raise ValueError("Invalid scaling method")

    new_df[num_cols] = scaler.fit_transform(new_df[num_cols])

    return new_df


# -------------------------------------------------------
# 6. ENCODE CATEGORICAL VARIABLES
# -------------------------------------------------------

def encode_categorical(df, method="frequency", target_variable=None):
    """
    Encode categorical variables.

    Small category variables (<10) → One-Hot Encoding.
    Large category variables → Frequency or Target encoding.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    method : str
        Encoding method for large categories:
        "frequency" or "target".

    target_variable : str
        Required when method="target".

    Returns
    -------
    pandas.DataFrame
        New DataFrame with encoded categorical variables.
    """

    new_df = df.copy()

    cat_cols = new_df.select_dtypes(include=["object", "string", "category"]).columns

    # exclude target colomn:
        
    if target_variable is not None:
        cat_cols = [col for col in cat_cols if col != target_variable]


    if len(cat_cols) == 0:
        return new_df

    for col in cat_cols:

        if new_df[col].nunique() < 10:

            dummies = pd.get_dummies(new_df[col], prefix=col)

            new_df = pd.concat([new_df.drop(col, axis=1), dummies], axis=1)

        else:

            if method == "frequency": #--> Used when categories are many

                freq = new_df[col].value_counts() / len(new_df)

                new_df[col] = new_df[col].map(freq)

            elif method == "target":
                if target_variable is None or target_variable not in new_df.columns:
                    raise ValueError("target_variable must be provided")

                if not pd.api.types.is_numeric_dtype(new_df[target_variable]):
                    raise ValueError(
                        "Target encoding requires a numeric target variable. "
                        "Encode the target first or use method='frequency'."
                    )

                target_mean = new_df.groupby(col)[target_variable].mean()
                global_mean = new_df[target_variable].mean()
                new_df[col] = new_df[col].map(target_mean).fillna(global_mean)

    return new_df


# -------------------------------------------------------
# 7. DISCRETIZE NUMERIC VARIABLE
# -------------------------------------------------------

def discretize_numeric(variable, method="equal_width", bins=5):
    """
    Discretize a numeric variable into bins.

    Parameters
    ----------
    variable : pandas.Series
        Numeric variable to discretize.

    method : str
        "equal_width", "equal_frequency", or "kmeans".

    bins : int
        Number of bins.

    Returns
    -------
    pandas.Series
        Discretized variable.
    """

    if method == "equal_width":         # Splits range into equal interval
        return pd.cut(variable, bins).astype(str).astype('category').cat.codes

    elif method == "equal_frequency":   # Each bin contains equal number of observations
        return pd.qcut(variable, bins).astype(str).astype('category').cat.codes

    elif method == "kmeans":            # Clusters the data into bins using KMeans clustering

        km = KMeans(n_clusters=bins, random_state=42)

        labels = km.fit_predict(variable.values.reshape(-1, 1))

        return pd.Series(labels, index=variable.index)

    else:
        raise ValueError("Invalid discretization method")
        


# =========================================================
# III) FEATURE SELECTION 
# =========================================================

# =========================================================
# 1. FILTER SELECTION  
# =========================================================

from scipy.stats import chi2_contingency

def filter_selection(df, target_variable, alpha=0.05, del_flag=True):
    """
    Filter selection following the PDF logic:
    - Categorical predictors vs categorical target -> Chi-square
    - Numeric / ordinal predictors vs target -> Mutual Information
    """

    print("\n" + "=" * 70)
    print("FILTER SELECTION START")
    print("=" * 70)

    new_df = df.copy()

    if target_variable not in new_df.columns:
        raise ValueError("Target variable not found")

    X = new_df.drop(columns=[target_variable]).copy()
    y = new_df[target_variable].copy()
    removed_features = []

    print("Original dataframe shape:", new_df.shape)
    print("\nTarget variable:", target_variable)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("y dtype:", y.dtype)

    # -------------------------------------------------
    # Encode target only for MI
    # -------------------------------------------------
    
    #  checks if the target column is stored as strings or Categorical
    
    is_classification = (
    not pd.api.types.is_numeric_dtype(y)
    or y.nunique() <= 10
)

    if is_classification:
        print("Task type: classification")
        y_encoded = LabelEncoder().fit_transform(y.astype(str).fillna("MISSING"))
    else:
        print("Task type: regression")
        y_encoded = pd.to_numeric(y, errors="coerce").fillna(y.median())

    # -------------------------------------------------
    # 1) Categorical predictors -> Chi-square if target categorical
    #    If target numeric, use MI after one-hot encoding
    # -------------------------------------------------
    
    numeric_vars, ordinal_vars, categorical_vars = classify_variables(X)

    categorical_cols = categorical_vars     # ONLY nominal
    ordinal_cols = ordinal_vars             # handled separately
    
    
    print("\n--- STEP 1: Categorical predictors ---")

    for col in categorical_cols:
        s = X[col].astype("object").fillna("MISSING")

        if s.nunique() < 2:
            continue

        if is_classification:
           
# If the target y is categorical (classification task):
# Build a contingency table (ct) between the predictor s and the target y.

            ct = pd.crosstab(s.astype(str), y.astype(str).fillna("MISSING"))

            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue

 # Chi-square test of independence
            _, p_val, _, _ = chi2_contingency(ct)

            print(f"{col} -> p-value = {p_val:.6f}")

            if p_val >= alpha:
                removed_features.append(col)
                if del_flag:
                    X = X.drop(columns=[col], errors="ignore")
                print(f"→ REMOVE {col}")
            else:
                print(f"→ KEEP {col}")

        else:
            # Numeric target: use MI on one-hot encoded predictor,
            #One‑hot encode the categorical predictor into dummy variables.
            dummies = pd.get_dummies(s.astype(str), prefix=col, drop_first=False)

            if dummies.shape[1] == 0:
                continue

            mi_vals = mutual_info_regression(dummies.astype(float), y_encoded)
            mi_score = float(np.mean(mi_vals)) # Take the mean MI score across dummy columns.

            print(f"{col} -> mean MI = {mi_score:.6f}")

            if mi_score < 0.005:
                removed_features.append(col)
                if del_flag:
                    X = X.drop(columns=[col], errors="ignore")
                print(f"→ REMOVE {col}")
            else:
                print(f"→ KEEP {col}")

    # -------------------------------------------------
    # 2) Numeric / ordinal predictors -> MI
    # -------------------------------------------------
    print("\n--- STEP 2: Numeric / ordinal predictors ---")

    numeric_cols = X.select_dtypes(include="number").columns.tolist()

    # -------- FIX: ENCODE ORDINAL VARIABLES --------
    for col in ordinal_cols:
        if col in X.columns:
            X[col] = pd.Categorical(
                X[col],
                categories=["Low", "Medium", "High"],
                ordered=True
            ).codes

    # ADD ordinal variables AFTER encoding
    numeric_cols += ordinal_cols
    numeric_cols = list(set(numeric_cols))

    print("Numeric vars (for MI):", numeric_cols)

    if len(numeric_cols) > 0:
        X_num = X[numeric_cols].copy()

        if is_classification:
            mi = mutual_info_classif(X_num, y_encoded, random_state=42)
        else:
            mi = mutual_info_regression(X_num, y_encoded, random_state=42)


        # converts raw MI results into a readable table, then sorts them.
        
        mi_series = pd.Series(mi, index=numeric_cols)
        
        # Map MI values to features

        print("\nMI values:")
        print(mi_series.sort_values())
        

        low_mi = mi_series[mi_series < 0.005].index.tolist()

        if low_mi:
            print("Low MI features:", low_mi)
            removed_features.extend(low_mi)

            if del_flag:
                X = X.drop(columns=low_mi, errors="ignore")

    print("\n--- FINAL RESULT ---")
    print("Removed features:", removed_features)
    print("Remaining shape:", X.shape)

    if X.shape[1] == 0:
        print("No features left")
        return new_df[[target_variable]]

    return pd.concat([X, y], axis=1)

# ----------------------------------------------------------------------
def reduce_collinearity(df, target_variable, VIF_cut=5, cor_cut=0.7, del_flag=True):
    """
    Reduce multicollinearity among numeric predictors.

    Order:
    1) Correlation
    2) VIF
    """

    new_df = df.copy()

    if target_variable not in new_df.columns:
        raise ValueError("Target variable not found")

    y = new_df[target_variable].copy()
    X = new_df.drop(columns=[target_variable]).copy()

    # Split predictors into numeric and non-numeric blocks
    X_num = X.select_dtypes(include="number").copy()
    X_other = X.select_dtypes(exclude="number").copy()

    # -------------------------------------------------
    # Encode ordinal variables for correlation (Spearman)
    # -------------------------------------------------

    # Only encode ordinal variables (NOT categorical)
    ordinal_map = {"Low": 0, "Medium": 1, "High": 2}

    for col in X_other.columns:
        if col == "rad":   # ordinal variable
            X_other[col] = X_other[col].map(ordinal_map)


    # Remove constant columns
    X_num = X_num.loc[:, X_num.nunique(dropna=False) > 1]

    if X_num.shape[1] == 0:
        print("No numeric features left after filtering")
        return pd.concat([X, y], axis=1)

    removed_features = []

    # Encode target for MI
    is_classification = (
    not pd.api.types.is_numeric_dtype(y)   # catches string, Arrow string, category
    or y.nunique(dropna=True) <= 10         # catches 0/1 numeric targets
)

    if is_classification:
        y_encoded = LabelEncoder().fit_transform(y.astype(str).fillna("MISSING"))
    else:
        y_encoded = pd.to_numeric(y, errors="coerce").fillna(y.median())

    # -------------------------------------------------
    # Step 1: Correlation -> remove lower MI feature
    # -------------------------------------------------

    print("\n--- CORRELATION STEP ---")
    
    step_count = 0

    while True:
        if X_num.shape[1] < 2:
            break

        X_scaled = X_num.fillna(0)

        if is_classification:
            mi = mutual_info_classif(X_scaled, y_encoded, random_state=42)
        else:
            mi = mutual_info_regression(X_scaled, y_encoded, random_state=42)

        mi_series = pd.Series(mi, index=X_num.columns)

        # Combine numeric + ordinal (encoded) for correlation
        # Keep only the ordinal variable for correlation
        ordinal_corr = pd.DataFrame(index=X.index)

        if "rad" in X_other.columns:
            ordinal_corr["rad"] = X_other["rad"].map({"Low": 0, "Medium": 1, "High": 2})

        # Do NOT include chas here, because it is nominal/string
        X_corr = pd.concat([X_num, ordinal_corr], axis=1)

        # Spearman is fine for ordinal-aware screening
        
        print("\nUsing Spearman correlation for collinearity detection (captures monotonic relationships and handles ordinal variables).")
       
        corr_matrix = X_corr.corr(method="spearman").abs()
     
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        high_pairs = []     # This will store --> (feature1, feature2, correlation_value)
        for col in upper.columns:
            for row in upper.index:
                val = upper.loc[row, col]
                if pd.notna(val) and val > cor_cut:  # pd.notna(val) --> Ignore NaN values (empty cells)
                    high_pairs.append((row, col, val))

        if not high_pairs:
            print("No feature pairs exceeded the correlation threshold.")
            break

        row, col, corr_val = max(high_pairs, key=lambda t: t[2])

        print(f"High correlation pair: {row} vs {col} -> corr = {corr_val:.4f}")

        if not del_flag:
            print("Reporting only; no deletion performed.")
            break

        drop_feature = row if mi_series[row] <= mi_series[col] else col
        print(f"→ Dropping lower MI feature: {drop_feature}")

        X_num = X_num.drop(columns=[drop_feature], errors="ignore")
        removed_features.append(drop_feature)
        step_count += 1
        
    print(f"Correlation step finished. Removed {step_count} feature(s).")

    # -------------------------------------------------
    # Step 2: VIF -> iterative removal
    # -------------------------------------------------
    print("\n--- VIF STEP ---")

    while X_num.shape[1] > 1:
        vif_data = pd.DataFrame({
            "feature": X_num.columns,
            "VIF": [
                variance_inflation_factor(X_num.fillna(0).values, i)
                for i in range(X_num.shape[1])
            ]
        })

        print("\nCurrent VIF values:")
        print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))

        max_vif = vif_data["VIF"].max()

        if max_vif <= VIF_cut:
            break

        drop_feature = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
        print(f"High VIF feature: {drop_feature} -> VIF = {max_vif:.4f}")

        if del_flag:
            X_num = X_num.drop(columns=[drop_feature], errors="ignore")
            removed_features.append(drop_feature)
        else:
            print("Reporting only; no deletion performed.")
            break

    print("Removed features (collinearity):", removed_features)

    # Put the untouched non-numeric predictors back
    result = pd.concat([X_num, X_other, y], axis=1)
    return result

# -------------------------------------------------------------

def wrapper_feature_selection(df, target_col,
                              Encoding_flag=True,
                              n_features=5,
                              direction="forward",
                              model_type="classification"):

    new_df = df.copy()

    if target_col not in new_df.columns:
        raise ValueError("Target column not found")

    X = new_df.drop(columns=[target_col]).copy()
    y = new_df[target_col].copy()

    if Encoding_flag:
        X = pd.get_dummies(X, drop_first=True)

    if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y) or pd.api.types.is_categorical_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))

    if model_type == "classification":
        model = RandomForestClassifier(random_state=42)
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(random_state=42)
        scoring = "r2"

    if X.shape[1] <= 1:
        print("Not enough features")
        return new_df

# Adjust number of features, Prevent error if: Requested features > available features
    n_features = min(n_features, X.shape[1])

# Select best subset of features using cross-validation
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction=direction,
        scoring=scoring,
        cv=3                 # 5-fold cross-validation
    )


# Fit selector, Tests many feature combinations, Chooses best subset
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()].tolist()

    print("Selected features (wrapper):", selected_features)

    return pd.concat([X[selected_features], new_df[[target_col]]], axis=1)

# =========================================================
# 4. FEATURE EXTRACTION (PCA)
# =========================================================

from sklearn.decomposition import PCA


def extract_features(df, target_col=None, scale_flag=True, CV_cut=80):
    
    """
    Apply PCA to numeric features and return principal components
    that explain at least CV_cut % of variance.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    scale_flag : bool
        Whether to scale data before PCA.

    CV_cut : int
        Cumulative variance threshold (e.g., 80).

    Returns
    -------
    pandas.DataFrame
        DataFrame of selected principal components.
    """

    new_df = df.copy()

    if new_df.empty:
        print("Empty DataFrame → PCA skipped")
        return new_df

    print("\n========== PCA START ==========")

    # Save target FIRST so it is not lost even if it is categorical
    y = None
    if target_col is not None and target_col in new_df.columns:
        y = new_df[target_col].copy()

    # ---------------------------
    # Select numeric predictors only
    # ---------------------------

    X = new_df.select_dtypes(include="number")

    # Remove target if it somehow exists among numeric columns
    if target_col is not None and target_col in X.columns:
        X = X.drop(columns=[target_col], errors="ignore")

    # Remove constant columns
    X = X.loc[:, X.nunique(dropna=False) > 1]


    if X.shape[1] == 0:
        print("No numeric features for PCA")
        if y is not None and target_col is not None:
            return pd.DataFrame({target_col: y.values}, index=new_df.index)
        return new_df
    

    print("Input shape:", X.shape)
    print("Numeric columns:", list(X.columns))

    # ---------- Scaling ----------
    
    #  use already scaled data
    if scale_flag:
        from sklearn.preprocessing import StandardScaler
        X_proc = StandardScaler().fit_transform(X)
        print("\nScaling applied before PCA")
    else:
        X_proc = X.values
        print("\nUsing pre-scaled features")


    # ---------- PCA ----------
    pca = PCA()
    pca.fit(X_proc)

    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)

    # ---------------------------
    # Print explained variance
    # ---------------------------
    print("\nExplained Variance per Component (%):")
    for i, var in enumerate(explained_var):
        print(f"PC{i+1}: {var:.2f}%")

    print("\nCumulative Variance (%):")
    for i, var in enumerate(cumulative_var):
        print(f"PC{i+1}: {var:.2f}%")

    # ---------------------------
    # number of components to keep
    # ---------------------------
    
    n_components = np.argmax(cumulative_var >= CV_cut) + 1

    print(f"Selected {n_components} components to reach {CV_cut}% variance")


    # transform again with selected components
    
    pca_final = PCA(n_components=n_components)
    X_reduced = pca_final.fit_transform(X_proc)


    # create DataFrame
    
    columns = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_reduced, columns=columns, index=new_df.index)

    # add target back from the original dataframe
    if target_col is not None and target_col in new_df.columns:
        df_pca[target_col] = new_df[target_col].values

    # ---------------------------
    # Loadings 
    # ---------------------------
    loadings = pd.DataFrame(
        pca_final.components_.T,
        index=X.columns,
        columns=columns
    )

    print("\nPCA Loadings:")
    print(loadings.round(4))


    # ---------------------------
    # Preview
    # ---------------------------
    print("\nPCA Output Preview:")
    print(df_pca.head())

    print("\n========== PCA END ==========\n")
    return df_pca

# =========================================================
# 5. SAMPLE SELECTION & BALANCING
# =========================================================

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def balance_dataset(df, target_col, method="oversampling", minor_per=30):
    """
    Balance dataset based on target variable.
    """

    new_df = df.copy()

    if target_col not in new_df.columns:
        raise ValueError("Target column not found")

    X = new_df.drop(columns=[target_col])
    y = new_df[target_col]

    class_counts = y.value_counts()
    class_perc = y.value_counts(normalize=True) * 100

    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    minority_percentage = class_perc.min()

    print("Before balancing:")
    print(class_perc)

    if minority_percentage >= minor_per:
        print("Dataset already balanced")
        return new_df

    target_ratio = minor_per / 100

    if method == "oversampling":
        desired_minority = int(np.ceil((target_ratio * majority_count) / (1 - target_ratio)))
        desired_minority = max(desired_minority, minority_count)

        sampler = RandomOverSampler(
            sampling_strategy={minority_class: desired_minority},
            random_state=42
        )

    elif method == "undersampling":
        desired_majority = int(np.floor(minority_count * (1 - target_ratio) / target_ratio))
        desired_majority = max(desired_majority, 1)

        sampler = RandomUnderSampler(
            sampling_strategy={majority_class: desired_majority},
            random_state=42
        )

    elif method == "smote":
        if minority_count < 2:
            raise ValueError("SMOTE needs at least 2 samples in the minority class.")

        # prevent SMOTE crash on categorical data
        if not X.select_dtypes(exclude="number").empty:
            raise ValueError(
                "SMOTE requires numeric features. Encode categorical variables first "
                "using encode_categorical()."
            )

        desired_minority = int(np.ceil((target_ratio * majority_count) / (1 - target_ratio)))
        desired_minority = max(desired_minority, minority_count)

        k_neighbors = min(5, minority_count - 1)

        sampler = SMOTE(
            sampling_strategy={minority_class: desired_minority},
            random_state=42,
            k_neighbors=k_neighbors
        )

    else:
        raise ValueError("Invalid balancing method")

    X_res, y_res = sampler.fit_resample(X, y)

    balanced_df = pd.concat(
        [pd.DataFrame(X_res, columns=X.columns),
         pd.Series(y_res, name=target_col)],
        axis=1
    )

    print("After balancing:")
    print(balanced_df[target_col].value_counts(normalize=True) * 100)

    return balanced_df

# =========================================================
# DATA QUALITY SUMMARY (ADDED)
# =========================================================

def count_missing_values(df):
    """
    Count total missing values in the dataset.
    """
    return int(df.isna().sum().sum())


def count_duplicates(df):
    """
    Count duplicate rows in the dataset.
    """
    return int(df.duplicated().sum())


def count_iqr_outliers(df):
    """
    Count outliers using IQR method for numeric columns.
    Returns total outliers and per-column breakdown.
    """
    total_outliers = 0
    outlier_by_col = {}

    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = int(((s < lower) | (s > upper)).sum())

        outlier_by_col[col] = n_outliers
        total_outliers += n_outliers

    return total_outliers, outlier_by_col


def print_data_quality_summary(df, title="DATA QUALITY SUMMARY"):
    """
    Print missing values, duplicates, and outliers.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    missing = count_missing_values(df)
    duplicates = count_duplicates(df)
    total_outliers, outlier_by_col = count_iqr_outliers(df)

    print(f"Missing values: {missing}")
    print(f"Duplicate rows: {duplicates}")
    print(f"Total outliers: {total_outliers}")

    print("\nOutliers by column:")
    for col, n in outlier_by_col.items():
        if n > 0:
            print(f"  - {col}: {n}")







