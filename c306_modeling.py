#c306_modeling

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  MinMaxScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from scipy.special import logsumexp
from sklearn.metrics import roc_curve
from sklearn.tree import plot_tree

RANDOM_STATE = 42


# =========================================================
# HELPERS
# =========================================================
def _to_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.copy()
    return pd.Series(y)


def _resolve_positive_label(y) -> object | None:
    """
    Return a reasonable positive label for binary classification.
    """
    y = _to_series(y).dropna()
    unique = list(pd.unique(y))

    if len(unique) != 2:
        return None

    preferred = ["Yes", "yes", 1, "1", True]
    for p in preferred:
        if p in unique:
            return p

    return unique[1]


def _make_scorer_for_y(y_train):
    pos_label = _resolve_positive_label(y_train)
    if pos_label is None:
        return make_scorer(f1_score, average="macro"), None
    return make_scorer(f1_score, pos_label=pos_label, zero_division=0), pos_label


# =========================================================
# CONFUSION MATRIX PLOT
# =========================================================
def plot_confusion_matrix(y_true, y_pred, title: str) -> None:
    """
    Plot a confusion matrix with real class labels.
    """
    y_true_s = _to_series(y_true)
    y_pred_s = _to_series(y_pred)

    labels = list(pd.unique(pd.concat([y_true_s.astype(object), y_pred_s.astype(object)], ignore_index=True)))
    cm = confusion_matrix(y_true_s, y_pred_s, labels=labels)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="black" if cm[i, j] > thresh else "white",
            )

    plt.tight_layout()
    plt.show()


# =========================================================
# ROC CURVE PLOT
# =========================================================
def plot_roc_curve(model, X_test, y_test, title="ROC Curve"):
    y_test = _to_series(y_test)

    positive_label = _resolve_positive_label(y_test)

    if not hasattr(model, "predict_proba") or positive_label is None:
        print("ROC curve cannot be plotted (no predict_proba or not binary).")
        return

    classes = list(model.classes_)
    if positive_label not in classes:
        print("Positive label not found in classes.")
        return

    pos_idx = classes.index(positive_label)
    y_score = model.predict_proba(X_test)[:, pos_idx]

    y_test_bin = (y_test == positive_label).astype(int)

    fpr, tpr, _ = roc_curve(y_test_bin, y_score)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid()
    plt.show()

# =========================================================
# MODEL EVALUATION
# =========================================================
def evaluate_model(model, X_test, y_test, title: str) -> dict:
    """
    Evaluate a fitted model on the untouched test set.
    Works for binary labels like Yes/No, 0/1, etc.
    """
    y_test = _to_series(y_test)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)

    positive_label = _resolve_positive_label(y_test)

    if hasattr(model, "predict_proba") and hasattr(model, "classes_") and positive_label is not None:
        classes = list(model.classes_)
        if positive_label in classes:
            pos_idx = classes.index(positive_label)
            y_score = model.predict_proba(X_test)[:, pos_idx]
        else:
            y_score = None
    else:
        y_score = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
    }

    if positive_label is not None and len(pd.unique(y_test)) == 2:
        metrics["Precision"] = precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
        metrics["Recall"] = recall_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
        metrics["F1-score"] = f1_score(y_test, y_pred, pos_label=positive_label, zero_division=0)

        if y_score is not None:
            y_test_bin = (y_test == positive_label).astype(int)
            metrics["ROC-AUC"] = roc_auc_score(y_test_bin, y_score)
            metrics["PR-AUC"] = average_precision_score(y_test_bin, y_score)
    else:
        metrics["Precision"] = precision_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["Recall"] = recall_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["F1-score"] = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n==================== {title} ====================")
    for name, value in metrics.items():
        print(f"{name:18s}: {value:.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {title}")
    plot_roc_curve(model, X_test, y_test, f"ROC Curve - {title}")
    return metrics


# =========================================================
# SAFE CROSS-VALIDATION
# =========================================================
def get_cv(y_train: pd.Series) -> StratifiedKFold:
    """
    Create a safe stratified CV object.
    """
    y_train = _to_series(y_train)
    min_class_count = int(y_train.value_counts().min())

    if min_class_count < 2:
        raise ValueError(
            "Not enough samples in the smallest class to run stratified cross-validation."
        )

    n_splits = min(3, min_class_count)

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# =========================================================
#  MIXED KNN + NAIVE BAYES
# =========================================================

# =========================================================
# MIXED KNN
# =========================================================
class MixedKNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X = pd.DataFrame(X)

        self.num_cols = X.select_dtypes(include=np.number).columns
        self.cat_cols = X.select_dtypes(exclude=np.number).columns

        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()

        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        # Numeric
        if len(self.num_cols):
            X_num = self.scaler.fit_transform(
                self.num_imputer.fit_transform(X[self.num_cols])
            )
        else:
            X_num = np.empty((len(X), 0))

        # Categorical
        if len(self.cat_cols):
            X_cat = self.encoder.fit_transform(
                self.cat_imputer.fit_transform(X[self.cat_cols])
            )
        else:
            X_cat = np.empty((len(X), 0))

        self.n_num = X_num.shape[1]

        X_train = np.hstack([X_num, X_cat])

        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self._mixed_distance,
            algorithm="brute"
        )

        self.knn.fit(X_train, y)

        self.classes_ = np.unique(y)

        return self

    def _mixed_distance(self, u, v):
        num_dist = np.sum((u[:self.n_num] - v[:self.n_num]) ** 2)
        cat_dist = np.sum(u[self.n_num:] != v[self.n_num:])
        return np.sqrt(num_dist + cat_dist)

    def _transform(self, X):
        X = pd.DataFrame(X)

        if len(self.num_cols):
            X_num = self.scaler.transform(
                self.num_imputer.transform(X[self.num_cols])
            )
        else:
            X_num = np.empty((len(X), 0))

        if len(self.cat_cols):
            X_cat = self.encoder.transform(
                self.cat_imputer.transform(X[self.cat_cols])
            )
        else:
            X_cat = np.empty((len(X), 0))

        return np.hstack([X_num, X_cat])

    def predict(self, X):
        return self.knn.predict(self._transform(X))

    def predict_proba(self, X):
        return self.knn.predict_proba(self._transform(X))


# =========================================================
# MIXED NAIVE BAYES
# =========================================================
class MixedNaiveBayesClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, var_smoothing=1e-9, alpha=1.0):
        self.var_smoothing = var_smoothing
        self.alpha = alpha

    def fit(self, X, y):
        X = pd.DataFrame(X)

        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = [c for c in X.columns if c not in self.num_cols]

        # Imputers
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")

        # Encoder
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        # ---------- NUMERIC ----------
        if len(self.num_cols) > 0:
            X_num = self.num_imputer.fit_transform(X[self.num_cols])
            self.gnb = GaussianNB(var_smoothing=self.var_smoothing)
            self.gnb.fit(X_num, y)

        # ---------- CATEGORICAL ----------
        if len(self.cat_cols) > 0:
            X_cat = self.encoder.fit_transform(
                self.cat_imputer.fit_transform(X[self.cat_cols])
            )

            X_cat = np.round(X_cat).astype(np.int64)
            X_cat = X_cat + 1   # avoid -1

            self.cnb = CategoricalNB(alpha=self.alpha)
            self.cnb.fit(X_cat, y)

        # Required by sklearn
        self.classes_ = np.unique(y)

        return self

    def _joint_log_proba(self, X):
        X = pd.DataFrame(X)
        log_prob = None

        # ---------- NUMERIC ----------
        if len(self.num_cols) > 0:
            X_num = self.num_imputer.transform(X[self.num_cols])
            num_log = self.gnb._joint_log_likelihood(X_num)
            log_prob = num_log

        # ---------- CATEGORICAL ----------
        if len(self.cat_cols) > 0:
            X_cat = self.encoder.transform(
                self.cat_imputer.transform(X[self.cat_cols])
            )

            X_cat = np.round(X_cat).astype(np.int64)
            X_cat = X_cat + 1

            cat_log = self.cnb._joint_log_likelihood(X_cat)

            if log_prob is None:
                log_prob = cat_log
            else:
                log_prob += cat_log

        return log_prob

    def predict(self, X):
        log_prob = self._joint_log_proba(X)
        return self.classes_[np.argmax(log_prob, axis=1)]

    def predict_proba(self, X):
        log_prob = self._joint_log_proba(X)
        return np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True))

# =========================================================
# GRID SEARCH
# =========================================================

def tune_knn(X_train, y_train):
    model = MixedKNNClassifier()

    param_grid = {
        "n_neighbors": [3, 5, 7]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=2,
        scoring="accuracy",
        n_jobs=1
    )

    grid.fit(X_train, y_train)
    return grid


def tune_naive_bayes(X_train, y_train):
    model = MixedNaiveBayesClassifier()

    param_grid = {
        "var_smoothing": [1e-9, 1e-8, 1e-7],
        "alpha": [0.5, 1.0]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=2,
        scoring="accuracy",
        n_jobs=1
    )

    grid.fit(X_train, y_train)
    return grid


# =========================================================
# FIT + EVALUATE
# =========================================================
def fit_and_evaluate_model(grid_search: GridSearchCV, X_test, y_test, title: str) -> dict:
    """
    Fit time is measured by GridSearchCV internally + evaluation on test set.
    """
    fit_time = grid_search.refit_time_ if hasattr(grid_search, "refit_time_") else 0
    best_model = grid_search.best_estimator_

    metrics = evaluate_model(best_model, X_test, y_test, title)
    metrics["Training Time (sec)"] = fit_time
    return metrics


# =========================================================
# COMPARISON
# =========================================================
def print_model_parameters(knn_grid, nb_grid, stage=""):
    print(f"\n==================== MODEL PARAMETERS ({stage}) ====================\n")

    # -------- KNN --------
    best_knn = knn_grid.best_params_

    print("(KNN Model)")
    print(f"1. Best number of neighbors (k) = {best_knn.get('n_neighbors', 'N/A')}")
    print("2. Weight type used = uniform")
    print("3. Distance metric = custom mixed distance")

    # -------- NAIVE BAYES --------
    best_nb = nb_grid.best_params_

    print("(Naïve Bayes Model)")
    print("1. Type of NB used = MixedNaiveBayesClassifier")
    print(f"2. Best var_smoothing value = {best_nb.get('var_smoothing', 'N/A')}")
    print(f"3. Best alpha value = {best_nb.get('alpha', 'N/A')}")


def compare_knn_and_nb(
    X_train,
    y_train,
    X_test,
    y_test,
    balance_fn=None,
    target_col=None,
    balance_kwargs=None,
    show_before=True
) -> pd.DataFrame:
    """
    Compare KNN and Naive Bayes before and after balancing.
    """

    results = []

    # =========================
    # BEFORE BALANCING
    # =========================
    if show_before:
        print("\n===== BEFORE BALANCING =====")

        knn_grid_before = tune_knn(X_train, y_train)
        knn_metrics = evaluate_model(
            knn_grid_before.best_estimator_,
            X_test,
            y_test,
            "KNN (Before)"
        )
        knn_metrics["Model"] = "KNN (Before)"
        results.append(knn_metrics)

        nb_grid_before = tune_naive_bayes(X_train, y_train)
        nb_metrics = evaluate_model(
            nb_grid_before.best_estimator_,
            X_test,
            y_test,
            "Naive Bayes (Before)"
        )
        nb_metrics["Model"] = "Naive Bayes (Before)"
        results.append(nb_metrics)

        print_model_parameters(knn_grid_before, nb_grid_before, stage="Before Balancing")

    # =========================
    # AFTER BALANCING
    # =========================
    if balance_fn is not None and target_col is not None:

        print("\n===== AFTER BALANCING =====")

        train_df = pd.concat([X_train, y_train], axis=1)

        train_balanced = balance_fn(
            train_df,
            target_col=target_col,
            **(balance_kwargs or {})
        )

        X_train_bal = train_balanced.drop(columns=[target_col])
        y_train_bal = train_balanced[target_col]

        knn_grid_after = tune_knn(X_train_bal, y_train_bal)
        knn_metrics = evaluate_model(
            knn_grid_after.best_estimator_,
            X_test,
            y_test,
            "KNN (After)"
        )
        knn_metrics["Model"] = "KNN (After)"
        results.append(knn_metrics)

        nb_grid_after = tune_naive_bayes(X_train_bal, y_train_bal)
        nb_metrics = evaluate_model(
            nb_grid_after.best_estimator_,
            X_test,
            y_test,
            "Naive Bayes (After)"
        )
        nb_metrics["Model"] = "Naive Bayes (After)"
        results.append(nb_metrics)

        print_model_parameters(knn_grid_after, nb_grid_after, stage="After Balancing")

    # =========================
    # FINAL TABLE
    # =========================
    results_df = pd.DataFrame(results).set_index("Model")

    cols = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score"]
    print(results_df[cols].round(4))

    return results_df


#=======================================================================================

# =========================================================
# DECISION TREE + ENSEMBLE METHODS
# =========================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# =========================================================
# PLOT DECISION TREE
# =========================================================

def plot_decision_tree_model(
    tree_model,
    feature_names,
    class_names=None,
    title="Decision Tree",
    max_depth=3
):

    plt.figure(figsize=(22, 10))

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8
    )

    plt.title(title)

    plt.tight_layout()

    plt.show()

# --------------------------------------------------------------------

def _make_bagging(base_estimator=None, **kwargs):

    try:
        if base_estimator is None:
            return BaggingClassifier(**kwargs)

        return BaggingClassifier(
            estimator=base_estimator,
            **kwargs
        )

    except TypeError:

        return BaggingClassifier(
            base_estimator=base_estimator,
            **kwargs
        )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def feature_importance_table(model, feature_names):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance (%)": model.feature_importances_ * 100
    })

    importance_df = importance_df.sort_values(
        by="Importance (%)",
        ascending=False
    )

    return importance_df


def plot_feature_importance(df_imp, title):

    top = df_imp.head(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top["Feature"],
        top["Importance (%)"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance (%)")
    plt.title(title)

    plt.tight_layout()
    plt.show()


# =========================================================
# PARAMETER PRINT
# =========================================================

def print_best_params(search, stage):

    print(f"\n==================== {stage} PARAMETERS ====================")

    for k, v in search.best_params_.items():
        print(f"{k} = {v}")


# =========================================================
# DECISION TREE
# =========================================================

def tune_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=get_cv(y_train),
        scoring=_make_scorer_for_y(y_train)[0],
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    return grid


# =========================================================
# BAGGING
# =========================================================

def tune_bagging(X_train, y_train):

    model = _make_bagging(
        base_estimator=DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE
    )

    param_grid = {
        "n_estimators": [50, 100]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=get_cv(y_train),
        scoring=_make_scorer_for_y(y_train)[0],
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    return grid


# =========================================================
# RANDOM FOREST
# =========================================================

def tune_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    param_grid = {
        "n_estimators": [100],
        "max_depth": [None, 10]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=get_cv(y_train),
        scoring=_make_scorer_for_y(y_train)[0],
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    return grid


# =========================================================
# MAIN COMPARISON
# =========================================================

def compare_tree_and_ensemble(
    X_train,
    y_train,
    X_test,
    y_test,
    balance_fn=None,
    target_col=None,
    balance_kwargs=None,
    show_before=True
):

    results = []

    def run_models(X_tr, y_tr, label):

        # ======================
        # DECISION TREE
        # ======================

        tree_grid = tune_decision_tree(X_tr, y_tr)

        tree_model = tree_grid.best_estimator_
        plot_decision_tree_model(
            tree_model,
            feature_names=X_tr.columns,
            class_names=["Low", "High"],
            title=f"Decision Tree ({label})",
            max_depth=3
        )

        tree_metrics = evaluate_model(
            tree_model,
            X_test,
            y_test,
            f"Decision Tree ({label})"
        )

        tree_metrics["Model"] = f"Decision Tree ({label})"

        results.append(tree_metrics)

        print_best_params(
            tree_grid,
            f"Decision Tree ({label})"
        )

        # ======================
        # BAGGING
        # ======================

        bag_grid = tune_bagging(X_tr, y_tr)

        bag_model = bag_grid.best_estimator_

        bag_metrics = evaluate_model(
            bag_model,
            X_test,
            y_test,
            f"Bagging ({label})"
        )

        bag_metrics["Model"] = f"Bagging ({label})"

        results.append(bag_metrics)

        print_best_params(
            bag_grid,
            f"Bagging ({label})"
        )

        # ======================
        # RANDOM FOREST
        # ======================

        rf_grid = tune_random_forest(X_tr, y_tr)

        rf_model = rf_grid.best_estimator_

        rf_metrics = evaluate_model(
            rf_model,
            X_test,
            y_test,
            f"Random Forest ({label})"
        )

        rf_metrics["Model"] = f"Random Forest ({label})"

        results.append(rf_metrics)

        print_best_params(
            rf_grid,
            f"Random Forest ({label})"
        )

        # ======================
        # FEATURE IMPORTANCE
        # ======================

        feature_names = list(X_tr.columns)

        tree_imp = feature_importance_table(
            tree_model,
            feature_names
        )

        rf_imp = feature_importance_table(
            rf_model,
            feature_names
        )

        print("\n========== DECISION TREE FEATURE IMPORTANCE ==========")
        print(tree_imp.round(3).to_string(index=False))

        print("\n========== RANDOM FOREST FEATURE IMPORTANCE ==========")
        print(rf_imp.round(3).to_string(index=False))

        plot_feature_importance(
            tree_imp,
            f"Decision Tree Feature Importance ({label})"
        )

        plot_feature_importance(
            rf_imp,
            f"Random Forest Feature Importance ({label})"
        )

    # =====================================================
    # BEFORE BALANCING
    # =====================================================

    if show_before:

        print("\n===== BEFORE BALANCING =====")

        run_models(
            X_train,
            y_train,
            "Before"
        )

    # =====================================================
    # AFTER BALANCING
    # =====================================================

    if balance_fn is not None and target_col is not None:

        print("\n===== AFTER BALANCING =====")

        train_df = pd.concat(
            [X_train, y_train],
            axis=1
        )

        balanced_df = balance_fn(
            train_df,
            target_col=target_col,
            **(balance_kwargs or {})
        )

        X_bal = balanced_df.drop(columns=[target_col])

        y_bal = balanced_df[target_col]

        run_models(
            X_bal,
            y_bal,
            "After"
        )

    # =====================================================
    # FINAL TABLE
    # =====================================================

    results_df = pd.DataFrame(results)

    cols = [
        "Model",
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    print("\n================ FINAL TABLE ================")

    print(
        results_df[cols]
        .round(4)
        .to_string(index=False)
    )

    return results_df

