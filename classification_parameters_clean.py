import pandas as pd
import numpy as np
from functools import reduce
from collections import defaultdict

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, precision_score, confusion_matrix
from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# -------------------------------
# DATA SETTINGS
# -------------------------------
STRAINS = ["A1", "A15"]      # One strain: ["A1"]
MODE = "N_vs_P"             # Options: "M_vs_S" or "N_vs_P"
BASE_PATH = "parameter/"
OUTPUT = f"classification_result/classification_{MODE}_{'_'.join(STRAINS)}.csv"


# -------------------------------
# MODEL SETTINGS
# -------------------------------
USE_ENSEMBLE = False         # True = VotingClassifier, False = single models

# -------------------------------
# FEATURE SELECTION SETTINGS
# -------------------------------
USE_FEATURE_SELECTION = False  # True = use SelectKBest
FIXED_K = 6                 # Used if K_SEARCH = False
K_SEARCH = True            # True = optimize K automatically
K_RANGE = range(2, 5)
K_OUTPUT_FILE = f"classification_result/k_search_{MODE}_{'_'.join(STRAINS)}.csv"
FEATURE_FILE = f"classification_result/selected_features_{MODE}_{'_'.join(STRAINS)}.csv"
PLOT_FILE = f"classification_result/k_search_plot_{MODE}_{'_'.join(STRAINS)}.png"     # plot K vs accuracy graph

# -------------------------------
# CROSS VALIDATION SETTINGS
# -------------------------------
N_REPEATS = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def load_data(strains, mode="M_vs_S", base_path="parameter/"):
    """
    Load and merge feature datasets from one or multiple bacterial strains.

    This function reads Excel files containing extracted parameter features
    for each strain. Each strain file is expected to contain two sheets:
    "Replicate 1" and "Replicate 2". The replicate sheets are stacked together,
    patient/group labels are extracted, and strain-specific feature names are
    prefixed to avoid collisions when merging multiple strains.

    The final output is a single combined feature matrix `X`, class labels `y`,
    and grouping labels (`groups`) for grouped cross-validation.

    Parameters
    ----------
    strains : str or list of str
        The strain(s) to load.
        
        - If a single string is provided (e.g., "A1"), the function loads
          only that strain.
        - If a list is provided (e.g., ["A1", "A15"]), the function loads
          all strains and merges them into one dataset.

    mode : str, default="M_vs_S"
        Classification task definition. Controls which groups are included
        and how labels are encoded.

        Supported modes:
        - "M_vs_S":
            Keeps only samples belonging to groups "M" and "S".
            Encodes:
                M → 0
                S → 1

        - "N_vs_P":
            Keeps samples belonging to groups "N", "M", and "S".
            Encodes:
                N → 0
                M/S → 1

    base_path : str, default="parameter/"
        Directory containing the Excel parameter files.
        Each file must follow the naming convention:

            "{strain} - parameters - final.xlsx"

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix containing all extracted parameters.
        Feature names are prefixed with the strain name, e.g.:

            A1_Feature1, A1_Feature2, A15_Feature1, ...

    y : pandas.Series
        Encoded class labels (binary classification output).

    groups : pandas.Series
        Patient identifiers extracted from the sample name.
        Used for grouped cross-validation to ensure no patient leakage.

    Notes
    -----
    Expected structure of the first column in the Excel sheet:
        - Begins with group letter: S, M, or N
        - Followed by patient number, e.g.:

            S12_rep1, M03_rep2, N05_sampleX

    Example
    -------
    Load a single strain:

    >>> X, y, groups = load_data("A1", mode="M_vs_S")

    Load multiple strains merged together:

    >>> X, y, groups = load_data(["A1", "A15"], mode="N_vs_P")
    """

    if isinstance(strains, str):
        strains = [strains]

    strain_dfs = []

    for strain in strains:
        path = f"{base_path}{strain} - parameters - final.xlsx"

        df1 = pd.read_excel(path, sheet_name="Replicate 1")
        df2 = pd.read_excel(path, sheet_name="Replicate 2")
        df = pd.concat([df1, df2], ignore_index=True)

        first_col = df.columns[0]

        df["Patient"] = df[first_col].astype(str).str.extract(r'^([SMN]\d+)')[0]
        df["Group"] = df[first_col].astype(str).str[0]

        exclude = ["Patient", "Group", first_col]
        rename_dict = {
            c: f"{strain}_{c}" for c in df.columns if c not in exclude
        }

        df = df[["Patient", "Group"] + list(rename_dict.keys())]
        df = df.rename(columns=rename_dict)

        strain_dfs.append(df)


    df_final = reduce(
        lambda left, right: pd.merge(left, right, on=["Patient", "Group"]),
        strain_dfs
    )


    if mode == "M_vs_S":
        df_final = df_final[df_final["Group"].isin(["M", "S"])]
        df_final["GroupCode"] = df_final["Group"].map({"M": 0, "S": 1})

    elif mode == "N_vs_P":
        df_final = df_final[df_final["Group"].isin(["N", "M", "S"])]
        df_final["GroupCode"] = df_final["Group"].map({"N": 0, "M": 1, "S": 1})

    X = df_final.drop(columns=["Patient", "Group", "GroupCode"])
    y = df_final["GroupCode"]
    groups = df_final["Patient"]

    return X, y, groups


def macro_specificity(y_true, y_pred):
    """
    Compute the macro-averaged specificity score for classification.

    Specificity (also called the True Negative Rate) measures how well
    a classifier identifies negative samples correctly.

    For each class, specificity is defined as:

        Specificity_i = TN_i / (TN_i + FP_i)

    where:
        - TN_i = True Negatives for class i
        - FP_i = False Positives for class i

    This function computes specificity independently for each class
    using the confusion matrix, then returns the average across all
    classes (macro-average).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (true) class labels.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels produced by the classifier.

    Returns
    -------
    float
        Macro-averaged specificity score.

        - Range: [0, 1]
        - 1.0 indicates perfect specificity (no false positives)
        - Lower values indicate more false positive errors

    Notes
    -----
    Macro-specificity is useful in imbalanced classification problems,
    since it gives equal weight to each class rather than being dominated
    by the majority class.
    """    
    cm = confusion_matrix(y_true, y_pred)
    specs = []

    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp))

    return np.mean(specs)


# PARAMETER GRID FOR MODELS
MODELS = {
    "SVM": (
        SVC(class_weight="balanced", probability=True),
        {"clf__C": [0.1, 1, 10]}
    ),

    "LogReg": (
        LogisticRegression(max_iter=500, class_weight="balanced"),
        {"clf__C": [0.01, 0.1, 1, 10]}
    ),

    "XGBoost": (
        XGBClassifier(eval_metric="logloss"),
        {"clf__n_estimators": [100, 300]}
    )
}


def run_experiment(X, 
                   y, 
                   groups, 
                   k_features=None,
                    save_csv=True,
                    output_file=OUTPUT
    ):
    """
    Run repeated grouped cross-validation experiments and report
    aggregated performance results for each individual model and
    optionally for a Voting Ensemble.

    This function evaluates all models defined in the global MODELS
    dictionary under a repeated GroupShuffleSplit scheme.

    For every repeat:

    - A patient-level train/test split is performed
    - Each model is tuned using GridSearchCV with GroupKFold
    - Each tuned model is evaluated separately on the test set
    - If USE_ENSEMBLE=True, a soft VotingClassifier is built from the
      tuned models and evaluated as well

    At the end, the function returns a DataFrame summarizing mean and
    standard deviation of performance metrics for:

    - Each individual model
    - The VotingEnsemble (if enabled)

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix of shape (n_samples, n_features).

    y : pandas.Series or array-like
        Target labels.

    groups : pandas.Series or array-like
        Patient/group IDs used to prevent leakage in splitting.

    k_features : int or None, default=None
        If provided, enables SelectKBest feature selection with the top
        `k_features` features inside each training fold.

    Returns
    -------
    pandas.DataFrame
        Aggregated results table with rows = models and columns = metrics:

        - Acc_Mean, Acc_Std
        - Prec_Mean
        - Spec_Mean

        Includes VotingEnsemble only if USE_ENSEMBLE=True.

    Notes
    -----
    If USE_ENSEMBLE=False, ensemble results are not computed but a message
    is printed indicating ensemble voting was skipped.
    """

    rng = np.random.RandomState(RANDOM_STATE)

    model_names = list(MODELS.keys())

    if USE_ENSEMBLE:
        model_names.append("VotingEnsemble")

    all_metrics = {name: defaultdict(list) for name in model_names}

    for repeat in range(N_REPEATS):

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=rng.randint(0, 10000)
        )

        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        tuned_estimators = []

        for name, (model, grid_params) in MODELS.items():

            steps = [("scaler", StandardScaler())]

            if k_features is not None:
                steps.append(("selector", SelectKBest(f_classif, k=k_features)))

            steps.append(("clf", model))

            pipe = Pipeline(steps)

            gkf = GroupKFold(n_splits=3)

            grid = GridSearchCV(
                pipe,
                grid_params,
                scoring="balanced_accuracy",
                cv=gkf.split(X_train, y_train, groups_train),
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            tuned_estimators.append((name, best_model))

            y_pred = best_model.predict(X_test)

            all_metrics[name]["Acc"].append(
                balanced_accuracy_score(y_test, y_pred)
            )
            all_metrics[name]["Prec"].append(
                precision_score(y_test, y_pred, average="macro", zero_division=0)
            )
            all_metrics[name]["Spec"].append(
                macro_specificity(y_test, y_pred)
            )


        if USE_ENSEMBLE:
            ensemble = VotingClassifier(
                estimators=tuned_estimators,
                voting="soft"
            )

            ensemble.fit(X_train, y_train)

            y_pred_ens = ensemble.predict(X_test)

            all_metrics["VotingEnsemble"]["Acc"].append(
                balanced_accuracy_score(y_test, y_pred_ens)
            )
            all_metrics["VotingEnsemble"]["Prec"].append(
                precision_score(y_test, y_pred_ens, average="macro", zero_division=0)
            )
            all_metrics["VotingEnsemble"]["Spec"].append(
                macro_specificity(y_test, y_pred_ens)
            )

    final_results = {}

    for name in all_metrics:

        final_results[name] = {
            "Acc_Mean": np.mean(all_metrics[name]["Acc"]),
            "Acc_Std": np.std(all_metrics[name]["Acc"]),
            "Prec_Mean": np.mean(all_metrics[name]["Prec"]),
            "Prec_Std": np.std(all_metrics[name]["Prec"]),
            "Spec_Mean": np.mean(all_metrics[name]["Spec"]),
            "Spec_Std": np.std(all_metrics[name]["Spec"]),
        }

    results_df = pd.DataFrame(final_results).T

    if save_csv:
        results_df.to_csv(output_file)
        print(f"\n[SAVED] Results table exported to: {output_file}")

    if not USE_ENSEMBLE:
        print("\n[INFO] Voting Ensemble was disabled → only individual models reported.")

    return results_df


def k_search(
    X, y, groups,
    save_csv=True,
    history_file=K_OUTPUT_FILE,
    features_file=FEATURE_FILE,
    plot_file=PLOT_FILE
):
    """
    Search for the optimal number of selected features (K) that maximizes
    balanced accuracy.

    This function loops over values in K_RANGE and runs the full
    repeated grouped cross-validation experiment using SelectKBest(k=K).

    The best K is selected based on:

    - VotingEnsemble accuracy (if USE_ENSEMBLE=True)
    - Best single-model accuracy (if USE_ENSEMBLE=False)

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target labels.

    groups : pandas.Series
        Group/patient labels for grouped CV.

    save_csv : bool, default=True
        If True, saves the full K-search table.

    output_file : str, default="k_search_results.csv"
        CSV filename for saving K optimization results.

    Returns
    -------
    best_k : int
        Feature count K with the highest balanced accuracy.
    """

    best_k_per_model = {}
    accuracy_history = []
    feature_history = []

    # Loop over K values
    for k in K_RANGE:
        print(f"\nTesting K = {k}...")

        # Run experiment with feature selection
        results_df = run_experiment(
            X, y, groups,
            k_features=k,
            save_csv=False,
            output_file=None
        )

        # Loop over models
        for model_name in results_df.index:

            # -------------------------------
            # Record Accuracy
            # -------------------------------
            acc = results_df.loc[model_name, "Acc_Mean"]
            accuracy_history.append({
                "K": k,
                "Model": model_name,
                "Accuracy": acc
            })

            # -------------------------------
            # Update Best K per Model
            # -------------------------------
            if model_name not in best_k_per_model or acc > best_k_per_model[model_name][1]:
                best_k_per_model[model_name] = (k, acc)

            # -------------------------------
            # Extract Selected Features
            # -------------------------------
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            feature_history.append({
                "K": k,
                "Model": model_name,
                "Selected_Features": ", ".join(selected_features)
            })

    # -------------------------------
    # Convert to DataFrames
    # -------------------------------
    accuracy_df = pd.DataFrame(accuracy_history)
    features_df = pd.DataFrame(feature_history)

    # -------------------------------
    # Save CSVs
    # -------------------------------
    if save_csv:
        accuracy_df.to_csv(history_file, index=False)
        features_df.to_csv(features_file, index=False)
        print(f"\n[SAVED] Accuracy table: {history_file}")
        print(f"[SAVED] Selected features table: {features_file}")


    plt.figure(figsize=(10, 6))
    for model_name in accuracy_df["Model"].unique():
        subset = accuracy_df[accuracy_df["Model"] == model_name]
        plt.plot(subset["K"], subset["Accuracy"], marker="o", label=model_name, linewidth=2)

    plt.xlabel("Number of Selected Features (K)")
    plt.ylabel("Balanced Accuracy (Mean)")
    plt.title("K Optimization Per Model")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(list(K_RANGE))
    plt.legend(title="Models")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.show()
    print(f"[SAVED] K vs Accuracy plot: {plot_file}")

    print("\nBest K per model:")
    for model, (k, acc) in best_k_per_model.items():
        print(f"{model}: K={k}, Accuracy={acc:.4f}")

    return best_k_per_model


# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":

    X, y, groups = load_data(STRAINS, mode=MODE)

    print("Samples:", len(X))
    print("Features:", X.shape[1])

    if USE_FEATURE_SELECTION:

        if K_SEARCH:
            best_k = k_search(X, y, groups)
        else:
            best_k = FIXED_K

        final_result = best_k

    else:
        final_result = run_experiment(X, y, groups)

    print("\nFINAL RESULT:")
    print(final_result)
