import pandas as pd
import numpy as np
from functools import reduce

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, precision_score, confusion_matrix

from xgboost import XGBClassifier
from collections import defaultdict

# strain = "A1" 
#  #  strains = ["A15", "A1", "A5"]
# file_path = f"parameter/{strain} - parameters - final.xlsx"
# mode = "M_vs_S"  # or "M_vs_S"
# result_name = f"classification_result/Traditional_ML_{strain}_{mode.strip()}.csv"
# sheet1_name = "Replicate 1"
# sheet2_name = "Replicate 2"

# df1 = pd.read_excel(file_path, sheet_name=sheet1_name)
# df2 = pd.read_excel(file_path, sheet_name=sheet2_name)

# df1["Replicate"] = 1
# df2["Replicate"] = 2

# df = pd.concat([df1, df2], ignore_index=True)
# columns = df.columns.tolist()[1:]

# first_col = df.columns[0]

# df["Group"] = df[first_col].astype(str).str[0]

# df["Patient"] = (
#     df[first_col]
#     .astype(str)
#     .str.extract(r'^([SMN]\d+)')[0]  
# )

# cols = df.columns.tolist()
# new_order = [first_col, "Group", "Patient", "Replicate"] + \
#             [c for c in cols if c not in [first_col, "Group", "Patient", "Replicate"]]

# df = df[new_order]

# def code_groups(df, mode="N_vs_P"):
#     df = df.copy()
    
#     if mode == "N_vs_P":
#         df["GroupCode"] = df["Group"].map(lambda g: 0 if g == "N" else 1)
    
#     elif mode == "M_vs_S":
#         df = df[df["Group"].isin(["M", "S"])].copy()
#         df["GroupCode"] = df["Group"].map({"M": 0, "S": 1})
    
#     else:
#         raise ValueError("mode must be 'N_vs_P' or 'M_vs_S'")
    
#     return df

# df = code_groups(df,mode = mode)

# X = df[columns]
# y = df["GroupCode"]
# groups = df['Patient']

def load_multi_strain_combinatorial(strains, mode="M_vs_S", base_path="parameter/"):
    strain_dfs = []
    if isinstance(strains, str):
        strains = [strains]

    for strain in strains:
        # 1. Stack R1 and R2 vertically for this strain
        path = f"{base_path}{strain} - parameters - final.xlsx"
        df1 = pd.read_excel(path, sheet_name="Replicate 1")
        df2 = pd.read_excel(path, sheet_name="Replicate 2")
        df_stacked = pd.concat([df1, df2], ignore_index=True)

        # 2. Extract IDs and Prefix Parameters
        first_col = df_stacked.columns[0]
        df_stacked["Patient"] = df_stacked[first_col].astype(str).str.extract(r'^([SMN]\d+)')[0]
        df_stacked["Group"] = df_stacked[first_col].astype(str).str[0]
        
        # Rename only measurements: e.g., 'A1_Param1'
        exclude = ["Patient", "Group", first_col]
        rename_dict = {c: f"{strain}_{c}" for c in df_stacked.columns if c not in exclude}
        
        # Keep only metadata and prefixed features
        df_clean = df_stacked[["Patient", "Group"] + list(rename_dict.keys())].rename(columns=rename_dict)
        strain_dfs.append(df_clean)

    # 3. MAGIC STEP: Combinatorial Merge across all Strains
    # 'on=["Patient", "Group"]' creates the Cartesian product for each strain added
    # 2 strains = 4 rows/patient, 3 strains = 8 rows/patient...
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=["Patient", "Group"]), strain_dfs)

    # 4. Mode Selection (M_vs_S or N_vs_P)
    if mode == "M_vs_S":
        df_merged = df_merged[df_merged["Group"].isin(["M", "S"])].copy()
        df_merged["GroupCode"] = df_merged["Group"].map({"M": 0, "S": 1})
    elif mode == "N_vs_P":
        df_merged = df_merged[df_merged["Group"].isin(["N", "M", "S"])].copy()
        df_merged["GroupCode"] = df_merged["Group"].map({"N": 0, "M": 1, "S": 1})

    y = df_merged["GroupCode"]
    groups = df_merged["Patient"]
    X = df_merged.drop(columns=["Patient", "Group", "GroupCode"])
    
    return X, y, groups


def macro_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    specs = []

    for i in range(n_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specs.append(spec)

    return np.mean(specs)


models = {
    "SVM": (
        SVC(class_weight="balanced", probability=True),
        {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", 0.01, 0.1],
            "clf__kernel": ["rbf"]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        ),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.01, 0.1],
            "clf__subsample": [0.8, 1.0]
        }
    ),

    "LogisticRegression": (
        LogisticRegression(max_iter=500, class_weight="balanced"),
        {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"]
        }
    )
}


def run_repeated_grouped_cv(
    X, y, groups,
    models,
    n_repeats=50,
    test_size=0.2,
    random_state=42
):
    results = {}

    rng = np.random.RandomState(random_state)

    for model_name, (model, param_grid) in models.items():
        print(f"\n=== Running model: {model_name} ===")

        metrics = defaultdict(list)

        for i in range(n_repeats):
            # --- grouped train/test split ---
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=rng.randint(0, 10_000)
            )

            train_idx, test_idx = next(gss.split(X, y, groups))

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups_train = groups.iloc[train_idx]

            # --- pipeline: scaling + classifier ---
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", model)
            ])

            # --- 3-fold grouped CV on training set ---
            gkf = GroupKFold(n_splits=3)

            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=gkf.split(X_train, y_train, groups_train),
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            # --- test prediction ---
            y_pred = best_model.predict(X_test)

            # --- metrics ---
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            macro_spec = macro_specificity(y_test, y_pred)

            metrics["balanced_accuracy"].append(bal_acc)
            metrics["macro_precision"].append(macro_prec)
            metrics["macro_specificity"].append(macro_spec)

        # --- aggregate ---
        results[model_name] = {
            "balanced_accuracy_mean": np.mean(metrics["balanced_accuracy"]),
            "balanced_accuracy_std":  np.std(metrics["balanced_accuracy"]),

            "macro_precision_mean": np.mean(metrics["macro_precision"]),
            "macro_precision_std":  np.std(metrics["macro_precision"]),

            "macro_specificity_mean": np.mean(metrics["macro_specificity"]),
            "macro_specificity_std":  np.std(metrics["macro_specificity"]),
        }

    return pd.DataFrame(results).T

def run_voting_ensemble(
    X, y, groups,
    models, # Expects SVM, XGB, and LogisticRegression in this dict
    n_repeats=50,
    test_size=0.2,
    random_state=42
):
    # Initialize results tracking for each model + the Ensemble
    model_names = list(models.keys()) + ["VotingEnsemble"]
    all_metrics = {name: defaultdict(list) for name in model_names}
    
    rng = np.random.RandomState(random_state)

    for i in range(n_repeats):
        print(f"Repeat {i+1}/{n_repeats}...", end="\r")
        
        # --- Grouped train/test split ---
        gss = GroupShuffleSplit(
            n_splits=1, 
            test_size=test_size, 
            random_state=rng.randint(0, 10_000)
        )
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        best_tuned_estimators = []

        # --- Phase 1: Tune Individual Models ---
        for model_name, (model, param_grid) in models.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", model)
            ])
            
            gkf = GroupKFold(n_splits=3)
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=gkf.split(X_train, y_train, groups_train),
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            # Store the best tuned estimator for this repeat
            best_est = grid.best_estimator_
            best_tuned_estimators.append((model_name, best_est))
            
            # Evaluate individual model
            y_pred = best_est.predict(X_test)
            all_metrics[model_name]["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))
            all_metrics[model_name]["macro_precision"].append(precision_score(y_test, y_pred, average="macro", zero_division=0))
            all_metrics[model_name]["macro_specificity"].append(macro_specificity(y_test, y_pred))

        # --- Phase 2: Create & Evaluate Voting Ensemble ---
        # 'soft' voting uses probabilities, which is usually more accurate for biological data
        # Ensure SVC has probability=True in your initial model dictionary
        ensemble = VotingClassifier(
            estimators=best_tuned_estimators, 
            voting='soft'
        )
        
        # No need to fit again if using 'prefit' logic, 
        # but sklearn's VotingClassifier needs a fit call on the train set
        ensemble.fit(X_train, y_train)
        
        y_pred_ens = ensemble.predict(X_test)
        all_metrics["VotingEnsemble"]["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred_ens))
        all_metrics["VotingEnsemble"]["macro_precision"].append(precision_score(y_test, y_pred_ens, average="macro", zero_division=0))
        all_metrics["VotingEnsemble"]["macro_specificity"].append(macro_specificity(y_test, y_pred_ens))

    # --- Phase 3: Aggregate Results ---
    final_results = {}
    for name in model_names:
        final_results[name] = {
            "Acc_Mean": np.mean(all_metrics[name]["balanced_accuracy"]),
            "Acc_Std":  np.std(all_metrics[name]["balanced_accuracy"]),
            "Prec_Mean": np.mean(all_metrics[name]["macro_precision"]),
            "Spec_Mean": np.mean(all_metrics[name]["macro_specificity"]),
        }

    return pd.DataFrame(final_results).T

def run_voting_ensemble_feature_selection(
    X, y, groups,
    models,
    n_repeats=30, # Reduced slightly for speed with Selection
    test_size=0.2,
    k_features=10, # Number of top features to keep
    random_state=42
):
    model_names = list(models.keys()) + ["VotingEnsemble"]
    all_metrics = {name: defaultdict(list) for name in model_names}
    rng = np.random.RandomState(random_state)

    for i in range(n_repeats):
        print(f"Repeat {i+1}/{n_repeats}...", end="\r")
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.randint(0, 10_000))
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        best_tuned_estimators = []

        for model_name, (model, param_grid) in models.items():
            # NEW: Added SelectKBest to the pipeline
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=k_features)),
                ("clf", model)
            ])
            
            gkf = GroupKFold(n_splits=3)
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=gkf.split(X_train, y_train, groups_train),
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            best_est = grid.best_estimator_
            best_tuned_estimators.append((model_name, best_est))
            
            # Predict and Score
            y_pred = best_est.predict(X_test)
            all_metrics[model_name]["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))

        # Ensemble using the pipelines that now include feature selection
        ensemble = VotingClassifier(estimators=best_tuned_estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        y_pred_ens = ensemble.predict(X_test)
        all_metrics["VotingEnsemble"]["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred_ens))

    # Aggregate results
    results = {name: {"Mean_Acc": np.mean(all_metrics[name]["balanced_accuracy"]), 
                      "Std": np.std(all_metrics[name]["balanced_accuracy"])} 
               for name in model_names}
    return pd.DataFrame(results).T

strains_to_use = ["A1","A15"]  # "A0","A1","A15","A19","A28","A5"
# current_mode = "M_vs_S" # "N_vs_P" or "M_vs_S"
# for strain in strains_to_use:
#     print(f"Using strain: {strain}")
#     result_name = f"classification_result/k_range_{strain}_{current_mode.strip()}.csv"
#     X, y, group = load_multi_strain_combinatorial(strain, mode=current_mode, base_path="parameter/")
#     result = run_repeated_grouped_cv(
#         X, y, group,
#         models,
#         n_repeats=50,
#         test_size=0.2,
#         random_state=42
#     )
#     print(f"Result for {strain}: {result}")
#     result.to_csv(result_name)

# print(f"Task: {current_mode}")
# print(f"Samples: {len(X)}, Features: {len(X.columns)}")

# k_range = range(2, 15)
# k_results = []
# feature_importance_history = {} # Key: K, Value: List of feature names selected

# for k in k_range:
#     print(f"Analyzing K = {k}...")
    
#     # Track features for this specific K across all repeats
#     all_selected_features_at_k = []
    
#     # Run your ensemble function
#     # Note: You'll need to modify your function to RETURN the selected features
#     # Or, we can extract them by running a single fit for the log below:
    
#     df_step = run_voting_ensemble_feature_selection(
#         X=X, y=y, groups=patient_groups, models=models, 
#         n_repeats=20, k_features=k
#     )
    
#     # --- Capture Feature Names for this K ---
#     # We do a 'Global' fit on the whole dataset to see which K features 
#     # are statistically strongest for this K value.
#     selector = SelectKBest(score_func=f_classif, k=k)
#     selector.fit(X, y)
#     feature_importance_history[k] = X.columns[selector.get_support()].tolist()

#     df_step['K'] = k
#     k_results.append(df_step)

# features_df = pd.DataFrame.from_dict(feature_importance_history, orient='index')
# features_df.to_csv("top_features_by_k.csv")

# full_k_df = pd.concat(k_results).reset_index()

# import matplotlib.pyplot as plt
# # 2. Rename that new column to 'ModelName' so the rest of the code works
# # Usually reset_index() names the old index 'index' or 'level_0'
# full_k_df = full_k_df.rename(columns={'index': 'ModelName', 'level_0': 'ModelName'})

# # 3. Now the plotting loop will work
# plt.figure(figsize=(10, 6))

# for model_name in full_k_df['ModelName'].unique():
#     subset = full_k_df[full_k_df['ModelName'] == model_name]
    
#     # Sort by K to ensure the line plots correctly from left to right
#     subset = subset.sort_values('K')
    
#     plt.plot(subset['K'], subset['Mean_Acc'], marker='o', label=model_name, linewidth=2)
    
#     # Optional: Shaded error bars
#     if 'Std' in subset.columns:
#         plt.fill_between(
#             subset['K'], 
#             subset['Mean_Acc'] - subset['Std'], 
#             subset['Mean_Acc'] + subset['Std'], 
#             alpha=0.1
#         )

# plt.title('Optimization of Feature Count (K) Accuracy', fontsize=14)
# plt.xlabel('Number of Selected Features (K)', fontsize=12)
# plt.ylabel('Balanced Accuracy (Mean)', fontsize=12)
# plt.xticks(list(k_range))
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# plt.savefig('k_optimization_plot.png')

# result = run_voting_ensemble_feature_selection(
#     X=X,
#     y=y,
#     groups=patient_groups,
#     models=models,
#     n_repeats=50,
#     test_size=0.2,
#     random_state=42,
#     k_features=4
# )
# print(result)

# result.to_csv(result_name)

# k_range = range(2, 15)
# k_results = []

# print("Starting K-Optimization Search...")

# for k in k_range:
#     print(f"Testing K = {k}...")
    
#     # Run your ensemble function for this specific K
#     df_step = run_voting_ensemble_feature_selection(
#         X=X, 
#         y=y, 
#         groups=patient_groups, 
#         models=models, 
#         n_repeats=20, # Reduced repeats for the search phase
#         test_size=0.2, 
#         random_state=42, 
#         k_features=k
#     )
    
#     # Add K as a column and save
#     df_step['K'] = k
#     df_step['ModelName'] = df_step.index
#     k_results.append(df_step)

# # Combine all results into one master DataFrame
# full_k_df = pd.concat(k_results, ignore_index=True)

# import matplotlib.pyplot as plt

# # 2. Plotting the results
# plt.figure(figsize=(10, 6))

# for model_name in full_k_df['ModelName'].unique():
#     subset = full_k_df[full_k_df['ModelName'] == model_name]
    
#     plt.plot(subset['K'], subset['Mean_Acc'], marker='o', label=model_name, linewidth=2)
    
#     # Optional: Add error bars (Shaded area for Standard Deviation)
#     plt.fill_between(
#         subset['K'], 
#         subset['Mean_Acc'] - subset['Std'], 
#         subset['Mean_Acc'] + subset['Std'], 
#         alpha=0.1
#     )

# plt.title('Optimization of Feature Count (K) Accuracy', fontsize=14)
# plt.xlabel('Number of Selected Features (K)', fontsize=12)
# plt.ylabel('Balanced Accuracy (Mean)', fontsize=12)
# plt.xticks(list(k_range))
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# plt.savefig('k_optimization_plot.png')








    

