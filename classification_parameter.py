import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import balanced_accuracy_score, precision_score, confusion_matrix

from xgboost import XGBClassifier
from collections import defaultdict

strain = "A1" 
file_path = f"parameter/{strain} - parameters - final.xlsx"
mode = "M_vs_S"  # or "M_vs_S"
result_name = f"classification_result/Traditional_ML_{strain}_{mode.strip()}.csv"
sheet1_name = "Replicate 1"
sheet2_name = "Replicate 2"

df1 = pd.read_excel(file_path, sheet_name=sheet1_name)
df2 = pd.read_excel(file_path, sheet_name=sheet2_name)

df1["Replicate"] = 1
df2["Replicate"] = 2

df = pd.concat([df1, df2], ignore_index=True)
columns = df.columns.tolist()[1:]

first_col = df.columns[0]

df["Group"] = df[first_col].astype(str).str[0]

df["Patient"] = (
    df[first_col]
    .astype(str)
    .str.extract(r'^([SMN]\d+)')[0]  
)

cols = df.columns.tolist()
new_order = [first_col, "Group", "Patient", "Replicate"] + \
            [c for c in cols if c not in [first_col, "Group", "Patient", "Replicate"]]

df = df[new_order]

def code_groups(df, mode="N_vs_P"):
    df = df.copy()
    
    if mode == "N_vs_P":
        df["GroupCode"] = df["Group"].map(lambda g: 0 if g == "N" else 1)
    
    elif mode == "M_vs_S":
        df = df[df["Group"].isin(["M", "S"])].copy()
        df["GroupCode"] = df["Group"].map({"M": 0, "S": 1})
    
    else:
        raise ValueError("mode must be 'N_vs_P' or 'M_vs_S'")
    
    return df

df = code_groups(df,mode = mode)

X = df[columns]
y = df["GroupCode"]
groups = df['Patient']

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
        SVC(class_weight="balanced"),
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

result =  run_repeated_grouped_cv(
    X=X,
    y=y,
    groups=groups,
    models=models,
    n_repeats=50,
    test_size=0.2,
    random_state=42
)

result.to_csv(result_name)








    

