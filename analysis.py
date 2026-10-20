import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from utils import ROOT, save, clf


def run():
    d = pd.read_csv(ROOT / "data/input.csv").drop_duplicates()
    y = d.pop("target").astype(int)
    a, b = train_test_split(
        np.arange(len(d)), test_size=0.25, random_state=24, stratify=y
    )
    cv = StratifiedKFold(4, shuffle=True, random_state=24)
    models = {
        "Logistic": LogisticRegression(max_iter=5000),
        "SGD_log_loss": SGDClassifier(
            loss="log_loss", max_iter=10000, tol=1e-6, random_state=24
        ),
        "SGD_modified_huber": SGDClassifier(
            loss="modified_huber", max_iter=10000, tol=1e-6, random_state=24
        ),
        "SGD_elastic_net": SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=0.5,
            max_iter=10000,
            tol=1e-6,
            random_state=24,
        ),
    }
    result = dict(
        rows=len(d),
        train_rows=len(a),
        test_rows=len(b),
        test_positive=int(y.iloc[b].sum()),
        positive_class="source target = 1",
        models={},
    )
    for name, model in models.items():
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", model),
            ]
        )
        grid = (
            {"model__C": [0.1, 1, 10]}
            if name == "Logistic"
            else {"model__alpha": [0.0001, 0.001, 0.01]}
        )
        fit = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc").fit(
            d.iloc[a], y.iloc[a]
        )
        prob = fit.predict_proba(d.iloc[b])[:, 1]
        result["models"][name] = {
            **clf(y.iloc[b], prob >= 0.5, prob),
            "selected": fit.best_params_,
        }
    return save(result)


if __name__ == "__main__":
    run()
