# Heart Classifier Optimization

Comparable logistic and SGD objectives with train-only preprocessing.

**Author and maintainer:** Abhinav Tanwar.

## Run

Requires Python 3.12. Install dependencies with `python -m pip install -r requirements.txt`.

```sh
python analysis.py
python -m unittest discover -s tests
```

## Methods and limitations

Uses the supplied 500-row heart CSV with source target=1 as the positive label. The scaler and imputer are fitted within training cross-validation. Logistic regression and SGD log-loss, modified-Huber, and elastic-net variants share one stratified held-out set. Modified-Huber is a different loss, not an identical stochastic implementation of logistic likelihood. Results are an exploratory classifier benchmark and do not establish clinical safety or causation.

## Measured results

- **Logistic:** roc auc = 0.9183; average precision = 0.9656; recall = 0.907; precision = 0.8764; accuracy = 0.848.
- **SGD log loss:** roc auc = 0.8831; average precision = 0.9504; recall = 0.8605; precision = 0.8605; accuracy = 0.808.
- **SGD modified huber:** roc auc = 0.9101; average precision = 0.9546; recall = 0.8721; precision = 0.9146; accuracy = 0.856.
- **SGD elastic net:** roc auc = 0.9258; average precision = 0.969; recall = 0.8953; precision = 0.9059; accuracy = 0.864.

[Full calculated metrics](results/metrics.json) · [Input provenance](data/README.md) · [Method changes](docs/methodology.md). Open `report.html` for the visual summary.

The clean notebook in `notebooks/` delegates to the same module. Notebook cells were syntax-checked with outputs cleared; native scripts and tests were executed.
