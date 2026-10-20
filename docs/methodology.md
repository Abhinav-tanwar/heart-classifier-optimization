# Methodology and limitations

Uses the supplied 500-row heart CSV with source target=1 as the positive label. The scaler and imputer are fitted within training cross-validation. Logistic regression and SGD log-loss, modified-Huber, and elastic-net variants share one stratified held-out set. Modified-Huber is a different loss, not an identical stochastic implementation of logistic likelihood. Results are an exploratory classifier benchmark and do not establish clinical safety or causation.
