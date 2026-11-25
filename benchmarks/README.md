# Benchmark history

`first_real_data_run.json` is the actual report emitted by GitHub Actions run 45635814307 (11 candidates, 100,836 MovieLens latest-small ratings), retained for transparency. The selected BPR model had test NDCG@10 0.11738 versus popularity 0.10084, 25 eligible users; paired bootstrap interval includes zero. Other candidate test metrics are diagnostic, not a selection criterion. Later EASE experiments reuse these historical periods, so they are exploratory rather than an independent untouched confirmation. A separate MovieLens 25M user-sample benchmark is configured, with potential overlap between MovieLens releases disclosed.

Training run: https://github.com/Awonke03/Movie-Recodomendation/actions/runs/45635814307
