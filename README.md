# Dataset setup

This project uses the genuine **Parkinsons** voice-measurement dataset from the
UCI Machine Learning Repository:

- Dataset page: <https://archive.ics.uci.edu/dataset/174/parkinsons>
- Direct official ZIP: <https://archive.ics.uci.edu/static/public/174/parkinsons.zip>
- Original data file: `parkinsons.data`

Download the ZIP from the official link, place it in this folder, and extract
`parkinsons.data` into this folder. The expected path, relative to the project
root, is `data\parkinsons.data`. The project intentionally does not include a
copy of the data.

The file is a comma-separated table with these columns:

```text
name, MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%),
MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer,
MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA,
NHR, HNR, status, RPDE, DFA, spread1, spread2, D2, PPE
```

There are 22 numeric voice/acoustic measurement columns. `status` is the
target; it uses `1` for a Parkinson's case and `0` for a control in this
dataset. `name` is a recording identifier, not a model feature, and is excluded
during training.

The notebook checks the downloaded file before analysis. If it is absent, follow
these instructions; the project will not fabricate records or claim to have
trained or evaluated a model. Numeric features are imputed and scaled inside
scikit-learn pipelines, fitted only on training data.