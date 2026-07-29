# Dairy Mate

Mastitis screening console. One Streamlit app serving two models — a
MobileNetV2 classifier for teat images and an SVM over handcrafted features for
udder images — with sign-in, Grad-CAM evidence, segmentation views and batch
scoring.

## Structure

```
DairyMate/
├── app.py                      entry point — sign-in gate and architecture overview
├── run.ps1 / run.sh            launchers
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   ├── config.toml             theme
│   ├── secrets.toml            sign-in config — you create this, never commit it
│   └── secrets.toml.example    annotated template
│
├── core/                       everything the pages share
│   ├── auth.py                 Google OIDC + local passcode, allowlist
│   ├── config.py               constants and pipeline discovery
│   ├── models.py               loading and inference for both models
│   ├── state.py                keeps uploads alive across page switches
│   ├── theme.py                palette and CSS
│   ├── ui.py                   masthead, verdict, decision strip
│   ├── udder_features.py       port of features.py — do not modify
│   └── udder_segment.py        port of segment.py, plus mask output
│
├── pages/
│   ├── 1_Teat_analysis.py      score + Grad-CAM
│   ├── 2_Udder_analysis.py     segmentation view + feature breakdown
│   ├── 3_Batch_run.py          folder at a time, CSV out
│   └── 4_Model_reports.py      metrics read from each pipeline
│
├── tools/
│   ├── organize.py             sorts scattered pipeline files into the layout below
│   └── rebuild_teat_model.py   re-saves the teat model for older Keras
│
└── pipelines/
    ├── teat_pipeline/
    │   ├── models/  reports/  scripts/  data/
    └── udder_pipeline/
        ├── models/  reports/  scripts/  data/
```

## Setup

```bash
pip install -r requirements.txt
```

Put the two pipelines under `pipelines/`. If yours are scattered, let the
organiser do it — dry run first, nothing changes until `--apply`:

```bash
python tools/organize.py --source /path/to/current/files --dest .
python tools/organize.py --source /path/to/current/files --dest . --apply
```

Then:

```bash
streamlit run app.py       # or  .\run.ps1  /  ./run.sh
```

The sidebar shows a tick per pipeline with the resolved path. Discovery needs
no configuration in this layout; to force a location anyway:

```bash
TEAT_PIPELINE_DIR=/path/to/teat_pipeline UDDER_PIPELINE_DIR=/path/to/udder_pipeline streamlit run app.py
```

## Sign-in

The app runs open until a provider is configured. Copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in
either section — the gate then covers every page.

**Google.** Cloud Console → Credentials → OAuth client ID → Web application.
Add the redirect URI `http://localhost:8501/oauth2callback` exactly, then:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "a long random string"
client_id = "....apps.googleusercontent.com"
client_secret = "...."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

**Restrict who gets in.** Without this, any account Google authenticates is
accepted:

```toml
[access]
emails = ["vet@example.com"]
domains = ["yourdairy.com"]
```

**Local passcode**, for before Google is set up or where there is no internet:

```bash
python core/auth.py "the password"      # prints a [local_users] line
```

Passwords are stored as PBKDF2-SHA256 hashes with a per-user salt and compared
in constant time. Google sign-in survives a browser refresh via an identity
cookie; local sign-in does not. Serve over HTTPS if this app leaves your own
machine — these are clinical records.

## Two things that matter for correct results

**Teat.** `train_teat_model.py` bakes `mobilenet_v2.preprocess_input` into the
graph, so the model is fed raw 0-255 RGB at 224x224. The app does not rescale.
Rescaling to [0,1] first would wreck accuracy and raise no error.

**Udder.** `core/udder_features.py` is a byte-for-byte port of the pipeline's
`features.py`. The saved `StandardScaler` was fitted on vectors from that exact
sequence, so a changed bin count or colour conversion shifts every feature
silently. Don't tidy it.

## Segmentation status

| | Teat | Udder |
|---|---|---|
| Segmentation code | none | GrabCut, in `scripts/segment.py` |
| Segmented output on disk | none | 95 images under `data/processed/segmented/` |
| Used at training time | — | **no** — the split manifests point at `data/raw/` |

The udder page can feed GrabCut output to the model, but the toggle is **off by
default**: the shipped SVM never saw segmented images, and scores shift enough
to flip labels when you give it some. Use it to preview, not to screen.

## Python 3.10

The teat model was saved by Keras 3.15.0, which needs Python 3.11+. On 3.10 the
newest Keras is 3.12.3 and the file will not deserialise. Either install Python
3.12, or use the rebuilt model — same weights, readable container, identical
predictions (0.9767 test accuracy either way):

```bash
python tools/rebuild_teat_model.py \
    --weights pipelines/teat_pipeline/models/teat_classifier.weights.h5 \
    --out pipelines/teat_pipeline/models/teat_classifier.keras \
    --check_dir pipelines/teat_pipeline/data/processed/test
```

## Scope

A screening aid. Both models were trained on clinical, visually obvious cases,
so neither sees subclinical mastitis, and a clean score is not a clean quarter.
Anything flagged still needs a hands-on check before treatment.
