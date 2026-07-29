# pipelines/

The two trained pipelines live here, one folder each:

```
pipelines/
├── teat_pipeline/
│   ├── models/     teat_classifier.keras, teat_classifier.weights.h5, hand_landmarker.task
│   ├── reports/    evaluation JSON, training history, confusion matrix, gradcam/
│   ├── scripts/    prepare_dataset.py, train_teat_model.py, evaluate.py, …
│   └── data/       raw/, processed/
└── udder_pipeline/
    ├── models/     udder_svm_model.joblib, decision_threshold.json, best_config.json
    ├── reports/    test_metrics.txt, model_comparison.csv, error_analysis.txt
    ├── scripts/    features.py, segment.py, train_final.py, …
    └── data/       raw/, processed/, splits/
```

Don't move things by hand — run the organiser, which sorts a scattered folder
into this shape and shows you the plan before touching anything:

```bash
python tools/organize.py --source /path/to/your/current/files --dest .
python tools/organize.py --source /path/to/your/current/files --dest . --apply
```

The app locates these by searching for the model files, so it also copes with
other layouts. This is just the tidy one.
