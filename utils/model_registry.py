"""Mirrors ml_models/saved_models/metrics.json into the model_versions table."""
from datetime import datetime
from ml_models import common
from models import db, ModelVersion


def register_models():
    metrics, added = common.load_metrics(common.MODEL_DIR), 0
    for disease, meta in metrics.items():
        if ModelVersion.query.filter_by(disease=disease, version=meta["version"]).first():
            continue
        ModelVersion.query.filter_by(disease=disease, is_active=True).update({"is_active": False})
        for algo, r in meta["algorithms"].items():
            db.session.add(ModelVersion(disease=disease, algorithm=algo, version=meta["version"], data_source=meta["data_source"],
                                        n_samples=meta["n_samples"], accuracy=r["accuracy"], precision_score=r["precision"],
                                        recall=r["recall"], f1=r["f1"], roc_auc=r["roc_auc"], confusion_matrix=r["confusion_matrix"],
                                        is_best=(algo == meta["best"]), is_active=True,
                                        trained_at=datetime.fromisoformat(meta["trained_at"])))
            added += 1
    db.session.commit()
    return added
