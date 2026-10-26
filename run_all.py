"""
run_all.py — Pipeline completo Home Credit Default Risk
=======================================================
Prerequisito: data/raw/application_train.csv

Uso:
    python run_all.py

Para la API después del pipeline:
    uvicorn src.api.main:app --reload --port 8000
"""
from loguru import logger

from src.config import DATA_PROCESSED, DATA_RAW, FIGURES_DIR, MODELS_DIR, REPORTS_DIR


def main():
    logger.info("=" * 60)
    logger.info("Home Credit — Model Validation Pipeline")
    logger.info("=" * 60)

    csv_path = DATA_RAW / "application_train.csv"
    if not csv_path.exists():
        logger.error(
            f"No se encontró {csv_path}\n"
            "Descargá application_train.csv de Kaggle:\n"
            "  https://www.kaggle.com/c/home-credit-default-risk/data\n"
            "y colocalo en data/raw/"
        )
        return

    for d in [DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, REPORTS_DIR / "model_cards"]:
        d.mkdir(parents=True, exist_ok=True)

    from src.data.ingest import ingest
    from src.features.build import build
    from src.models.train import train
    from src.validation.model_validation import validate
    from src.governance.fairness import compute_fairness
    from src.validation.explain import explain
    from src.models.scorecard import build_scorecard
    from src.models.challengers import run_challengers
    from src.monitoring.baseline import create_baseline
    from src.governance.model_card import generate_model_card
    from src.governance.export_dashboard import export_dashboard_data
    from src.governance.validation_report import generate_validation_report

    stages = [
        ("Ingesta + data quality",                ingest),
        ("Feature engineering",                   build),
        ("Entrenamiento + calibración",           train),
        ("Validación (SR 26-2 / SR 11-7)",        validate),
        ("Fairness",                              compute_fairness),
        ("Explainability (SHAP)",                 explain),
        ("Scorecard WoE/IV",                      build_scorecard),
        ("Challengers (monotónico / pagos)",      run_challengers),
        ("Drift baseline",                        create_baseline),
        ("Model Card",                            generate_model_card),
        ("Export para dashboard",                 export_dashboard_data),
        ("Informe de validación (PDF)",           generate_validation_report),
    ]
    for i, (name, fn) in enumerate(stages, 1):
        logger.info(f"\n[{i}/{len(stages)}] {name}...")
        fn()

    logger.success("\n" + "=" * 60)
    logger.success("Pipeline completo!")
    logger.success("=" * 60)
    logger.info("\nOutputs principales:")
    logger.info("  reports/model_validation.json          — validación completa")
    logger.info("  reports/fairness_report.json           — fairness")
    logger.info("  reports/scorecard.json                 — scorecard WoE/IV")
    logger.info("  reports/challengers.json               — challengers M1–M3")
    logger.info("  reports/model_cards/model_card.html    — Model Card")
    logger.info("  reports/validation_report.pdf          — informe de validación")


if __name__ == "__main__":
    main()
