"""
Estado de la validación: combina los checks cuantitativos, el fairness y los hallazgos.

    NO APROBADO               → falla un check cuantitativo o hay un hallazgo de severidad Alta
    APROBADO CON CONDICIONES  → checks OK, pero hay flags de fairness u observaciones de severidad Media
    APROBADO                  → sin observaciones relevantes
"""
import json

from src.config import REPORTS_DIR, ROOT


def manual_findings() -> list:
    p = ROOT / "docs" / "validation_findings.json"
    return json.load(open(p, encoding="utf-8"))["hallazgos"] if p.exists() else []


def validation_status(val: dict, fair: dict, findings: list) -> dict:
    sev = {f["severidad"] for f in findings}
    if not val["overall_pass"] or "Alta" in sev:
        label, color = "NO APROBADO — requiere remediación", "fail"
    elif not fair["overall_fairness_passed"] or "Media" in sev:
        label, color = "APROBADO CON CONDICIONES", "warn"
    else:
        label, color = "APROBADO", "pass"
    return {
        "label": label, "level": color,
        "n_findings": {s: sum(f["severidad"] == s for f in findings) for s in ("Alta", "Media", "Baja")},
    }


def load_status() -> dict:
    """Atajo para el Model Card y el dashboard (usa sólo los reportes y los hallazgos documentados)."""
    from src.governance.validation_report import auto_findings
    val  = json.load(open(REPORTS_DIR / "model_validation.json", encoding="utf-8"))
    fair = json.load(open(REPORTS_DIR / "fairness_report.json"))
    sc_p = REPORTS_DIR / "scorecard.json"
    sc   = json.load(open(sc_p, encoding="utf-8")) if sc_p.exists() else None
    findings = auto_findings(val, fair, sc) + manual_findings()
    return validation_status(val, fair, findings) | {"findings": findings}
