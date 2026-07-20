"""Risk-model inference and rule-based risk explanations."""

import os
from typing import Any

import joblib

from services.text_processing import has_blank_or_placeholder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "risk_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml_model", "vectorizer.pkl")

risk_model = None
vectorizer = None

try:
    risk_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Risk model loaded successfully.")
except Exception as error:
    print("Risk model not loaded. Using rule-based risk fallback only.")
    print("Risk model error:", error)


RISK_KEYWORDS = {
        "arrears": "Rent arrears/default wording detected.",
    "re-entry": "Re-entry remedy wording detected.",
    "reentry": "Re-entry remedy wording detected.",
    "without prejudice": "Without-prejudice rights wording detected.",
    "not entitled": "Restriction or limitation wording detected.",
    "shall not be entitled": "Restriction or limitation wording detected.",
    "without payment of any compensation": "No-compensation wording detected.",
    "assign": "Assignment restriction wording detected.",
    "sublet": "Subletting restriction wording detected.",
    "mortgage": "Mortgage/transfer restriction wording detected.",
    "termination": "Termination wording detected.",
    "earlier determination": "Early termination wording detected.",
    "default": "Default wording detected.",
    "breach": "Breach wording detected.",
    "penalty": "Penalty wording detected.",
    "liability": "Liability wording detected.",
    "indemnify": "Indemnity wording detected.",
    "indemnification": "Indemnity wording detected.",
    "indemnified": "Indemnity wording detected.",
    "damages": "Damages wording detected.",
    "late payment": "Late payment wording detected.",
    "dispute": "Dispute wording detected.",
    "waiver": "Waiver wording detected.",
    "waive": "Waiver wording detected.",
    "without notice": "No-notice wording detected.",
    "liquidated damages": "Liquidated damages wording detected.",
    "legal costs": "Legal cost wording detected.",
    "hold harmless": "Hold harmless wording detected.",
    "arbitration": "Arbitration wording detected.",
    "confidential": "Confidentiality wording detected.",
    "non-compete": "Non-compete wording detected.",
}


def predict_ml_risk(clause_text: str) -> str:
    if risk_model is None or vectorizer is None:
        return "Unknown"

    try:
        transformed = vectorizer.transform([clause_text])
        prediction = risk_model.predict(transformed)[0]
        return str(prediction)
    except Exception:
        return "Unknown"


def detect_risk_signals(clause_text: str) -> list[dict[str, str]]:
    found = []

    lower_text = clause_text.lower()

    for keyword, reason in RISK_KEYWORDS.items():
        if keyword in lower_text:
            found.append({
                "keyword": keyword,
                "reason": reason,
            })

    return found


def final_risk_level(ml_prediction: str, risk_signals: list, clause_text: str) -> str:
    lower_text = clause_text.lower()

    high_risk_terms = [
        "unlimited liability",
        "without notice",
        "liquidated damages",
        "hold harmless",
        "indemnify",
        "indemnification",
        "indemnified",
        "penalty",
    ]

    medium_risk_terms = [
        "termination",
        "earlier determination",
        "default",
        "breach",
        "damages",
        "late payment",
        "dispute",
        "waiver",
        "legal costs",
    ]

    if any(term in lower_text for term in high_risk_terms):
        return "High"

    if any(term in lower_text for term in medium_risk_terms):
        return "Medium"

    if ml_prediction in ["High", "Medium"] and risk_signals:
        return ml_prediction

    return "Low"


def detect_party_affected(clause_text: str) -> str:
    """
    Detects which party is most affected by the clause.
    This is rule-based for now and can later become an ML label.
    """

    text = clause_text.lower()

    lessee_terms = ["lessee", "tenant", "renter"]
    lessor_terms = ["lessor", "landlord", "owner"]

    lessee_count = sum(1 for term in lessee_terms if term in text)
    lessor_count = sum(1 for term in lessor_terms if term in text)

    if lessee_count > lessor_count:
        return "Lessee / Tenant"

    if lessor_count > lessee_count:
        return "Lessor / Landlord"

    if lessee_count and lessor_count:
        return "Both parties"

    return "Not clearly specified"


def generate_risk_reason(clause_text: str, risk_level: str, clause_type: str, risk_signals: list) -> str:
    """
    Creates a plain-language risk reason for the frontend clause card.
    """

    text = clause_text.lower()

    if risk_signals:
        reasons = [signal.get("reason", "") for signal in risk_signals[:2] if signal.get("reason")]
        if reasons:
            return " ".join(reasons)

    if clause_type == "Payment / Rent":
        if has_blank_or_placeholder(clause_text):
            return "This payment/rent clause appears to contain blank or incomplete commercial terms."
        return "This clause creates a payment obligation and should be checked for amount, due date, late fees, and consequences of non-payment."

    if clause_type == "Termination / Default":
        return "This clause may affect how and when the contract can end, especially after default or breach."

    if clause_type == "Liability / Indemnity":
        return "This clause may shift financial responsibility, liability, damages, or indemnity obligations to one party."

    if clause_type == "Assignment / Subletting":
        return "This clause may restrict transfer, assignment, mortgage, or subletting rights."

    if clause_type == "Dispute Resolution":
        return "This clause may affect how disputes are handled and where claims must be brought."

    if clause_type == "Notice":
        return "This clause may affect whether formal written notice is required before a party can act."

    if clause_type == "Repairs / Maintenance":
        return "This clause allocates responsibility for repairs, maintenance, damage, or the condition of the premises."

    if clause_type == "Alterations / Improvements":
        return "This clause controls alterations, construction work, improvements, consent, or reinstatement obligations."

    if clause_type == "Use of Premises":
        return "This clause limits how the premises may be occupied or used and may require legal or regulatory compliance."

    if clause_type == "Quiet Enjoyment":
        return "This clause concerns the tenant's right to occupy and use the premises without unlawful interference."

    if clause_type == "Taxes / Utilities":
        return "This clause allocates responsibility for taxes, rates, utilities, or other property-related charges."

    if clause_type == "Possession / Surrender":
        return "This clause governs delivery, return, or recovery of possession and the required condition of the premises."

    if clause_type == "Term and Renewal":
        return "This clause defines the lease duration, expiry, extension, holding over, or renewal rights."

    if clause_type == "Lease Grant":
        return "This clause creates or describes the tenant's leasehold right to occupy the premises."

    if risk_level == "High":
        return "This clause contains wording that may create significant legal or financial exposure."

    if risk_level == "Medium":
        return "This clause contains wording that should be reviewed before signing."

    return "No major risk wording was detected by the current prototype model."


def generate_recommended_action(clause_type: str, risk_level: str, clause_text: str) -> str:
    """
    Suggests a practical review action based on clause type and risk level.
    """

    if clause_type == "Payment / Rent":
        return "Confirm the amount, due date, grace period, late fee or interest rate, and payment method are clearly written."

    if clause_type == "Termination / Default":
        return "Check whether the clause includes written notice, a cure period, default period, and a fair termination process."

    if clause_type == "Liability / Indemnity":
        return "Review whether liability is capped, mutual, limited to direct losses, and covered by insurance where needed."

    if clause_type == "Assignment / Subletting":
        return "Check whether consent is required and whether consent can be unreasonably withheld."

    if clause_type == "Dispute Resolution":
        return "Confirm the governing law, forum, arbitration process, and cost responsibility are acceptable."

    if clause_type == "Confidentiality":
        return "Check the scope, duration, exceptions, and permitted disclosures."

    if clause_type == "Notice":
        return "Confirm the notice method, address, delivery timing, and required written form."

    if clause_type == "Signature / Execution":
        return "Confirm all required parties, witnesses, names, dates, and signatures are completed."

    if clause_type == "Repairs / Maintenance":
        return "Confirm which party handles structural, interior, routine, and emergency repairs and who pays the related costs."

    if clause_type == "Alterations / Improvements":
        return "Check consent, permit, contractor, restoration, ownership, and removal requirements for alterations or improvements."

    if clause_type == "Use of Premises":
        return "Confirm the permitted use is broad enough for the tenant's needs and complies with licensing and zoning requirements."

    if clause_type == "Quiet Enjoyment":
        return "Check that access, landlord entry, service interruptions, and interference rights are clearly and fairly limited."

    if clause_type == "Taxes / Utilities":
        return "Confirm exactly which taxes, rates, utilities, shared charges, and reconciliation methods each party must pay."

    if clause_type == "Possession / Surrender":
        return "Check delivery and surrender dates, required condition, fixture removal, abandoned property, and holding-over consequences."

    if clause_type == "Term and Renewal":
        return "Confirm commencement, expiry, notice deadlines, renewal options, renewal rent, and holding-over rules."

    if clause_type == "Lease Grant":
        return "Confirm the premises, included rights, commencement conditions, and scope of occupation are accurately described."

    if risk_level == "High":
        return "Review this clause carefully with a qualified legal professional before signing."

    if risk_level == "Medium":
        return "Review and clarify this clause before relying on the contract."

    return "Keep this clause as part of the review record."