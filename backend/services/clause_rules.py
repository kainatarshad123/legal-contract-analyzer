"""High-precision rules for hybrid clause classification."""

import re


def detect_high_precision_clause_rule(clause_text: str) -> dict:
    if not isinstance(clause_text, str) or not clause_text.strip():
        return {"rule_label": None, "rule_matches": []}

    text = clause_text.lower()

    rule_patterns = [
        ("Signature / Execution", [
            r"\bin witness whereof\b",
            r"\bsigned and delivered\b",
            r"\bexecuted by\b",
            r"\belectronic signatures?\b",
            r"\bcounterparts?\b",
            r"\bwitnessed by\b",
            r"\bin the presence of\b",
        ]),
        ("Quiet Enjoyment", [
            r"\bquiet enjoyment\b",
            r"\bpeaceably and quietly\b",
            r"\bquietly hold\b",
            r"\bquietly enjoy\b",
            r"\bwithout interruption\b",
            r"\bwithout disturbance\b",
        ]),
        ("Assignment / Subletting", [
            r"\bshall not assign\b",
            r"\bnot to assign\b",
            r"\bassignment\b",
            r"\bsublet(?:ting)?\b",
            r"\bunderlet\b",
            r"\bpart with possession\b",
        ]),
        ("Insurance", [
            r"\bcommercial general liability insurance\b",
            r"\brenter'?s insurance\b",
            r"\bproperty insurance\b",
            r"\badditional insured\b",
            r"\binsured risks?\b",
            r"\binsurance premium\b",
        ]),
        ("Liability / Indemnity", [
            r"\bindemnif(?:y|ied|ication)\b",
            r"\bhold harmless\b",
            r"\blimitation of liability\b",
            r"\bunlimited liability\b",
        ]),
        ("Termination / Default", [
            r"\bevent of default\b",
            r"\bright of re-entry\b",
            r"\bright of reentry\b",
            r"\bre-entry\b",
            r"\breentry\b",
            r"\bearlier determination\b",
            r"\bterminate this agreement\b",
            r"\btermination of this lease\b",
        ]),
        ("Possession / Surrender", [
            r"\bsurrender the premises\b",
            r"\byield up the premises\b",
            r"\bvacant possession\b",
            r"\bdeliver possession\b",
            r"\breturn possession\b",
            r"\bholding over\b",
        ]),
        ("Alterations / Improvements", [
            r"\bstructural alteration\b",
            r"\btenant improvements?\b",
            r"\badditions or alterations\b",
            r"\binstall fixtures?\b",
            r"\bapproved plans\b",
        ]),
        ("Repairs / Maintenance", [
            r"\bgood and substantial repair\b",
            r"\btenantable repair\b",
            r"\bstructural repairs?\b",
            r"\bkeep the premises clean\b",
            r"\bmaintain the premises\b",
        ]),
        ("Taxes / Utilities", [
            r"\bproperty taxes?\b",
            r"\breal estate taxes?\b",
            r"\brates and taxes\b",
            r"\butility charges?\b",
            r"\bwater charges?\b",
            r"\belectricity charges?\b",
            r"\bgas charges?\b",
        ]),
        ("Governing Law", [
            r"\bgoverned by the laws?\b",
            r"\bgoverning law\b",
            r"\bchoice of law\b",
        ]),
        ("Dispute Resolution", [
            r"\barbitration\b",
            r"\bmediation\b",
            r"\bwaiver of jury trial\b",
        ]),
        ("Confidentiality", [
            r"\bconfidential information\b",
            r"\bnon-disclosure\b",
            r"\bnondisclosure\b",
            r"\bproprietary information\b",
        ]),
        ("Notice", [
            r"\bnotice shall be in writing\b",
            r"\bcertified mail\b",
            r"\bregistered post\b",
            r"\bformal notice\b",
        ]),
        ("Lease Grant", [
            r"\bhereby leases\b",
            r"\bhereby lets\b",
            r"\bhereby demises\b",
            r"\bto hold the demised premises\b",
            r"\baccepts the lease\b",
        ]),
        ("Term and Renewal", [
            r"\boption to renew\b",
            r"\brenewal term\b",
            r"\binitial term\b",
            r"\bfixed term\b",
        ]),
        ("Payment / Rent", [
            r"\bmonthly rent\b",
            r"\bannual rent\b",
            r"\bbase rent\b",
            r"\bsecurity deposit\b",
            r"\badditional rent\b",
        ]),
        ("Use of Premises", [
            r"\bused only for\b",
            r"\bpermitted use\b",
            r"\bresidential purposes only\b",
            r"\bbusiness purposes only\b",
        ]),
    ]

    matched_labels = []

    for label, patterns in rule_patterns:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            matched_labels.append(label)

    unique_matches = list(dict.fromkeys(matched_labels))

    return {
        "rule_label": unique_matches[0] if len(unique_matches) == 1 else None,
        "rule_matches": unique_matches,
    }


def detect_secondary_clause_types(clause_text: str, primary_clause_type: str) -> list[str]:
    """Return additional legal concepts found in a mixed-purpose clause.

    The existing single-label classifier remains the source of the primary label.
    These rules add distinct secondary labels without duplicating the primary one.
    """

    if not isinstance(clause_text, str) or not clause_text.strip():
        return []

    text = clause_text.lower()

    category_terms = {
        "Payment / Rent": [
            "monthly rent", "ground rent", "rent shall be paid", "shall pay",
            "payment due", "due date", "interest thereon", "security deposit",
        ],
        "Termination / Default": [
            "termination", "terminate", "earlier determination",
            "absolutely determine", "re-entry", "reentry", "event of default",
            "in default", "arrears", "failed to comply", "breach of the covenants",
        ],
        "Assignment / Subletting": [
            "shall not assign", "not to assign", "assignment", "sublet",
            "subletting", "part with possession", "assign mortgage",
        ],
        "Liability / Indemnity": [
            "indemnify", "indemnified", "indemnification", "indemnity",
            "hold harmless", "liable for", "limitation of liability",
        ],
        "Repairs / Maintenance": [
            "good repair", "tenantable repair", "tenantable repairs",
            "structural repairs", "maintenance", "maintain the premises",
            "keep the premises clean",
        ],
        "Alterations / Improvements": [
            "alteration", "alterations", "improvement", "improvements",
            "construct any new building", "construct any new buildings",
            "demolish any existing building", "install fixtures", "approved plans",
        ],
        "Use of Premises": [
            "used only for", "use the premises", "use or permit to be used",
            "lawful purposes", "permitted use", "residential purposes",
            "business purposes",
        ],
        "Quiet Enjoyment": [
            "quiet enjoyment", "peaceably and quietly", "quietly hold",
            "quietly enjoy", "without interruption", "without disturbance",
            "without eviction",
        ],
        "Possession / Surrender": [
            "surrender the premises", "return possession", "vacant possession",
            "deliver possession", "yield up the premises", "holding over",
            "shall automatically vest in the lessor", "hand over all keys",
        ],
        "Taxes / Utilities": [
            "property taxes", "municipal taxes", "real property taxes",
            "rates and taxes", "assessment duties", "business rates",
            "electricity charges", "water charges", "utility charges", "utilities",
        ],
        "Term and Renewal": [
            "initial term", "renewal term", "option to renew", "automatically renew",
            "renew the lease", "fixed term", "extension of the term",
            "expiration of the term",
        ],
        "Lease Grant": [
            "hereby leases", "hereby lets", "hereby demises", "grants the lessee",
            "grants the tenant", "leasehold interest", "to hold the demised premises",
        ],
        "Notice": [
            "notice in writing", "written notice", "notice shall be",
            "notice must be", "service of notice",
        ],
        "Insurance": [
            "insurance policy", "maintain insurance", "insurance coverage",
            "insured against",
        ],
        "Dispute Resolution": [
            "arbitration", "dispute resolution", "waiver of jury",
            "exclusive jurisdiction", "submission to jurisdiction", "venue",
        ],
        "Governing Law": [
            "governing law", "governed by the laws", "choice of law",
        ],
        "Confidentiality": [
            "confidential information", "confidentiality", "non-disclosure",
            "nondisclosure", "keep confidential",
        ],
        "Signature / Execution": [
            "in witness whereof", "signed and delivered", "executed by",
            "counterpart", "witnessed by",
        ],
    }

    secondary_types = []

    for category, terms in category_terms.items():
        if category == primary_clause_type:
            continue

        if any(term in text for term in terms):
            secondary_types.append(category)

    # Keep the output compact and deterministic.
    return secondary_types[:4]