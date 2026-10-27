"""Canonical human-facing stage label + one-line role for each agent.

Single source of truth for the workflow schema — used by the orchestrator's
ExecutionPlan and the PDF renderer so the plan and the report reflect the
EXACT same stage -> agent -> role mapping.
"""

from __future__ import annotations

AGENT_ROLES: dict[str, tuple[str, str]] = {
    "orchestrator_minion": ("Orchestration", "interprets the project, plans the stages, and drives iteration"),
    "researcher_minion": ("Research", "gathers what is already known about the target and starting molecule"),
    "designer_minion": ("Design", "proposes new analog molecules (deterministic RDKit library, or LLM with a "
                                  "reasoning backend)"),
    "screener_minion": ("Screening", "filters molecules by drug-likeness and structural liabilities (PAINS is hard, "
                                     "Brenk advisory)"),
    "predictor_minion": ("ADMET prediction", "estimates ADMET properties from five published rules"),
    "retrosynthesis_minion": ("Synthesis", "scores synthetic accessibility and proposes a theoretical retrosynthetic "
                                           "disconnection)"),
    "tox_minion": ("Toxicity", "flags structural toxicophore alerts (predicted liabilities, not observed toxicity)"),
    "cyp_minion": ("CYP DDI", "flags CYP inhibition/induction structural alerts (predicted drug-drug-interaction "
                              "liabilities, not a functional assay)"),
    "herg_minion": ("hERG", "flags potential hERG cardiotoxicity liability (predicted pharmacophore match, not a "
                            "patch-clamp assay)"),
    "grounding_minion": ("Grounding", "resolves molecules against ChEMBL/PubChem/PubMed (observed external "
                                      "records with URL + identifier; opt-in)"),
    "export_minion": ("Export", "writes a machine-readable candidate table (CSV/JSON/SDF) with a deterministic "
                                "go/no-go verdict)"),
    "formulation_minion": ("Formulation", "estimates developability — solubility, permeability, and BCS class"),
    "docking_minion": ("Docking", "docks candidates into the receptor and reports predicted affinity (opt-in)"),
    "dynamics_minion": ("Receptor MD", "runs a short receptor molecular-dynamics relaxation (opt-in, gated on all four "
                                       "branches)"),
    "selectivity_minion": ("Target selectivity", "docks a configurable receptor panel and derives mechanistic ADR "
                                                 "interpretations from occupancy (SSRI sexual-ADR, NSAID GI, "
                                                 "antitarget ADRs) — opt-in)"),
    "safety_minion": ("Safety", "safety gate — flags controlled substances and dual-use targets"),
    "report_minion": ("Report", "aggregates the run into a structured report and ranks the survivors"),
    "memory_minion": ("Memory", "distils the generation into best candidates, avoided scaffolds, and insights"),
    "validation_minion": ("Validation", "audits the run's bookkeeping — schemas, hashes, cross-agent consistency"),
    "report_pdf_minion": ("PDF export", "renders this document"),
    "patent_minion": ("Patent landscape", "patent / freedom-to-operate knowledge (opt-in, LLM)"),
    "clinical_minion": ("Clinical knowledge", "clinical-stage knowledge — known drugs, trials (opt-in, LLM)"),
    "regulatory_minion": ("Regulatory knowledge", "regulatory status — approvals, scheduling, 7D readiness "
                                                  "(opt-in, LLM)"),
}
