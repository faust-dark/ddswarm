"""RuleBasedAdmetProvider: deterministic, RDKit descriptor-based ADMET rules.

Every endpoint is a published QSAR rule with an exact formula/threshold and a
literature citation, computed deterministically from RDKit descriptors (no ML
model, no torch/CUDA). All outputs are 'predicted' — a computational rule is
not experimental evidence. The exact formula/threshold for each rule lives in
the code below; the citation lets a reader recover the primary source.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from ..contracts.admet_prediction import (
    ApplicabilityDomain,
    EndpointPrediction,
    Uncertainty,
)
from .base import AdmetProvider

MODEL_VERSION = "1.0.0"

_CITATIONS: dict[str, str] = {
    "LogS_ESOL": "Delaney JS (2004). ESOL: Estimating Aqueous Solubility Directly from Molecular "
                 "Structure. J Chem Inf Comput Sci 44(3):1000-1005.",
    "QED": "Bickerton GR et al. (2012). Quantifying the chemical beauty of drugs. Nat Chem 4:90-98.",
    "Lipinski_Violations": "Lipinski CA et al. (1997). Experimental and computational approaches to estimate "
                           "solubility and permeability. Adv Drug Deliv Rev 23:3-25.",
    "Veber_Bioavailability": "Veber DF et al. (2002). Molecular properties that influence oral bioavailability. "
                             "J Med Chem 45:2615-2623.",
    "Egan_Absorption": "Egan WJ et al. (2000). Prediction of drug absorption using multivariate statistics. "
                       "J Med Chem 43:3867-3877.",
    "BBB_Permeability": "Pajouhesh H, Lenz GR (2005). Medicinal chemical properties of successful central "
                        "nervous system drugs. NeuroRx 2(4):541-553.",
    "Clearance_ml_min_kg": "Lombardo F et al. (2002). Prediction of volume of distribution values in humans... "
                           "J Med Chem 45:2867-2876. (QSPR — no precise structure-only rule)",
    "Volume_of_Distribution_l_kg": "Lombardo F et al. (2004). Prediction of human volume of distribution... "
                                   "J Med Chem 47:1242-1250. (QSPR — no precise structure-only rule)",
    "Half_Life_hours": "Obach RS et al. (2008). Trend analysis of a database of intravenous pharmacokinetic "
                       "parameters... Drug Metab Dispos 36:1385-1405. (derived from CL and Vd)",
    "Plasma_Protein_Binding": "Requires pKa + logD QSPR (no precise structure-only rule).",
}

_RULE_NOTES: dict[str, str] = {
    "LogS_ESOL": "LogS = 0.16 - 0.63*clogP - 0.0062*MW + 0.066*RB - 0.74*AP",
    "QED": "RDKit QED.default(); 0 (unfavorable) to 1 (favorable)",
    "Lipinski_Violations": "Rule-of-5 filter count: MW>500, clogP>5, HBD>5, HBA>10",
    "Veber_Bioavailability": "good oral bioavailability if RB <= 10 AND TPSA <= 140 A^2",
    "Egan_Absorption": "good passive absorption if 0.4 < clogP < 5.88 AND TPSA < 131.6 A^2 (box approximation)",
}


class RuleBasedAdmetProvider(AdmetProvider):
    name = "rule_admet"
    model_version = MODEL_VERSION
    supported_endpoints = [
        "LogS_ESOL",
        "QED",
        "Lipinski_Violations",
        "Veber_Bioavailability",
        "Egan_Absorption",
        "BBB_Permeability",
        "Clearance_ml_min_kg",
        "Volume_of_Distribution_l_kg",
        "Half_Life_hours",
        "Plasma_Protein_Binding",
    ]

    def predict(self, canonical_smiles: str) -> list[EndpointPrediction]:
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            return self._out_of_domain("SMILES could not be parsed")
        try:
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            rb = Descriptors.NumRotatableBonds(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            heavy = mol.GetNumHeavyAtoms()
        except Exception:  # noqa: BLE001
            return self._out_of_domain("descriptors not computable")
        if heavy == 0:
            return self._out_of_domain("no heavy atoms")
        arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        ap = arom / heavy

        ad = ApplicabilityDomain(
            status="in_domain",
            method="RDKit descriptors (Crippen clogP, TPSA, MW, RB, HBD/HBA)",
            note="neutral organic molecule with computable descriptors",
        )

        out: list[EndpointPrediction] = []

        # ---- LogS_ESOL (Delaney 2004) ----
        logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap
        out.append(self._ep(
            "LogS_ESOL", value=round(logs, 4), units="log10(mol/L)", ad=ad,
            uncertainty=Uncertainty(type="std", value=0.75,
                                    note="reported RMSE 0.75 log units (Delaney 2004)"),
        ))

        # ---- QED (Bickerton 2012) ----
        out.append(self._ep(
            "QED", value=round(QED.default(mol), 4), units="score", ad=ad,
            uncertainty=Uncertainty(type="not_available",
                                    note="composite score; no calibrated uncertainty"),
        ))

        # ---- Lipinski violations (Lipinski 1997) ----
        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        out.append(self._ep(
            "Lipinski_Violations", value=float(violations), units="count",
            label=f"{violations} violation" + ("" if violations == 1 else "s"), ad=ad,
            uncertainty=Uncertainty(type="not_available",
                                    note="filter count, not a calibrated prediction"),
        ))

        # ---- Veber bioavailability (Veber 2002) ----
        veber_ok = (rb <= 10) and (tpsa <= 140)
        out.append(self._ep(
            "Veber_Bioavailability", label="good" if veber_ok else "poor", ad=ad,
            uncertainty=Uncertainty(type="not_available",
                                    note="rule-based threshold; no calibrated uncertainty"),
        ))

        # ---- Egan absorption (Egan 2000) ----
        egan_ok = (0.4 < logp < 5.88) and (tpsa < 131.6)
        out.append(self._ep(
            "Egan_Absorption", label="good" if egan_ok else "poor", ad=ad,
            uncertainty=Uncertainty(type="not_available",
                                    note="box approximation of the Egan ellipse; no calibrated uncertainty"),
        ))

        # ---- BBB permeability (Pajouhesh & Lenz 2005) ----
        bbb_ok = (mw <= 400) and (tpsa <= 90) and (hbd <= 3)
        out.append(self._ep(
            "BBB_Permeability", label="CNS penetrant likely" if bbb_ok else "not likely", ad=ad,
            uncertainty=Uncertainty(type="not_available",
                                    note="threshold rule (MW<=400, TPSA<=90, HBD<=3); not a calibrated model"),
        ))

        # ---- PK endpoints that need a QSPR model (honest not_available) ----
        out.append(self._na_ep(
            "Clearance_ml_min_kg",
            "no precise structure-only rule; supply a value for known analogs (literature or reasoning recall)"))
        out.append(self._na_ep(
            "Volume_of_Distribution_l_kg",
            "no precise structure-only rule; supply a value for known analogs (literature or reasoning recall)"))
        out.append(self._na_ep(
            "Half_Life_hours",
            "derived as ln2*Vd/CL; supply clearance and volume of distribution (or a recalled value)"))
        out.append(self._na_ep(
            "Plasma_Protein_Binding",
            "requires pKa + logD QSPR; no precise structure-only rule"))

        return out

    def _na_ep(self, endpoint: str, note: str) -> EndpointPrediction:
        """An endpoint that is in scope but has no precise deterministic rule."""
        return EndpointPrediction(
            endpoint=endpoint,
            model=self.name,
            model_version=self.model_version,
            prediction_status="not_available",
            applicability_domain=ApplicabilityDomain(status="unknown", method="QSPR required", note=note),
            uncertainty=Uncertainty(type="not_available", note=note),
            citation=_CITATIONS.get(endpoint),
        )

    def _ep(self, endpoint: str, *, value: float | None = None, units: str | None = None,
            label: str | None = None, ad: ApplicabilityDomain | None = None,
            uncertainty: Uncertainty | None = None) -> EndpointPrediction:
        return EndpointPrediction(
            endpoint=endpoint,
            model=self.name,
            model_version=self.model_version,
            predicted_value=value,
            units=units,
            predicted_label=label,
            prediction_status="predicted",
            applicability_domain=ad or ApplicabilityDomain(status="in_domain", method="RDKit descriptors"),
            uncertainty=uncertainty or Uncertainty(type="not_available"),
            citation=_CITATIONS[endpoint],
        )

    def _out_of_domain(self, reason: str) -> list[EndpointPrediction]:
        ad = ApplicabilityDomain(status="out_of_domain", method="RDKit descriptors", note=reason)
        return [
            EndpointPrediction(
                endpoint=ep, model=self.name, model_version=self.model_version,
                prediction_status="out_of_domain",
                applicability_domain=ad,
                uncertainty=Uncertainty(type="not_available", note=reason),
                citation=_CITATIONS[ep],
            )
            for ep in self.supported_endpoints
        ]
