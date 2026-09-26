"""Tests that PK/PD simulation is aggregated into the report and PDF narrative."""

from ddswarm.agents.report import ReportMinion
from ddswarm.contracts import (
    AgentResultStatus,
    ClinicalSimulation,
    ClinicalSimulationSummary,
    PKParameters,
    PKSimulation,
)
from ddswarm.reporting.pdf_render import _narrative_clinical_simulation


def _simulation():
    return ClinicalSimulation(
        project_id="p", status=AgentResultStatus.success,
        simulations=[PKSimulation(
            molecule_id="m1", canonical_smiles="c1ccccc1", dose_mg=10.0, route="oral",
            parameters=PKParameters(clearance_ml_min_kg=12.0, volume_of_distribution_l_kg=4.5,
                                    bioavailability=0.9, half_life_hours=4.33, source="user",
                                    source_ref="DrugBank"),
            half_life_hours=4.33, cmax_mg_l=0.0286, auc_mg_h_l=0.1786,
        )],
        summary=ClinicalSimulationSummary(total_molecules=1, simulated=1, insufficient_inputs=0),
    )


def test_report_exec_summary_includes_pkpd():
    by_kind = {"clinical_simulation": [_simulation()]}
    s = ReportMinion._executive_summary(by_kind)
    assert "PK/PD 1/1 simulated" in s
    assert "m1: t1/2 4.33h" in s


def test_report_evidence_counts_pkpd_is_calculated():
    by_kind = {"clinical_simulation": [_simulation()]}
    c = ReportMinion._evidence_counts(by_kind)
    assert c["calculated"] == 1  # one-compartment model is deterministic math


def test_report_gap_when_no_simulation():
    assert any("PK/PD simulation" in g for g in ReportMinion._gaps({}))


def test_narrative_clinical_simulation_text():
    lines = []
    _narrative_clinical_simulation([_simulation()], lines.append)
    joined = " ".join(lines)
    assert "half-life t1/2 = 4.33 h" in joined
    assert "Cmax = 0.029 mg/L" in joined
    assert "AUC = 0.179 mg*h/L" in joined
    assert "Simulated 1 of 1" in joined
