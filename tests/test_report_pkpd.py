"""Tests that PK/PD simulation is aggregated into the report and PDF narrative."""

from ddswarm.agents.report import ReportMinion
from ddswarm.contracts import (
    AgentResultStatus,
    ClinicalSimulation,
    ClinicalSimulationSummary,
    PDParameters,
    PDSimulation,
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
    assert "PK/PD 1/1 PK simulated" in s
    assert "m1: one_compartment t1/2 4.33h" in s


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


def _simulation_2comp_pd():
    return ClinicalSimulation(
        project_id="p", status=AgentResultStatus.success,
        simulations=[PKSimulation(
            molecule_id="m1", canonical_smiles="c1ccccc1", dose_mg=10.0, route="oral",
            model="two_compartment",
            parameters=PKParameters(clearance_ml_min_kg=12.0, central_volume_l_kg=3.0,
                                    intercompartmental_clearance_ml_min_kg=10.0,
                                    peripheral_volume_l_kg=7.0, source="user",
                                    source_ref="DrugBank"),
            half_life_hours=16.28, distribution_half_life_hours=1.435,
            cmax_mg_l=0.0476, auc_mg_h_l=0.1984, selection_reason="mechanism_hint",
        )],
        pd_simulations=[PDSimulation(
            molecule_id="m1", model="hill",
            parameters=PDParameters(baseline_effect=0.0, max_effect=100.0,
                                    ec50_mg_l=0.02, hill_coefficient=1.2),
            selection_reason="mechanism_hint",
        )],
        summary=ClinicalSimulationSummary(total_molecules=1, simulated=1,
                                          insufficient_inputs=0, pd_simulated=1),
    )


def test_narrative_two_compartment_and_pd():
    lines = []
    _narrative_clinical_simulation([_simulation_2comp_pd()], lines.append)
    joined = " ".join(lines)
    assert "two-compartment PK model" in joined
    assert "distribution t1/2 = 1.435 h" in joined
    assert "terminal t1/2 = 16.28 h" in joined
    assert "hill PD model" in joined
    assert "Hill n 1.2" in joined
    assert "selected by a mechanism hint" in joined
