"""Tests for chemistry/chembl_client (network fully mocked)."""

import math

from ddswarm.chemistry import chembl_client as cc


def test_inchi_key():
    assert cc.inchi_key("CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1") == "RTHCYVBBDHJXIQ-UHFFFAOYSA-N"
    assert cc.inchi_key("not a smiles") is None


def test_search_target_prefers_human_single_protein(monkeypatch):
    monkeypatch.setattr(cc, "_get_json", lambda url, retries=2, timeout=cc.TIMEOUT: {
        "targets": [
            {"target_chembl_id": "CHEMBL313", "organism": "Rattus norvegicus",
             "target_type": "SINGLE PROTEIN"},
            {"target_chembl_id": "CHEMBL228", "organism": "Homo sapiens",
             "target_type": "SINGLE PROTEIN"},
            {"target_chembl_id": "CHEMBL999", "organism": "Homo sapiens",
             "target_type": "PROTEIN FAMILY"},
        ]})
    # the rat orthologue and the non-single-protein entry are both skipped
    assert cc.search_target("SLC6A4") == "CHEMBL228"


def test_search_target_no_human_returns_none(monkeypatch):
    monkeypatch.setattr(cc, "_get_json", lambda url, retries=2, timeout=cc.TIMEOUT: {
        "targets": [
            {"target_chembl_id": "CHEMBL313", "organism": "Rattus norvegicus",
             "target_type": "SINGLE PROTEIN"},
        ]})
    assert cc.search_target("SLC6A4") is None


def test_search_target_error_returns_none(monkeypatch):
    monkeypatch.setattr(cc, "_get_json", lambda *a, **k: None)
    assert cc.search_target("SLC6A4") is None
    assert cc.search_target("") is None


def test_compound_affinities_best_per_target(monkeypatch):
    def fake(url, retries=2, timeout=cc.TIMEOUT):
        if "/molecule.json" in url:
            return {"molecules": [{"molecule_chembl_id": "CHEMBL41", "pref_name": "fluoxetine"}]}
        return {"activities": [
            {"target_chembl_id": "CHEMBL228", "target_pref_name": "Serotonin transporter",
             "standard_type": "Ki", "standard_value": "0.8", "standard_units": "nM",
             "standard_relation": "=", "document_year": 1993},
            {"target_chembl_id": "CHEMBL228", "target_pref_name": "Serotonin transporter",
             "standard_type": "Ki", "standard_value": "5.0", "standard_units": "nM",
             "standard_relation": "=", "document_year": 2000},
            {"target_chembl_id": "CHEMBL217", "target_pref_name": "D(2) dopamine receptor",
             "standard_type": "Kd", "standard_value": "120", "standard_units": "nM",
             "standard_relation": "=", "document_year": 1995},
            {"target_chembl_id": "CHEMBL217", "target_pref_name": "D(2) dopamine receptor",
             "standard_type": "Ki", "standard_value": "80", "standard_units": "nM",
             "standard_relation": ">", "document_year": 1995},  # inequality dropped
        ]}
    monkeypatch.setattr(cc, "_get_json", fake)
    out = cc.compound_affinities("RTHCYVBBDHJXIQ-UHFFFAOYSA-N")
    # one best value per (target, type), strongest first; ">" relation dropped
    assert [(r["target_chembl_id"], r["standard_type"], r["value_nm"]) for r in out] == [
        ("CHEMBL228", "Ki", 0.8), ("CHEMBL217", "Kd", 120.0)]


def test_compound_affinities_no_molecule_returns_empty(monkeypatch):
    monkeypatch.setattr(cc, "_get_json", lambda *a, **k: {"molecules": []})
    assert cc.compound_affinities("RTHCYVBBDHJXIQ-UHFFFAOYSA-N") == []


def test_receptor_reference_ligands_median_and_top_n(monkeypatch):
    monkeypatch.setattr(cc, "_get_json", lambda url, retries=2, timeout=cc.TIMEOUT: {
        "activities": [
            # molecule A: two Ki values -> median 2.0 nM (strongest)
            {"molecule_chembl_id": "A", "canonical_smiles": "CC(=O)N",
             "molecule_pref_name": "ligA", "standard_relation": "=",
             "standard_value": "1.0", "standard_units": "nM",
             "document_journal": "J Med Chem", "document_year": 2000},
            {"molecule_chembl_id": "A", "canonical_smiles": "CC(=O)N",
             "molecule_pref_name": "ligA", "standard_relation": "=",
             "standard_value": "3.0", "standard_units": "nM",
             "document_journal": "J Med Chem", "document_year": 2001},
            # molecule B: single value 10 nM
            {"molecule_chembl_id": "B", "canonical_smiles": "CCN",
             "molecule_pref_name": "ligB", "standard_relation": "=",
             "standard_value": "10.0", "standard_units": "nM",
             "document_journal": None, "document_year": 2005},
            # molecule C: no SMILES -> dropped
            {"molecule_chembl_id": "C", "canonical_smiles": None,
             "molecule_pref_name": "ligC", "standard_relation": "=",
             "standard_value": "0.5", "standard_units": "nM"},
            # molecule D: inequality -> dropped
            {"molecule_chembl_id": "D", "canonical_smiles": "CC",
             "molecule_pref_name": "ligD", "standard_relation": ">",
             "standard_value": "0.1", "standard_units": "nM"},
        ]})
    out = cc.receptor_reference_ligands("CHEMBL217", n=5)
    assert [r["pref_name"] for r in out] == ["ligA", "ligB"]  # C/D dropped, sorted ascending
    assert math.isclose(out[0]["ki_molar"], 2.0e-9)  # median of [1.0, 3.0]
    assert out[0]["n_assays"] == 2
    assert "J Med Chem" in out[0]["citation"]


def test_receptor_reference_ligands_empty_on_no_data(monkeypatch):
    monkeypatch.setattr(cc, "_get_json", lambda *a, **k: {"activities": []})
    assert cc.receptor_reference_ligands("CHEMBL217") == []
