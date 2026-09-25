"""Tests for the RDKit chemistry layer (deterministic, tool-computed values)."""

from rdkit import Chem

from ddswarm.chemistry import compute_properties, match_alerts, sanitize_and_canonicalize

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


def test_sanitize_valid():
    r = sanitize_and_canonicalize("CCO")
    assert r.ok and r.canonical_smiles == "CCO"


def test_canonicalizes_noncanonical():
    a = sanitize_and_canonicalize("CCO")
    b = sanitize_and_canonicalize("OCC")
    assert a.canonical_smiles == b.canonical_smiles == "CCO"


def test_sanitize_invalid():
    r = sanitize_and_canonicalize("CC(C")
    assert not r.ok and r.error


def test_sanitize_empty():
    assert not sanitize_and_canonicalize("").ok


def test_aspirin_properties():
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    p = compute_properties(mol)
    assert abs(p["molecular_weight"] - 180.16) < 0.05
    assert p["hbd"] == 1
    assert abs(p["tpsa"] - 63.6) < 0.1
    assert p["formal_charge"] == 0


def test_gefitinib_pains_clean():
    mol = Chem.MolFromSmiles(GEFITINIB)
    assert match_alerts(mol, "pains") == []


def test_gefitinib_brenk_flags_aliphatic_chain():
    # Brenk is conservative: it flags gefitinib's OCCCN propyl-morpholine chain,
    # even though gefitinib is an approved drug. This is the tool's honest output.
    mol = Chem.MolFromSmiles(GEFITINIB)
    alerts = match_alerts(mol, "brenk")
    assert any("Aliphatic_long_chain" in a for a in alerts)


def test_rhodanine_pains_flagged():
    mol = Chem.MolFromSmiles("O=C1CSC(=S)N1")
    assert any("PAINS:" in a for a in match_alerts(mol, "pains"))


def test_nitrobenzene_brenk_flagged():
    mol = Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1")
    assert any("nitro_group" in a for a in match_alerts(mol, "brenk"))
