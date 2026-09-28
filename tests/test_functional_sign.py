"""Tests for the general functional-sign chemotype library (offline)."""

from ddswarm.chemistry.functional_sign import classify_function

# Reference compounds (canonical; verified against the shipped SMARTS).
REF = {
    "haloperidol": "Fc1ccc(C(=O)CCCN2CCC(c3ccc(Cl)cc3)(O)CC2)cc1",
    "chlorpromazine": "CN(C)CCCN1c2ccccc2Sc2ccccc12",
    "sulpiride": "CCN1CCCC1CNC(=O)c1cc(S(N)(=O)=O)ccc1OC",
    "7oh_dpat": "CCCN(CCC)C1CCc2cc(O)ccc2C1",
    "diphenhydramine": "O(C(c1ccccc1)c1ccccc1)CCN(C)C",
    "cyclizine": "C1CN(CCN1)C(c1ccccc1)c1ccccc1",
    "atropine": "CN1C2CCC1CC(C2)OC(=O)C(CO)c1ccccc1",
    "prazosin": "COc1cc2c(nc(N3CCNCC3)nc2cc1OC)",
    "meperidine": "CCOC(=O)C1(c2ccccc2)CCN(C)CC1",
    "morphine": "CN1CCC23C=CC4C1Cc1cc(O)ccc1-c1ccc(O)c(c12)OC43",
    "naloxone": "C=CCN1CCC23C=CC4C1Cc1cc(O)ccc1-c1ccc(O)c(c12)OC43",
    "u50488": "Clc1ccc(cc1Cl)CC(=O)N(C)C1CCCCC1N1CCCC1",
}


def test_d2_antagonists():
    for name in ("haloperidol", "chlorpromazine", "sulpiride"):
        sign, motifs, citations = classify_function(REF[name], "D2")
        assert sign == "antagonist_like", f"{name} -> {sign} {motifs}"
        assert citations


def test_d2_agonist():
    sign, motifs, _ = classify_function(REF["7oh_dpat"], "D2")
    assert sign == "agonist_like" and "aminotetralin" in motifs


def test_h1_antagonists():
    for name in ("diphenhydramine", "cyclizine"):
        sign, _, _ = classify_function(REF[name], "H1")
        assert sign == "antagonist_like", name


def test_m1_antagonist():
    sign, motifs, _ = classify_function(REF["atropine"], "M1")
    assert sign == "antagonist_like" and "tropane_antimuscarinic" in motifs


def test_alpha1_antagonist():
    sign, motifs, _ = classify_function(REF["prazosin"], "alpha1A")
    assert sign == "antagonist_like" and "quinazoline" in motifs


def test_mu_agonist_antagonist_distinction():
    ag_sign, ag_motifs, _ = classify_function(REF["meperidine"], "mu_opioid")
    assert ag_sign == "agonist_like" and "phenylpiperidine" in ag_motifs
    mo_sign, mo_motifs, _ = classify_function(REF["morphine"], "mu_opioid")
    assert mo_sign == "agonist_like" and "morphinan" in mo_motifs
    an_sign, an_motifs, _ = classify_function(REF["naloxone"], "mu_opioid")
    assert an_sign == "antagonist_like" and "n_allyl_morphinan" in an_motifs


def test_kappa_agonist():
    sign, motifs, _ = classify_function(REF["u50488"], "kappa_opioid")
    assert sign == "agonist_like" and "arylacetamide" in motifs


def test_unknown_receptor_and_bad_input():
    assert classify_function(REF["haloperidol"], "not_a_receptor") == ("unknown", [], [])
    assert classify_function(None, "D2") == ("unknown", [], [])
    assert classify_function("", "D2") == ("unknown", [], [])
    assert classify_function("not-a-smiles", "D2") == ("unknown", [], [])
    assert classify_function(REF["diphenhydramine"], "D2") == ("unknown", [], [])
