"""Tests for the documented-chemotype functional-sign classifier (5-HT2A/2C)."""

from ddswarm.chemistry.ht2_function import classify_ht2_function

ANTAGONISTS = {
    "mianserin": "CN1CCN2c3ccccc3Cc3ccccc3C2C1",
    "mirtazapine": "CN1CCN2c3ncccc3Cc3ccccc3C2C1",
    "ketanserin": "Fc1ccc(C(=O)C2CCN(CCN3C(=O)NC(=O)c4ccccc43)CC2)cc1",
    "clozapine": "CN1CCN(C2=Nc3cc(Cl)ccc3Nc3ccccc32)CC1",
}

AGONISTS = {
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "psilocin": "CN(C)CCc1c[nH]c2cccc(O)c12",
    "doi": "COc1cc(OC)c(CC(C)N)cc1I",
}

NEGATIVES = {
    "escitalopram": "CN(C)CCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
    "n_ethyl": "CCN(C)CCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
    "n_demethyl": "CNCCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
}


def test_known_antagonists_are_antagonist_like():
    for name, smi in ANTAGONISTS.items():
        sign, motifs, citations = classify_ht2_function(smi)
        assert sign == "antagonist_like", f"{name}: got {sign}"
        assert motifs, f"{name}: expected non-empty motifs"
        assert len(motifs) == len(citations)


def test_known_agonists_are_agonist_like():
    for name, smi in AGONISTS.items():
        sign, motifs, citations = classify_ht2_function(smi)
        assert sign == "agonist_like", f"{name}: got {sign}"
        assert motifs, f"{name}: expected non-empty motifs"
        assert len(motifs) == len(citations)


def test_escitalopram_and_analogs_are_unknown():
    for name, smi in NEGATIVES.items():
        sign, motifs, citations = classify_ht2_function(smi)
        assert sign == "unknown", f"{name}: got {sign}"
        assert motifs == []
        assert citations == []


def test_none_and_invalid_smiles_are_unknown():
    assert classify_ht2_function(None) == ("unknown", [], [])
    assert classify_ht2_function("") == ("unknown", [], [])
    assert classify_ht2_function("not-a-smiles") == ("unknown", [], [])


def test_no_cross_matching_between_libraries():
    # antagonists must not match any agonist motif, and vice versa
    for smi in ANTAGONISTS.values():
        assert classify_ht2_function(smi)[0] == "antagonist_like"
    for smi in AGONISTS.values():
        assert classify_ht2_function(smi)[0] == "agonist_like"
