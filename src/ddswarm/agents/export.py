"""export_minion: machine-readable candidate export (CSV / JSON / SDF + go/no-go).

Reads every by-kind artifact from the store, builds one flat row per screened
molecule with all computed/predicted scores, applies a deterministic go/no-go
verdict (reusing the pipeline's existing gates — no new thresholds), and writes
three machine-readable artifacts into the store (versioned, never overwritten):

  - candidate_table (JSON) — the authoritative, provenance-hashed artifact
  - candidates.csv        — flat spreadsheet
  - candidates.sdf        — molecule + properties for external cheminformatics tools

The verdict:
  no-go   eliminated at screening (PAINS/Ro5/invalid) OR a safety block flag
  review  passed screening but has >=1 advisory liability (tox/CYP/hERG/Brenk/BCS IV)
  go      passed screening, no safety flags, no advisories
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit import rdBase

from ..contracts.agent_result import AgentResult
from ..contracts.candidate_table import CandidateRow, CandidateTable, ExportSummary, Verdict
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from ..orchestrator.runner import ARTIFACT_MODELS
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)

# Advisory liabilities that push a passed candidate to "review".
_BCS_ADVISORY = {"IV"}


def _verdict(screening_status: str, safety_flags: int, tox_alerts: int, cyp_flags: list[str],
             herg_flag: bool, brenk_alerts: int, bcs_class: str | None,
             elimination_reasons: list[str]) -> tuple[Verdict, list[str]]:
    if screening_status != "passed":
        return "no-go", [f"eliminated at screening: {', '.join(elimination_reasons) or screening_status}"]
    if safety_flags > 0:
        return "no-go", [f"{safety_flags} safety flag(s)"]
    advisories: list[str] = []
    if tox_alerts > 0:
        advisories.append(f"{tox_alerts} tox alert(s)")
    if cyp_flags:
        advisories.append(f"{len(cyp_flags)} CYP DDI flag(s)")
    if herg_flag:
        advisories.append("hERG liability")
    if brenk_alerts > 0:
        advisories.append(f"{brenk_alerts} Brenk alert(s)")
    if bcs_class in _BCS_ADVISORY:
        advisories.append("BCS IV (poor solubility + permeability)")
    if advisories:
        return "review", advisories
    return "go", []


class ExportMinion(Agent):
    name = "export_minion"
    reads_whole_run = True  # scans every task's result in the run manifest

    def run(self, ctx: AgentContext) -> AgentResult:
        manifest = self.store.load_run_manifest(ctx.project_id, ctx.run_id)
        by_kind: dict[str, list] = {}
        for task_id in manifest.tasks:
            if task_id == ctx.task_id:
                continue
            try:
                res = self.store.load_latest_agent_result(ctx.project_id, ctx.run_id, task_id)
            except FileNotFoundError:
                continue
            for a in res.artifacts:
                if not a.uri or a.kind not in ARTIFACT_MODELS:
                    continue
                try:
                    payload = ARTIFACT_MODELS[a.kind].model_validate_json(Path(a.uri).read_text())
                except Exception:  # noqa: BLE001
                    continue
                by_kind.setdefault(a.kind, []).append(payload)

        batches = by_kind.get("screened_batch", [])
        if not batches:
            return AgentResult(
                project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence, artifacts=[],
                input_refs=ctx.input_refs, findings=["no screened_batch to export"],
                tool_versions=[TOOL, RDKIT_TOOL], reproducibility={"reason": "no_screened_batch"},
            )

        # Per-molecule lookups keyed by canonical SMILES (the machine backbone).
        qed: dict[str, float] = {}
        for ap in by_kind.get("admet_prediction", []):
            for m in ap.molecules:
                for ep in m.endpoints:
                    if (ep.endpoint == "QED" and ep.prediction_status == "predicted"
                            and ep.predicted_value is not None):
                        qed[m.canonical_smiles] = ep.predicted_value
        sa: dict[str, float] = {}
        for rp in by_kind.get("retrosynthesis_plan", []):
            for m in rp.molecules:
                sa[m.canonical_smiles] = m.sa_score
        tox: dict[str, int] = {}
        for tp in by_kind.get("tox_profile", []):
            for m in tp.molecules:
                tox[m.canonical_smiles] = len(m.alerts)
        cyp: dict[str, list[str]] = {}
        for cp in by_kind.get("cyp_assessment", []):
            for m in cp.molecules:
                cyp[m.canonical_smiles] = [f"{l.enzyme} {l.effect}" for l in m.liabilities]
        herg: dict[str, bool] = {}
        for ha in by_kind.get("herg_assessment", []):
            for m in ha.molecules:
                herg[m.canonical_smiles] = m.flagged
        bcs: dict[str, str] = {}
        for fa in by_kind.get("formulation_assessment", []):
            for m in fa.molecules:
                bcs[m.canonical_smiles] = m.bcs_class
        safety: dict[str, int] = {}
        for sfa in by_kind.get("safety_assessment", []):
            for f in sfa.flags:
                if f.level == "block":
                    safety[f.subject] = safety.get(f.subject, 0) + 1

        # Ranking order + IUPAC labels from the report (survivors only).
        rank_by_smiles: dict[str, int] = {}
        iupac_by_smiles: dict[str, str] = {}
        for rep in by_kind.get("report", []):
            for c in rep.ranked_candidates:
                rank_by_smiles[c.smiles] = c.rank
                if c.iupac_name:
                    iupac_by_smiles[c.smiles] = c.iupac_name

        rows: list[CandidateRow] = []
        for sb in batches:
            for m in sb.molecules:
                smi = m.canonical_smiles or m.input_smiles
                props = m.properties
                s_status = ("passed" if m.passed else
                            "invalid" if not m.valid else "rejected")
                verdict, reasons = _verdict(
                    s_status,
                    safety.get(m.id, 0),
                    tox.get(smi, 0),
                    cyp.get(smi, []),
                    herg.get(smi, False),
                    len(m.brenk_alerts),
                    bcs.get(smi),
                    m.elimination_reasons,
                )
                rows.append(CandidateRow(
                    rank=rank_by_smiles.get(smi),
                    molecule_id=m.id,
                    input_smiles=m.input_smiles,
                    canonical_smiles=smi if m.valid else None,
                    screening_status=s_status,
                    molecular_weight=round(props.molecular_weight, 2) if props else None,
                    logp=round(props.logp, 3) if props else None,
                    hbd=props.hbd if props else None,
                    hba=props.hba if props else None,
                    tpsa=round(props.tpsa, 2) if props else None,
                    qed=round(qed[smi], 3) if smi in qed else None,
                    ro5_violations=len(m.rule_of_five_violations),
                    bcs_class=bcs.get(smi),
                    sa_score=round(sa[smi], 2) if smi in sa else None,
                    pains_alerts=len(m.structural_alerts),
                    brenk_alerts=len(m.brenk_alerts),
                    tox_alerts=tox.get(smi, 0),
                    cyp_flags=cyp.get(smi, []),
                    herg_flag=herg.get(smi, False),
                    safety_flags=safety.get(m.id, 0),
                    verdict=verdict, reasons=reasons,
                    iupac_name=iupac_by_smiles.get(smi),
                ))

        summary = ExportSummary(
            total_molecules=len(rows),
            go=sum(1 for r in rows if r.verdict == "go"),
            review=sum(1 for r in rows if r.verdict == "review"),
            no_go=sum(1 for r in rows if r.verdict == "no-go"),
        )

        input_hash = hash_jsonable({
            kind: [a.model_dump(mode="json", exclude={"provenance", "run_id"}) for a in arts]
            for kind, arts in sorted(by_kind.items())
        })
        output_hash = hash_jsonable([r.model_dump(exclude={"iupac_name"}) for r in rows])

        table = CandidateTable(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            rows=rows,
            summary=summary,
            note=("go/no-go verdict reuses the pipeline's existing gates (screening, safety, "
                  "tox/CYP/hERG/Brenk/BCS advisories) — no new thresholds. It is a ranking/decision "
                  "aid, not an experimental claim."),
            provenance=[ProvenanceEntry(
                source="machine-readable candidate export", tool_versions=[TOOL, RDKIT_TOOL],
                input_hash=input_hash, output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "candidate_table", f"candidates_{ctx.project_id}", table)

        # Sidecar files: CSV + SDF (versioned, never overwritten).
        csv_bytes = self._to_csv(rows).encode("utf-8")
        sdf_bytes = self._to_sdf(rows)
        csv_path, _v, csv_hash = self.store.save_binary_artifact(
            ctx.project_id, ctx.run_id, f"candidates_{ctx.project_id}", "candidate_csv", "csv", csv_bytes)
        sdf_path, _v, sdf_hash = self.store.save_binary_artifact(
            ctx.project_id, ctx.run_id, f"candidates_{ctx.project_id}", "candidate_sdf", "sdf", sdf_bytes)

        finding = (f"export: {summary.total_molecules} molecules; "
                   f"{summary.go} go, {summary.review} review, {summary.no_go} no-go "
                   f"(CSV + JSON + SDF)")
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[finding],
            provenance=[ProvenanceEntry(source="machine-readable candidate export",
                                        output_hash=art.hash, tool_versions=[TOOL, RDKIT_TOOL])],
            tool_versions=[TOOL, RDKIT_TOOL],
            reproducibility={"input_hash": input_hash,
                             "csv_hash": csv_hash, "sdf_hash": sdf_hash,
                             "csv_uri": str(csv_path), "sdf_uri": str(sdf_path)},
        )

    @staticmethod
    def _to_csv(rows: list[CandidateRow]) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["rank", "molecule_id", "canonical_smiles", "iupac_name", "screening_status", "verdict",
                    "molecular_weight", "logp", "hbd", "hba", "tpsa", "qed", "ro5_violations",
                    "bcs_class", "sa_score", "pains_alerts", "brenk_alerts", "tox_alerts",
                    "cyp_flags", "herg_flag", "safety_flags", "reasons"])
        for r in rows:
            w.writerow([
                r.rank if r.rank is not None else "",
                r.molecule_id, r.canonical_smiles or "", r.iupac_name or "",
                r.screening_status, r.verdict,
                r.molecular_weight if r.molecular_weight is not None else "",
                r.logp if r.logp is not None else "",
                r.hbd if r.hbd is not None else "",
                r.hba if r.hba is not None else "",
                r.tpsa if r.tpsa is not None else "",
                r.qed if r.qed is not None else "",
                r.ro5_violations, r.bcs_class or "", r.sa_score if r.sa_score is not None else "",
                r.pains_alerts, r.brenk_alerts, r.tox_alerts,
                "; ".join(r.cyp_flags), "1" if r.herg_flag else "0", r.safety_flags,
                "; ".join(r.reasons),
            ])
        return buf.getvalue()

    @staticmethod
    def _to_sdf(rows: list[CandidateRow]) -> bytes:
        fd, fname = tempfile.mkstemp(suffix=".sdf")
        os.close(fd)
        try:
            w = Chem.SDWriter(fname)
            for r in rows:
                mol = Chem.MolFromSmiles(r.canonical_smiles) if r.canonical_smiles else None
                if mol is None:
                    continue
                mol.SetProp("molecule_id", r.molecule_id)
                mol.SetProp("verdict", r.verdict)
                if r.iupac_name:
                    mol.SetProp("iupac_name", r.iupac_name)
                mol.SetProp("rank", str(r.rank) if r.rank is not None else "")
                mol.SetProp("screening_status", r.screening_status)
                mol.SetProp("qed", f"{r.qed:.3f}" if r.qed is not None else "")
                mol.SetProp("logp", f"{r.logp:.3f}" if r.logp is not None else "")
                mol.SetProp("bcs_class", r.bcs_class or "")
                mol.SetProp("sa_score", f"{r.sa_score:.2f}" if r.sa_score is not None else "")
                mol.SetProp("tox_alerts", str(r.tox_alerts))
                mol.SetProp("cyp_flags", "; ".join(r.cyp_flags))
                mol.SetProp("herg_flag", "1" if r.herg_flag else "0")
                mol.SetProp("safety_flags", str(r.safety_flags))
                mol.SetProp("reasons", "; ".join(r.reasons))
                w.write(mol)
            w.close()
            with open(fname, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass
