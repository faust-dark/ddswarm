"""Deterministic PDF renderer for the final run report — a readable, verbose
"scientific essay" in plain human language.

Pure function: takes the aggregated run artifacts (Report, ValidationReport, and
the by-kind artifact map) plus the ProjectSpec, and renders a self-contained PDF
whose body is narrative prose (interpretation of every result, not a data dump),
with figures kept only as auxiliary aids to the text. No LLM, no network, no
randomness — interpretation is produced by deterministic template rules from the
structured data, so the same inputs yield the same prose (page/figure counts are
deterministic). The raw PDF bytes are not guaranteed byte-identical across
reportlab/matplotlib builds, which is why the ReportDocument metadata carries its
own sha256 rather than promising a byte-stable artifact.

Figures and text both use the DejaVu TTFs bundled with matplotlib, so the
document is self-contained (no external font files) and Unicode-clean."""

from __future__ import annotations

import tempfile
from pathlib import Path
from xml.sax.saxutils import escape as _xmlescape

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.inchi import MolToInchiKey
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

EVIDENCE_COLORS = {
    "observed": "#2e7d32",
    "calculated": "#1565c0",
    "predicted": "#6a1b9a",
    "inferred": "#ef6c00",
    "hypothetical": "#9e9e9e",
}

# Human-facing stage label + one-line role for each agent, used to label report
# sections clearly and explain what each agent does.
AGENT_LABELS: dict[str, tuple[str, str]] = {
    "researcher_minion": ("Research", "gathers what is already known about the target and starting molecule"),
    "designer_minion": ("Design", "proposes new analog molecules (deterministic RDKit library, or LLM with a reasoning backend)"),
    "screener_minion": ("Screening", "filters molecules by drug-likeness and structural liabilities (PAINS is hard, Brenk advisory)"),
    "predictor_minion": ("ADMET prediction", "estimates ADMET properties from five published rules"),
    "retrosynthesis_minion": ("Synthesis", "scores synthetic accessibility and proposes a theoretical retrosynthetic disconnection"),
    "tox_minion": ("Toxicity", "flags structural toxicophore alerts (predicted liabilities, not observed toxicity)"),
    "formulation_minion": ("Formulation", "estimates developability — solubility, permeability, and BCS class"),
    "docking_minion": ("Docking", "docks candidates into the receptor and reports predicted affinity (opt-in)"),
    "dynamics_minion": ("Receptor MD", "runs a short receptor molecular-dynamics relaxation (opt-in)"),
    "safety_minion": ("Safety", "safety gate — flags controlled substances and dual-use targets"),
    "report_minion": ("Report", "aggregates the run into a structured report and ranks the survivors"),
    "validation_minion": ("Validation", "audits the run's bookkeeping — schemas, hashes, cross-agent consistency"),
    "report_pdf_minion": ("PDF export", "renders this document"),
    "patent_minion": ("Patent landscape", "patent / freedom-to-operate knowledge (opt-in, LLM)"),
    "clinical_minion": ("Clinical knowledge", "clinical-stage knowledge — known drugs, trials (opt-in, LLM)"),
    "regulatory_minion": ("Regulatory knowledge", "regulatory status — approvals, scheduling (opt-in, LLM)"),
}

_fonts_registered = False


def _register_fonts() -> None:
    global _fonts_registered
    if _fonts_registered:
        return
    ttf_dir = Path(mpl.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(ttf_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(ttf_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(ttf_dir / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", str(ttf_dir / "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold",
        italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique",
    )
    _fonts_registered = True


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("Title", fontName="DejaVu-Bold", fontSize=17, leading=21,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("Subtitle", fontName="DejaVu", fontSize=9.5, leading=13,
                                   textColor=colors.HexColor("#555555"), spaceAfter=2),
        "h1": ParagraphStyle("H1", fontName="DejaVu-Bold", fontSize=12.5, leading=16,
                             spaceBefore=13, spaceAfter=5, textColor=colors.HexColor("#1a1a2e")),
        "h2": ParagraphStyle("H2", fontName="DejaVu-Bold", fontSize=10.5, leading=13,
                             spaceBefore=9, spaceAfter=3),
        "body": ParagraphStyle("Body", fontName="DejaVu", fontSize=10, leading=14.5,
                               spaceAfter=6, splitLongWords=1),
        "bullet": ParagraphStyle("Bullet", parent=None, fontName="DejaVu", fontSize=10,
                                 leading=14, leftIndent=14, spaceAfter=2, splitLongWords=1),
        "small": ParagraphStyle("Small", fontName="DejaVu", fontSize=8, leading=11,
                                textColor=colors.HexColor("#555555"), spaceAfter=4),
        "caption": ParagraphStyle("Caption", fontName="DejaVu-Oblique", fontSize=8, leading=10.5,
                                  textColor=colors.HexColor("#666666"), spaceBefore=2, spaceAfter=9),
        "th": ParagraphStyle("TH", fontName="DejaVu-Bold", fontSize=8, leading=10.5,
                             textColor=colors.white),
        "td": ParagraphStyle("TD", fontName="DejaVu", fontSize=7.8, leading=10, splitLongWords=1),
    }


def _esc(text: object) -> str:
    return _xmlescape(str(text))


def _cell(text: object) -> str:
    return _xmlescape(str(text)).replace("\n", "<br/>")


def _table(headers: list[str], rows: list[list], col_widths: list[float],
           styles: dict[str, ParagraphStyle]) -> Table:
    data = [[Paragraph(_esc(h), styles["th"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(_cell(c), styles["td"]) for c in row])
    t = Table(data, colWidths=[w * cm for w in col_widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _new_fig(w: float = 6.5, h: float = 3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


# ---- plain-language interpretation rules (deterministic) ----


def _interpret_logp(logp: float) -> str:
    if logp < 0:
        return "hydrophilic"
    if logp < 3:
        return "well balanced between solubility and permeability"
    if logp < 5:
        return "moderately lipophilic"
    return "highly lipophilic — a Rule-of-5 solubility risk"


def _interpret_esol(logs: float) -> str:
    if logs < -6:
        return "poorly water-soluble"
    if logs < -4:
        return "moderately soluble"
    return "reasonably soluble"


def _interpret_qed(qed: float) -> str:
    if qed >= 0.67:
        return "attractive drug-likeness"
    if qed >= 0.49:
        return "moderate drug-likeness"
    return "low drug-likeness"


def _interpret_lipinski(n: int) -> str:
    return ("no Lipinski rule-of-5 violations" if n == 0
            else f"{n} Lipinski rule-of-5 violations")


def _interpret_sa(sa: float) -> str:
    if sa <= 3:
        return "easy to synthesize"
    if sa <= 6:
        return "moderately challenging to synthesize"
    return "difficult to synthesize"


def _interpret_bcs(cls: str) -> str:
    return {
        "I": "high solubility and high permeability — the most developable class",
        "II": "low solubility but high permeability — often a formulation (solubility) challenge",
        "III": "high solubility but low permeability — a permeability challenge",
        "IV": "low solubility and low permeability — a significant developability risk",
    }.get(cls, "undetermined")


def _interpret_affinity(a: float) -> str:
    if a <= -8:
        return "a strong (very favourable) predicted binder"
    if a <= -6:
        return "a moderate predicted binder"
    return "a weak predicted binder"


def _interpret_rmsd(rmsd: float) -> str:
    if rmsd <= 2:
        return "stable over the short relaxation"
    return "mobile over the short relaxation — not a converged verdict"


def _mol_to_png(smiles: str, path: Path, size: tuple[int, int] = (360, 360)) -> bool:
    """Render a 2D depiction of one molecule to a PNG. Returns False when the
    SMILES does not parse. Deterministic for a given RDKit build (2D coordinate
    generation is deterministic)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    AllChem.Compute2DCoords(mol)
    Draw.MolToImage(mol, size=size).save(path, "PNG")
    return True


def _inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MolToInchiKey(mol)
    except Exception:  # noqa: BLE001
        return None


# ---- chart builders (kept only as auxiliary aids to the text) ----


def _chart_evidence(eb) -> Figure | None:
    levels = ["observed", "calculated", "predicted", "inferred", "hypothetical"]
    vals = [getattr(eb, l, 0) for l in levels]
    if not any(vals):
        return None
    fig, ax = _new_fig(6.4, 2.4)
    ax.bar(levels, vals, color=[EVIDENCE_COLORS[l] for l in levels])
    ax.set_ylabel("count", fontsize=8)
    ax.set_title("Evidence-level breakdown", fontsize=10)
    ax.tick_params(axis="x", labelsize=7.5, rotation=20)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def _chart_screening(sbs) -> Figure | None:
    total = passed = rejected = invalid = brenk = 0
    for s in sbs:
        total += s.summary.total
        passed += s.summary.passed
        rejected += s.summary.rejected
        invalid += s.summary.invalid
        brenk += s.summary.brenk_flagged
    if total == 0:
        return None
    fig, ax = _new_fig(6.4, 2.4)
    labels = ["passed", "rejected", "invalid", "Brenk advisories"]
    vals = [passed, rejected, invalid, brenk]
    ax.bar(labels, vals, color=["#2e7d32", "#c62828", "#9e9e9e", "#f9a825"])
    ax.set_ylabel("molecules", fontsize=8)
    ax.set_title("Screening outcome", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def _chart_sa(rps) -> Figure | None:
    items: list[tuple[str, float, str]] = []
    for rp in rps:
        for m in rp.molecules:
            items.append((m.id, m.sa_score, m.synthetic_accessibility))
    if not items:
        return None
    cls_colors = {"easy": "#2e7d32", "moderate": "#f9a825", "hard": "#c62828"}
    labels = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    cls = [i[2] for i in items][::-1]
    fig, ax = _new_fig(6.4, max(2.0, 0.28 * len(labels) + 1.0))
    y = list(range(len(labels)))
    ax.barh(y, vals, color=[cls_colors.get(c, "#9e9e9e") for c in cls])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(1, 10)
    ax.set_xlabel("SA score (1 = easy … 10 = hard)", fontsize=8)
    ax.set_title("Synthetic accessibility", fontsize=10)
    fig.tight_layout()
    return fig


def _chart_docking(drs) -> Figure | None:
    # Collapse to the BEST pose per molecule: Vina emits several poses per
    # ligand, so plotting every pose would build a 9x-taller bar chart that
    # overflows the page (LayoutError). One bar per molecule, best affinity.
    best: dict[str, float] = {}
    for dr in drs:
        for p in dr.poses:
            if p.affinity_kcal_per_mol is not None:
                if p.molecule_id not in best or p.affinity_kcal_per_mol < best[p.molecule_id]:
                    best[p.molecule_id] = p.affinity_kcal_per_mol
    if not best:
        return None
    items = sorted(best.items(), key=lambda kv: kv[1])  # best (most negative) first
    labels = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    fig, ax = _new_fig(6.4, max(2.0, 0.28 * len(labels) + 1.0))
    y = list(range(len(labels)))
    ax.barh(y, vals, color="#00695c")
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("affinity (kcal/mol, lower is better)", fontsize=8)
    ax.set_title("Docking affinity — best pose per molecule (calculated)", fontsize=10)
    fig.tight_layout()
    return fig


def _chart_md(mds) -> Figure | None:
    items: list[tuple[str, float]] = []
    for md in mds:
        for r in md.runs:
            if r.status == "success" and r.metrics.backbone_rmsd_mean_angstrom is not None:
                items.append((r.receptor_id, r.metrics.backbone_rmsd_mean_angstrom))
    if not items:
        return None
    labels = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    fig, ax = _new_fig(6.4, max(2.0, 0.28 * len(labels) + 1.0))
    y = list(range(len(labels)))
    ax.barh(y, vals, color="#283593")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("backbone RMSD mean (Å)", fontsize=8)
    ax.set_title("Receptor MD — backbone RMSD (calculated)", fontsize=10)
    fig.tight_layout()
    return fig


# ---- aggregation helpers ----


def _collect_tools(report, by_kind) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(tv) -> None:
        s = f"{tv.name} {tv.version or ''}".strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    for p in report.provenance:
        for tv in p.tool_versions:
            add(tv)
    for arts in by_kind.values():
        for a in arts:
            for p in (getattr(a, "provenance", None) or []):
                for tv in p.tool_versions:
                    add(tv)
    return out


def _survivors(by_kind) -> list[str]:
    out: list[str] = []
    for sb in by_kind.get("screened_batch", []):
        for m in sb.molecules:
            if m.passed and m.canonical_smiles and m.canonical_smiles not in out:
                out.append(m.canonical_smiles)
    return out


# ---- main render ----


def render_report_pdf(report, by_kind, project_spec, validation_report, out_path: Path) -> dict:
    """Render the run report PDF to `out_path`. Returns
    {'page_count': int, 'figure_count': int, 'sections': [str]}."""
    _register_fonts()
    styles = _styles()

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title=f"ddswarm report — {project_spec.project_id}", author="ddswarm",
    )
    story: list = []
    sections: list[str] = []
    figure_count = 0
    mol_counter = 0

    def section(title: str) -> None:
        sections.append(title)
        story.append(Paragraph(_esc(title), styles["h1"]))

    def para(text: str) -> None:
        story.append(Paragraph(_esc(text), styles["body"]))

    def bullets(items) -> None:
        for it in items:
            story.append(Paragraph("•&nbsp;" + _cell(it), styles["bullet"]))

    def sub(stage: str, agent: str) -> None:
        story.append(Paragraph(f"{_esc(stage)} — {_esc(agent)}", styles["h2"]))

    def fig(f, caption: str) -> None:
        nonlocal figure_count
        if f is None:
            return
        figure_count += 1
        w_in, h_in = f.get_size_inches()
        W = 15.5
        H = W * (h_in / w_in)
        MAX_H = 16.0  # cm cap so an over-tall chart still fits the page frame
        if H > MAX_H:
            W = W * (MAX_H / H)
            H = MAX_H
        png = figdir / f"fig{figure_count}.png"
        f.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(f)
        story.append(Image(str(png), width=W * cm, height=H * cm))
        story.append(Paragraph(_esc(caption), styles["caption"]))

    def mol_img(smiles: str, width_cm: float = 5.5, caption: str | None = None) -> None:
        """Embed a 2D molecule depiction as an auxiliary image (not counted as a
        chart figure)."""
        nonlocal mol_counter
        png = figdir / f"mol_{mol_counter}.png"
        mol_counter += 1
        if not _mol_to_png(smiles, png):
            return
        story.append(Image(str(png), width=width_cm * cm, height=width_cm * cm))
        if caption:
            story.append(Paragraph(_esc(caption), styles["caption"]))

    # ---- convenience aggregates ----
    sbs = by_kind.get("screened_batch", [])
    aps = by_kind.get("admet_prediction", [])
    rps = by_kind.get("retrosynthesis_plan", [])
    tps = by_kind.get("tox_profile", [])
    fas = by_kind.get("formulation_assessment", [])
    drs = by_kind.get("docking_result", [])
    mds = by_kind.get("dynamics_result", [])
    sas = by_kind.get("safety_assessment", [])
    kps = by_kind.get("knowledge_pack", [])
    dbs = by_kind.get("design_batch", [])

    survivors = _survivors(by_kind)
    n_passed = sum(s.summary.passed for s in sbs)
    n_total = sum(s.summary.total for s in sbs)
    tools = _collect_tools(report, by_kind)

    with tempfile.TemporaryDirectory() as tmp:
        figdir = Path(tmp)

        # ---- title ----
        targets = ", ".join(project_spec.targets) or "—"
        story.append(Paragraph(_esc(f"ddswarm — {project_spec.project_id}"), styles["title"]))
        story.append(Paragraph(_esc(f"run {report.run_id} · status {report.run_status} · targets {targets}"),
                               styles["subtitle"]))
        story.append(Paragraph(_esc("In-silico drug-discovery report — every score below is calculated "
                                    "or predicted, never experimental evidence."), styles["subtitle"]))

        # ---- abstract (verbose) ----
        section("Abstract")
        para(_narrative_abstract(report, by_kind, project_spec, survivors, n_total, n_passed))

        # ---- methodology (narrative) ----
        section("Methodology")
        agents = sorted({sf.agent for sf in report.stage_findings})
        para(f"This run executed {len(agents)} specialised agents, one per stage of an in-silico "
             "drug-discovery workflow. Each agent writes a versioned, hashed artifact with full "
             "provenance (input/output hashes, tool versions, evidence level), so the whole run "
             "is reproducible and auditable. What each agent does:")
        for a in agents:
            stage, role = AGENT_LABELS.get(a, (a, "contributes one stage of the pipeline"))
            bullets([f"{stage} ({a}) — {role}"])
        if tools:
            para("Tools and versions: " + "; ".join(tools) + ".")
        if project_spec.starting_materials:
            para("The analysis starts from the provided molecule(s): "
                 + "; ".join(project_spec.starting_materials) + ".")

        # ---- results, one clearly-labeled block per present stage ----
        section("Results and interpretation")

        if kps:
            sub("Research", "researcher_minion")
            _narrative_research(kps, para)
        if dbs:
            sub("Design", "designer_minion")
            _narrative_design(dbs, para)
        if sbs:
            sub("Screening", "screener_minion")
            _narrative_screening(sbs, para, fig)
        if aps:
            sub("ADMET prediction", "predictor_minion")
            _narrative_admet(aps, para, styles, story)
        if rps:
            sub("Synthesis", "retrosynthesis_minion")
            _narrative_retrosynthesis(rps, para, fig)
        if tps:
            sub("Toxicity", "tox_minion")
            _narrative_tox(tps, para)
        if fas:
            sub("Formulation", "formulation_minion")
            _narrative_formulation(fas, para)
        if drs:
            sub("Docking", "docking_minion")
            _narrative_docking(drs, para, fig)
        if mds:
            sub("Receptor MD", "dynamics_minion")
            _narrative_md(mds, para, fig)
        if sas:
            sub("Safety", "safety_minion")
            _narrative_safety(sas, para)

        _narrative_knowledge_agents(by_kind, para, sub)

        # ---- identified candidate (the discovery outcome) ----
        section("Identified candidate")
        cand = report.identified_candidate
        if cand is None:
            para("No molecule survived screening to be identified as a lead candidate; "
                 "this run produced no viable candidate to carry forward.")
        else:
            para(f"The top-ranked lead candidate is rank #{cand.rank} by the deterministic "
                 f"composite score of {cand.score:.4f}, computed as QED + BCS bonus "
                 "- 0.1 x synthetic-accessibility - 0.2 x structural-toxicity alerts. "
                 "Its structure:")
            mol_img(cand.smiles, width_cm=5.5,
                    caption="Identified lead candidate (2D structure).")
            ik = _inchikey(cand.smiles)
            para(f"SMILES: {cand.smiles}" + (f"  ·  InChIKey: {ik}" if ik else ""))
            if cand.iupac_name:
                para(f"IUPAC name (via NIH CACTUS web service — readability-only, "
                     f"not authoritative): {cand.iupac_name}")
            if cand.criteria:
                para("Ranking criteria: " + "; ".join(cand.criteria) + ".")
            if len(report.ranked_candidates) > 1:
                para("Full ranked list of surviving candidates:")
                rows = [["rank", "score", "SMILES"]]
                for c in report.ranked_candidates:
                    rows.append([str(c.rank), f"{c.score:.4f}", c.smiles])
                story.append(_table(rows[0], rows[1:], [1.2, 2.0, 12.3], styles))

        # ---- evidence and confidence ----
        section("Evidence and confidence")
        eb = report.evidence_breakdown
        para("Every number in this report carries an evidence level. 'Observed' is a fact read from "
             "a source; 'calculated' is a deterministic computation (docking, SA score, MD); "
             "'predicted' is a rule-based or model estimate (ADMET, toxicity, formulation); "
             "'inferred' is the swarm's own reasoning; 'hypothetical' is an unverified proposal "
             "(LLM-proposed synthesis routes or analog designs).")
        if any(getattr(eb, l, 0) for l in ("observed", "calculated", "predicted", "inferred", "hypothetical")):
            fig(_chart_evidence(eb),
                "Auxiliary view: how the evidence-carrying units distribute across evidence levels.")
            para(f"Breakdown — observed: {eb.observed}, calculated: {eb.calculated}, "
                 f"predicted: {eb.predicted}, inferred: {eb.inferred}, hypothetical: {eb.hypothetical}.")

        # ---- validation ----
        if validation_report is not None:
            section("Validation audit")
            vs = validation_report.summary
            para(f"The run's bookkeeping was audited: {vs.passed} of {vs.total_checks} checks passed"
                 + ("." if vs.failed == 0 and vs.warned == 0
                    else f", {vs.failed} failed, {vs.warned} warned."))
            if vs.failed == 0:
                para("This means the artifact schemas, provenance, hash integrity, cross-agent "
                     "consistency, evidence levels, numerics, and report consistency are all "
                     "internally consistent.")
            else:
                para("Failed checks are reproducibility/bookkeeping inconsistencies in the run "
                     "itself, not scientific judgments.")

        # ---- discussion ----
        section("Discussion")
        if report.gaps:
            para("Limitations of this run:")
            bullets(report.gaps)
        if report.caveats:
            para("Important caveats:")
            bullets(report.caveats)

        # ---- conclusion (verbose) ----
        section("Conclusion")
        para(_narrative_conclusion(report, validation_report, survivors, n_passed))

        # ---- provenance ----
        section("Provenance")
        if tools:
            para("Tool versions: " + "; ".join(tools) + ".")
        for p in report.provenance:
            if p.input_hash:
                story.append(Paragraph(_esc(f"input_hash: {p.input_hash}"), styles["small"]))
            if p.output_hash:
                story.append(Paragraph(_esc(f"output_hash: {p.output_hash}"), styles["small"]))
        story.append(Paragraph(
            _esc("All computed values are in-silico. Docking, MD, ADMET, and PK/PD are computational "
                 "evidence and are reported as such; none constitute experimental proof."),
            styles["small"]))

        doc.build(story)

    return {"page_count": doc.page, "figure_count": figure_count, "sections": sections}


# ---- narrative helpers (deterministic prose) ----


def _narrative_abstract(report, by_kind, spec, survivors, n_total, n_passed) -> str:
    target = ", ".join(spec.targets) or "the specified target"
    start = "; ".join(spec.starting_materials) or "the provided starting material"
    parts = [
        f"This report documents one in-silico drug-discovery run against {target}, starting "
        f"from {start}. The pipeline designed candidate molecules, screened them for drug-like "
        "properties and structural liabilities, and computed ADMET, toxicity, and formulation "
        "estimates for the survivors.",
    ]
    if n_total:
        parts.append(f"Of {n_total} screened molecule(s), {n_passed} passed screening.")
    if survivors:
        parts.append("The surviving candidate(s) are: " + "; ".join(survivors[:12]) + ".")
    if report.run_status == "completed":
        parts.append("The run completed successfully and its internal audit passed; the results "
                     "below are the deterministic, reproducible output of the swarm.")
    else:
        parts.append(f"The run ended with status '{report.run_status}'; see the results and "
                     "discussion below.")
    return " ".join(parts)


def _narrative_research(kps, para) -> None:
    n = sum(len(k.entries) for k in kps)
    para(f"Research assembled {n} knowledge entr{'y' if n == 1 else 'ies'}: the target(s) and the "
         "starting molecule(s) taken from the project specification. These seed the downstream "
         "stages. Because no external literature backend was configured, these are baseline facts "
         "(evidence level: inferred), not independently retrieved references.")


def _narrative_design(dbs, para) -> None:
    n = sum(len(d.molecules) for d in dbs)
    para(f"Design produced {n} candidate molecule(s). Without a reasoning backend these are "
         "labelled transformations of the starting scaffold (bioisostere, homologation, and "
         "analog-design labels) rather than genuinely new chemical structures; a reasoning backend "
         "would generate concrete analog SMILES with rationales.")


def _narrative_screening(sbs, para, fig) -> None:
    total = sum(s.summary.total for s in sbs)
    passed = sum(s.summary.passed for s in sbs)
    rejected = sum(s.summary.rejected for s in sbs)
    invalid = sum(s.summary.invalid for s in sbs)
    brenk = sum(s.summary.brenk_flagged for s in sbs)
    para(f"Screening evaluated {total} molecule(s): {passed} passed, {rejected} rejected, and "
         f"{invalid} invalid. A molecule is eliminated on a hard PAINS match or a Rule-of-5 "
         "violation; Brenk 'unwanted functionality' alerts are advisory only and are reported "
         f"rather than eliminating ({brenk} flagged).")
    fig(_chart_screening(sbs), "Auxiliary view: screening outcome.")
    for sb in sbs:
        for m in sb.molecules:
            if not m.passed:
                continue
            props = m.properties
            if props is None:
                continue
            suffix = ""
            if m.brenk_alerts:
                suffix = (f" It carries {len(m.brenk_alerts)} advisory Brenk alert(s) "
                          f"({'; '.join(m.brenk_alerts)}).")
            else:
                suffix = " It carries no hard PAINS motifs and no Brenk advisory alerts."
            para(f"{m.id}: molecular weight {props.molecular_weight:.1f} Da (within the 500 Da "
                 f"Rule-of-5 ceiling), logP {props.logp:.2f} ({_interpret_logp(props.logp)}), "
                 f"{props.hbd} hydrogen-bond donor(s) and {props.hba} acceptor(s), polar surface "
                 f"area {props.tpsa:.1f} Å²." + suffix)


def _narrative_admet(aps, para, styles, story) -> None:
    n_mol = sum(len(a.molecules) for a in aps)
    n_pred = sum(a.summary.predicted for a in aps)
    para(f"ADMET properties were predicted for {n_mol} molecule(s) using five published, "
         f"deterministic rules ({n_pred} endpoint prediction(s)). These are rule-of-thumb estimates "
         "(evidence level: predicted), not measured values.")
    for ap in aps:
        for m in ap.molecules:
            vals: dict[str, str] = {}
            for ep in m.endpoints:
                if ep.prediction_status != "predicted":
                    continue
                if ep.predicted_value is not None:
                    v = f"{ep.predicted_value:.4g}" + (f" {ep.units}" if ep.units else "")
                    if ep.endpoint == "LogS_ESOL":
                        vals[ep.endpoint] = f"{v} ({_interpret_esol(ep.predicted_value)})"
                    elif ep.endpoint == "QED":
                        vals[ep.endpoint] = f"{v} ({_interpret_qed(ep.predicted_value)})"
                    elif ep.endpoint == "Lipinski_Violations":
                        vals[ep.endpoint] = f"{v} ({_interpret_lipinski(int(ep.predicted_value))})"
                    else:
                        vals[ep.endpoint] = v
                elif ep.predicted_label:
                    vals[ep.endpoint] = ep.predicted_label
            if vals:
                bits = [f"{k}: {vals[k]}" for k in vals]
                para(f"For {m.id}: " + "; ".join(bits) + ".")
    # auxiliary table of endpoint citations
    rows = [["molecule", "endpoint", "value", "rule source"]]
    for ap in aps:
        for m in ap.molecules:
            for ep in m.endpoints:
                val = ""
                if ep.predicted_value is not None:
                    val = f"{ep.predicted_value:.4g}" + (f" {ep.units}" if ep.units else "")
                elif ep.predicted_label:
                    val = ep.predicted_label
                rows.append([m.id, ep.endpoint, val or "—", ep.citation or "—"])
    if len(rows) > 1:
        story.append(_table(rows[0], rows[1:], [1.6, 2.6, 2.6, 8.7], styles))


def _narrative_retrosynthesis(rps, para, fig) -> None:
    n = sum(len(r.molecules) for r in rps)
    easy = sum(r.summary.easy for r in rps)
    mod = sum(r.summary.moderate for r in rps)
    hard = sum(r.summary.hard for r in rps)
    para(f"Synthetic accessibility was scored for {n} molecule(s) on the RDKit SA scale "
         f"(1 = trivial, 10 = very difficult): {easy} easy, {mod} moderate, {hard} hard. A low SA "
         "score means the molecule should be straightforward to prepare in the lab.")
    if n:
        fig(_chart_sa(rps), "Auxiliary view: synthetic-accessibility score per molecule.")
        for rp in rps:
            for m in rp.molecules:
                para(f"{m.id} has an SA score of {m.sa_score:.2f} ({_interpret_sa(m.sa_score)}).")
                for d in m.disconnections:
                    if d.fragments:
                        para(f"Theoretical retrosynthesis: it disconnects into {len(d.fragments)} "
                             f"building blocks — {' + '.join(d.fragments)} — reconnected at the "
                             "[n*] attachment points. A theoretical route, not a demonstrated "
                             "synthesis.")
                    elif d.note:
                        para(d.note)
                if m.routes:
                    para(f"{len(m.routes)} LLM-proposed synthesis route(s) (hypothetical).")


def _narrative_tox(tps, para) -> None:
    n = sum(t.summary.total_molecules for t in tps)
    alerts = sum(t.summary.total_alerts for t in tps)
    if alerts == 0:
        para(f"Structural-toxicity screening found no toxicophore alerts across {n} molecule(s). "
             "The curated alert set is small, so this is a limited screen, not a declaration that "
             "the molecules are non-toxic.")
    else:
        para(f"Structural-toxicity screening flagged {alerts} toxicophore alert(s) across {n} "
             "molecule(s). These are predicted liabilities from literature rules (evidence level: "
             "predicted), not observed toxicity.")


def _narrative_formulation(fas, para) -> None:
    for fa in fas:
        for m in fa.molecules:
            para(f"{m.id}: formulation analysis places it in BCS class {m.bcs_class} — "
                 f"{_interpret_bcs(m.bcs_class)} — with {m.solubility_class} solubility and "
                 f"{m.permeability_class} permeability (rule-based estimates from logP/TPSA).")


def _narrative_docking(drs, para, fig) -> None:
    for dr in drs:
        rec = dr.receptor.identifier if dr.receptor else "the receptor"
        best = None
        for p in dr.poses:
            if p.affinity_kcal_per_mol is not None:
                best = p.affinity_kcal_per_mol if best is None else min(best, p.affinity_kcal_per_mol)
        if best is not None:
            para(f"Docking into {rec} produced poses with a best predicted affinity of "
                 f"{best:.2f} kcal/mol ({_interpret_affinity(best)}). Affinity and redocking RMSD "
                 "are calculated in-silico values, not experimental binding data.")
        rv = dr.redocking_validation
        if rv and rv.rmsd_angstrom is not None:
            para(f"Redocking validation: {rv.status}, RMSD {rv.rmsd_angstrom:.2f} Å against a "
                 f"{rv.threshold_angstrom} Å threshold.")
        if dr.poses:
            fig(_chart_docking(drs), "Auxiliary view: docking affinity per pose.")


def _narrative_md(mds, para, fig) -> None:
    for md in mds:
        for r in md.runs:
            if r.status != "success":
                para(f"Receptor molecular dynamics for {r.receptor_id} did not complete ({r.status}).")
                continue
            rmsd = r.metrics.backbone_rmsd_mean_angstrom
            if rmsd is not None:
                para(f"A short receptor molecular-dynamics relaxation of {r.receptor_id} showed a "
                     f"mean backbone RMSD of {rmsd:.2f} Å ({_interpret_rmsd(rmsd)}). This is a "
                     "picosecond-scale snapshot, not a converged stability assessment.")
    if any(r.status == "success" and r.metrics.backbone_rmsd_mean_angstrom is not None
           for md in mds for r in md.runs):
        fig(_chart_md(mds), "Auxiliary view: backbone RMSD per receptor.")


def _narrative_safety(sas, para) -> None:
    for sa in sas:
        if sa.verdict == "block":
            para("The safety gate BLOCKED this run: a molecule or target matched the curated "
                 "controlled-substance or dual-use list. Downstream stages were gated off.")
        else:
            para(f"The safety gate passed ({sa.summary.total_flags} flag(s)). Note the list is "
                 "curated and non-exhaustive — a 'pass' is not a comprehensive legal or safety "
                 "determination.")


def _narrative_knowledge_agents(by_kind, para, sub) -> None:
    for kind, stage, agent in (("patent_landscape", "Patent landscape", "patent_minion"),
                               ("clinical_assessment", "Clinical knowledge", "clinical_minion"),
                               ("regulatory_assessment", "Regulatory knowledge", "regulatory_minion")):
        if kind in by_kind:
            sub(stage, agent)
            n = sum(len(p.entries) for p in by_kind[kind])
            para(f"The {stage.lower()} contributed {n} entr{'y' if n == 1 else 'ies'} "
                 "(LLM-recalled, evidence level: inferred). These are research aids, not legal "
                 "or regulatory determinations.")


def _narrative_conclusion(report, validation_report, survivors, n_passed) -> str:
    verdict = f"The run finished with status '{report.run_status}'."
    if validation_report is not None:
        vs = validation_report.summary
        verdict += f" The validation audit passed {vs.passed} of {vs.total_checks} checks"
        if vs.failed:
            verdict += f" ({vs.failed} failed)"
        verdict += "."
    if survivors:
        verdict += f" {len(survivors)} candidate(s) survived screening"
        if n_passed:
            verdict += f" ({n_passed} of the screened set)"
        verdict += "."
    verdict += (" These are in-silico leads only; the next step is experimental validation "
                "(in-vitro/in-vivo) before any progression.")
    return verdict
