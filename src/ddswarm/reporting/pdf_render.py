"""Deterministic PDF renderer for the final run report ("scientific essay").

Pure function: takes the aggregated run artifacts (Report, ValidationReport, and
the by-kind artifact map) plus the ProjectSpec, and renders a self-contained PDF
with narrative text, tables, and matplotlib figures. No LLM, no network, no
randomness — the same inputs produce the same document content (page/figure
counts are deterministic). The raw PDF bytes are not guaranteed byte-identical
across reportlab/matplotlib builds, which is why the ReportDocument metadata
carries its own sha256 rather than promising a byte-stable artifact.

Figures are drawn with matplotlib's Agg backend and bundled DejaVu fonts, which
also back the PDF text via reportlab, so the whole document is self-contained
(no external font files) and Unicode-clean."""

from __future__ import annotations

import tempfile
from pathlib import Path
from xml.sax.saxutils import escape as _xmlescape

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
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

_FONT_NAME = "DejaVu"
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
        "body": ParagraphStyle("Body", fontName="DejaVu", fontSize=9.5, leading=13.5,
                               spaceAfter=5, splitLongWords=1),
        "bullet": ParagraphStyle("Bullet", parent=None, fontName="DejaVu", fontSize=9.5,
                                 leading=13, leftIndent=12, spaceAfter=2, splitLongWords=1),
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


# ---- chart builders (each returns a Figure, or None when there is no data) ----


def _chart_evidence(eb) -> Figure | None:
    levels = ["observed", "calculated", "predicted", "inferred", "hypothetical"]
    vals = [getattr(eb, l, 0) for l in levels]
    if not any(vals):
        return None
    fig, ax = _new_fig(6.6, 2.6)
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
    fig, ax = _new_fig(6.6, 2.6)
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


def _chart_admet(aps) -> Figure | None:
    items: list[tuple[str, float]] = []
    for ap in aps:
        for m in ap.molecules:
            for ep in m.endpoints:
                if ep.prediction_status == "predicted" and ep.predicted_value is not None:
                    label = f"{m.id} · {ep.endpoint}" + (f" ({ep.units})" if ep.units else "")
                    items.append((label, ep.predicted_value))
    if not items:
        return None
    items = items[:30]
    labels = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    fig, ax = _new_fig(6.6, max(2.0, 0.28 * len(labels) + 1.0))
    y = list(range(len(labels)))
    ax.barh(y, vals, color="#1565c0")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted value", fontsize=8)
    ax.set_title("ADMET endpoint predictions", fontsize=10)
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
    fig, ax = _new_fig(6.6, max(2.0, 0.28 * len(labels) + 1.0))
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
    items: list[tuple[str, float]] = []
    for dr in drs:
        for p in dr.poses:
            if p.affinity_kcal_per_mol is not None:
                items.append((f"{p.molecule_id} · rank{p.rank}", p.affinity_kcal_per_mol))
    if not items:
        return None
    labels = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    fig, ax = _new_fig(6.6, max(2.0, 0.28 * len(labels) + 1.0))
    y = list(range(len(labels)))
    ax.barh(y, vals, color="#00695c")
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("affinity (kcal/mol, lower is better)", fontsize=8)
    ax.set_title("Docking affinity (calculated)", fontsize=10)
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
    fig, ax = _new_fig(6.6, max(2.0, 0.28 * len(labels) + 1.0))
    y = list(range(len(labels)))
    ax.barh(y, vals, color="#283593")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("backbone RMSD mean (Å)", fontsize=8)
    ax.set_title("Receptor MD — backbone RMSD (calculated)", fontsize=10)
    fig.tight_layout()
    return fig


def _chart_properties(sbs) -> Figure | None:
    pts: list[tuple[float, float, bool]] = []
    for s in sbs:
        for m in s.molecules:
            if m.properties is not None:
                pts.append((m.properties.molecular_weight, m.properties.logp, m.passed))
    if not pts:
        return None
    fig, ax = _new_fig(6.6, 3.2)
    for passed, color, label in ((True, "#2e7d32", "passed"), (False, "#c62828", "rejected")):
        xs = [p[0] for p in pts if p[2] is passed]
        ys = [p[1] for p in pts if p[2] is passed]
        if xs:
            ax.scatter(xs, ys, c=color, label=label, s=22, alpha=0.85)
    ax.set_xlabel("molecular weight (Da)", fontsize=8)
    ax.set_ylabel("logP", fontsize=8)
    ax.set_title("Molecular properties of screened molecules", fontsize=10)
    ax.legend(fontsize=8)
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

    def section(title: str) -> None:
        sections.append(title)
        story.append(Paragraph(_esc(title), styles["h1"]))

    def fig(f, caption: str) -> None:
        nonlocal figure_count
        if f is None:
            return
        figure_count += 1
        w_in, h_in = f.get_size_inches()
        W = 16.0  # cm text width
        png = figdir / f"fig{figure_count}.png"
        f.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(f)
        story.append(Image(str(png), width=W * cm, height=W * (h_in / w_in) * cm))
        story.append(Paragraph(_esc(caption), styles["caption"]))

    def bullets(items, style_key="bullet") -> None:
        for it in items:
            story.append(Paragraph("•&nbsp;" + _cell(it), styles[style_key]))

    with tempfile.TemporaryDirectory() as tmp:
        figdir = Path(tmp)

        # ---- title ----
        story.append(Paragraph(_esc(f"ddswarm — {project_spec.project_id}"), styles["title"]))
        targets = ", ".join(project_spec.targets) or "—"
        story.append(Paragraph(_esc(f"run {report.run_id} · status {report.run_status} · targets {targets}"),
                               styles["subtitle"]))
        story.append(Paragraph(_esc("In-silico drug-discovery pipeline report — every score below is "
                                    "calculated or predicted, never experimental evidence."),
                               styles["subtitle"]))

        # ---- abstract ----
        section("Abstract")
        story.append(Paragraph(_cell(report.executive_summary), styles["body"]))

        # ---- methodology ----
        section("Methodology")
        agents = sorted({sf.agent for sf in report.stage_findings})
        story.append(Paragraph(_esc("Agents executed: " + ", ".join(agents) + "."), styles["body"]))
        tools = _collect_tools(report, by_kind)
        if tools:
            story.append(Paragraph(_esc("Tool versions: " + "; ".join(tools) + "."), styles["small"]))
        story.append(Paragraph(_esc("Input: " + "; ".join(project_spec.starting_materials)
                                    if project_spec.starting_materials
                                    else "no starting materials declared"), styles["body"]))

        # ---- stage overview ----
        section("Stage overview")
        rows = [["task", "agent", "status", "findings", "warnings"]]
        for sf in report.stage_findings:
            rows.append([sf.task_id, sf.agent, sf.status,
                         "\n".join(sf.findings) or "—",
                         "\n".join(sf.warnings) or "—"])
        story.append(_table(rows[0], rows[1:], [2.4, 2.8, 1.9, 4.9, 4.0], styles))

        # ---- research ----
        if "knowledge_pack" in by_kind:
            section("Research knowledge")
            rows = [["id", "type", "source", "evidence", "value / notes"]]
            for kp in by_kind["knowledge_pack"]:
                for e in kp.entries:
                    rows.append([e.id, e.type, e.source, e.evidence_level.value,
                                 (e.observed_value or e.notes or "—")])
            story.append(_table(rows[0], rows[1:], [1.8, 1.6, 2.6, 1.9, 8.1], styles))

        # ---- design ----
        if "design_batch" in by_kind:
            section("Molecular design")
            rows = [["id", "smiles", "transformation", "evidence", "rationale"]]
            for db in by_kind["design_batch"]:
                for m in db.molecules:
                    rows.append([m.id, m.smiles, m.transformation or "—",
                                 m.evidence_level.value, m.rationale or "—"])
            story.append(_table(rows[0], rows[1:], [1.5, 5.6, 2.3, 1.8, 4.8], styles))

        # ---- screening ----
        if "screened_batch" in by_kind:
            section("Screening")
            fig(_chart_screening(by_kind["screened_batch"]),
                "Screening outcome. PAINS matches are hard (eliminating); Brenk 'unwanted "
                "functionality' alerts are advisory and reported, not eliminating.")
            fig(_chart_properties(by_kind["screened_batch"]),
                "Molecular weight vs logP for screened molecules, colored by pass/reject.")
            rows = [["id", "canonical SMILES", "passed", "Ro5 violations", "PAINS", "Brenk"]]
            for sb in by_kind["screened_batch"]:
                for m in sb.molecules:
                    rows.append([m.id, m.canonical_smiles or m.input_smiles,
                                 "yes" if m.passed else "no",
                                 "; ".join(m.rule_of_five_violations) or "—",
                                 "; ".join(m.structural_alerts) or "—",
                                 "; ".join(m.brenk_alerts) or "—"])
            story.append(_table(rows[0], rows[1:], [1.2, 6.4, 1.1, 2.3, 2.5, 2.5], styles))

        # ---- ADMET ----
        if "admet_prediction" in by_kind:
            section("ADMET prediction")
            fig(_chart_admet(by_kind["admet_prediction"]),
                "Numeric endpoint predictions (rule-based; evidence_level = predicted).")
            rows = [["molecule", "endpoint", "value", "label", "status", "citation"]]
            for ap in by_kind["admet_prediction"]:
                for m in ap.molecules:
                    for ep in m.endpoints:
                        val = ""
                        if ep.predicted_value is not None:
                            val = f"{ep.predicted_value:.4g}"
                            if ep.units:
                                val += f" {ep.units}"
                        rows.append([m.id, ep.endpoint, val or "—",
                                     ep.predicted_label or "—", ep.prediction_status,
                                     ep.citation or "—"])
            story.append(_table(rows[0], rows[1:], [1.4, 2.2, 2.4, 2.0, 2.0, 6.0], styles))

        # ---- retrosynthesis ----
        if "retrosynthesis_plan" in by_kind:
            section("Retrosynthesis")
            fig(_chart_sa(by_kind["retrosynthesis_plan"]),
                "RDKit synthetic-accessibility (SA) score per molecule (calculated); "
                "synthesis routes, when present, are LLM-proposed (hypothetical).")
            rows = [["id", "canonical SMILES", "SA score", "accessibility", "routes"]]
            for rp in by_kind["retrosynthesis_plan"]:
                for m in rp.molecules:
                    rows.append([m.id, m.canonical_smiles, f"{m.sa_score:.2f}",
                                 m.synthetic_accessibility, str(len(m.routes))])
            story.append(_table(rows[0], rows[1:], [1.2, 6.2, 1.6, 2.4, 1.6], styles))

        # ---- toxicity ----
        if "tox_profile" in by_kind:
            section("Structural toxicity alerts")
            rows = [["id", "alerts", "count", "risk note"]]
            for tp in by_kind["tox_profile"]:
                for m in tp.molecules:
                    alerts = "; ".join(f"{a.name} ({a.endpoint})" for a in m.alerts)
                    rows.append([m.id, alerts or "—", str(m.liability_count), m.risk_note or "—"])
            story.append(_table(rows[0], rows[1:], [1.2, 8.4, 1.2, 5.2], styles))

        # ---- formulation ----
        if "formulation_assessment" in by_kind:
            section("Formulation / developability")
            rows = [["id", "logP", "TPSA", "solubility", "permeability", "BCS"]]
            for fa in by_kind["formulation_assessment"]:
                for m in fa.molecules:
                    rows.append([m.id,
                                 f"{m.logp:.2f}" if m.logp is not None else "—",
                                 f"{m.tpsa:.1f}" if m.tpsa is not None else "—",
                                 m.solubility_class, m.permeability_class, m.bcs_class])
            story.append(_table(rows[0], rows[1:], [1.2, 1.4, 1.5, 2.4, 2.5, 1.4], styles))

        # ---- docking ----
        if "docking_result" in by_kind:
            section("Docking")
            fig(_chart_docking(by_kind["docking_result"]),
                "Docking affinity per pose (calculated); a failed redocking (RMSD > 2 Å) is reported as failed.")
            rows = [["receptor", "molecule", "rank", "affinity (kcal/mol)", "RMSD l.b. / u.b."]]
            for dr in by_kind["docking_result"]:
                rec = dr.receptor.identifier if dr.receptor else "—"
                for p in dr.poses:
                    rlb = f"{p.rmsd_lower_bound:.2f}" if p.rmsd_lower_bound is not None else "—"
                    rub = f"{p.rmsd_upper_bound:.2f}" if p.rmsd_upper_bound is not None else "—"
                    rows.append([rec, p.molecule_id, str(p.rank),
                                 f"{p.affinity_kcal_per_mol:.2f}" if p.affinity_kcal_per_mol is not None else "—",
                                 f"{rlb} / {rub}"])
            story.append(_table(rows[0], rows[1:], [1.8, 2.0, 1.0, 3.4, 3.0], styles))
            rv = by_kind["docking_result"][0].redocking_validation
            if rv and rv.rmsd_angstrom is not None:
                story.append(Paragraph(
                    _esc(f"Redocking validation: {rv.status}, RMSD {rv.rmsd_angstrom:.2f} Å "
                         f"(threshold {rv.threshold_angstrom} Å)."), styles["small"]))

        # ---- dynamics ----
        if "dynamics_result" in by_kind:
            section("Receptor molecular dynamics")
            fig(_chart_md(by_kind["dynamics_result"]),
                "Backbone RMSD mean per receptor (calculated); a short relaxation snapshot, "
                "not converged production MD.")
            rows = [["receptor", "status", "backbone RMSD mean/max (Å)",
                     "energy mean ± std (kJ/mol)"]]
            for md in by_kind["dynamics_result"]:
                for r in md.runs:
                    mtr = r.metrics
                    rmsd = "—" if mtr.backbone_rmsd_mean_angstrom is None else (
                        f"{mtr.backbone_rmsd_mean_angstrom:.2f} / "
                        f"{(mtr.backbone_rmsd_max_angstrom or 0):.2f}")
                    en = "—" if mtr.potential_energy_mean_kj_per_mol is None else (
                        f"{mtr.potential_energy_mean_kj_per_mol:.0f} ± "
                        f"{(mtr.potential_energy_std_kj_per_mol or 0):.0f}")
                    rows.append([r.receptor_id, r.status, rmsd, en])
            story.append(_table(rows[0], rows[1:], [2.0, 1.8, 4.0, 4.4], styles))

        # ---- safety ----
        if "safety_assessment" in by_kind:
            section("Safety")
            for sa in by_kind["safety_assessment"]:
                story.append(Paragraph(
                    _esc(f"Verdict: {sa.verdict} ({sa.summary.total_flags} flag(s); "
                         f"block={sa.summary.block}, flag={sa.summary.flag})."), styles["body"]))
            rows = [["id", "kind", "level", "subject", "reason", "source"]]
            for sa in by_kind["safety_assessment"]:
                for f in sa.flags:
                    rows.append([f.id, f.kind, f.level, f.subject, f.reason, f.source])
            if len(rows) > 1:
                story.append(_table(rows[0], rows[1:], [1.2, 2.2, 1.1, 2.2, 5.6, 3.7], styles))

        # ---- knowledge agents (opt-in) ----
        for kind, title, id_cols in (
            ("patent_landscape", "Patent landscape",
             [["id", "evidence", "note"]],),
            ("clinical_assessment", "Clinical assessment",
             [["id", "phase", "evidence", "note"]],),
            ("regulatory_assessment", "Regulatory assessment",
             [["id", "evidence", "note"]],),
        ):
            if kind in by_kind:
                section(title)
                rows = [id_cols[0]]
                for payload in by_kind[kind]:
                    for e in payload.entries:
                        note = getattr(e, "note", None) or getattr(e, "observed_value", None) or "—"
                        if kind == "clinical_assessment":
                            rows.append([e.id, getattr(e, "phase", "—"),
                                         e.evidence_level.value, note])
                        else:
                            rows.append([e.id, e.evidence_level.value, note])
                widths = [1.4, 1.6, 1.9, 11.1] if kind == "clinical_assessment" else [1.4, 1.9, 12.7]
                story.append(_table(rows[0], rows[1:], widths, styles))

        # ---- evidence breakdown ----
        section("Evidence breakdown")
        fig(_chart_evidence(report.evidence_breakdown),
            "Count of evidence-carrying units by evidence level (coarse rollup; individual "
            "entries carry finer-grained levels).")
        eb = report.evidence_breakdown
        rows = [["observed", "calculated", "predicted", "inferred", "hypothetical"],
                [str(eb.observed), str(eb.calculated), str(eb.predicted),
                 str(eb.inferred), str(eb.hypothetical)]]
        story.append(_table(rows[0], rows[1:], [3.2, 3.2, 3.2, 3.2, 3.2], styles))

        # ---- validation ----
        if validation_report is not None:
            section("Validation audit")
            vs = validation_report.summary
            story.append(Paragraph(
                _esc(f"{vs.passed}/{vs.total_checks} checks passed, {vs.failed} failed, "
                     f"{vs.warned} warned."), styles["body"]))
            rows = [["check", "status", "detail"]]
            for c in validation_report.checks:
                rows.append([c.name, c.status, c.detail or "—"])
            story.append(_table(rows[0], rows[1:], [4.5, 1.5, 10.0], styles))

        # ---- discussion ----
        section("Discussion — gaps and caveats")
        story.append(Paragraph(_esc("Gaps:"), styles["h2"]))
        bullets(report.gaps or ["none reported"])
        story.append(Paragraph(_esc("Caveats:"), styles["h2"]))
        bullets(report.caveats or ["none reported"])

        # ---- conclusion ----
        section("Conclusion")
        verdict = f"Run status: {report.run_status}."
        if validation_report is not None:
            vs = validation_report.summary
            verdict += f" Validation: {vs.passed}/{vs.total_checks} checks passed, {vs.failed} failed."
        story.append(Paragraph(_esc(verdict), styles["body"]))
        survivors = _survivors(by_kind)
        if survivors:
            story.append(Paragraph(
                _esc("Surviving candidates: " + "; ".join(survivors[:12]) + "."), styles["body"]))
        story.append(Paragraph(_esc("Next steps:"), styles["h2"]))
        bullets(report.next_steps or ["validate top candidates experimentally before progressing"])

        # ---- provenance ----
        section("Provenance & references")
        if tools:
            story.append(Paragraph(_esc("Tool versions: " + "; ".join(tools) + "."), styles["small"]))
        for p in report.provenance:
            if p.input_hash:
                story.append(Paragraph(_esc(f"input_hash: {p.input_hash}"), styles["small"]))
            if p.output_hash:
                story.append(Paragraph(_esc(f"output_hash: {p.output_hash}"), styles["small"]))
        story.append(Paragraph(
            _esc("All computed values are in-silico. Docking, MD, ADMET, and PK/PD are "
                 "computational evidence and are reported as such; none constitute "
                 "experimental proof."), styles["small"]))

        doc.build(story)

    page_count = doc.page
    return {"page_count": page_count, "figure_count": figure_count, "sections": sections}
