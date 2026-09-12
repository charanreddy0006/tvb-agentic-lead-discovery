"""Streamlit presentation layer over the existing TVB backend pipeline."""
import os
from html import escape
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from core.lead_models import QualifiedLead
from services.mock_pipeline import run_mock_pipeline, validate_pipeline_config

st.set_page_config(page_title="TVB Agentic Lead Discovery", page_icon="TVB", layout="wide")

st.markdown("""
<style>
:root {
    --ink: #14213d;
    --muted: #60708b;
    --line: #dce4ef;
    --surface: #ffffff;
    --surface-soft: #f4f7fb;
    --blue: #2563eb;
    --blue-soft: #eaf1ff;
    --green: #138a68;
    --green-soft: #e8f7f0;
    --amber: #a86613;
    --amber-soft: #fff5e5;
}
.stApp { background: #f7f9fc; color: var(--ink); overflow-x: hidden; }
[data-testid="stSidebar"] { background: #eef3f9; border-right: 1px solid var(--line); min-width: 260px !important; width: 260px !important; }
[data-testid="stSidebar"] > div:first-child { width: 260px !important; }
[data-testid="stSidebar"] h2 { color: var(--ink); letter-spacing: -.02em; }
.block-container { box-sizing: border-box; max-width: none; padding: 2.4rem clamp(1rem, 3vw, 3.2rem) 1.5rem; width: 100%; }
.hero { padding: .4rem 0 1.4rem; }
.eyebrow { color: var(--blue); font-size: .74rem; font-weight: 800; letter-spacing: .18em; margin-bottom: .7rem; }
.hero h1 { color: var(--ink); font-size: clamp(2rem, 5vw, 4.35rem); letter-spacing: -.055em; line-height: .98; margin: 0; max-width: 100%; overflow-wrap: anywhere; }
.hero-sub { color: var(--muted); font-size: 1.08rem; line-height: 1.55; margin: 1rem 0 1.15rem; max-width: 670px; }
.badge, .mode-badge { align-items: center; background: var(--green-soft); border: 1px solid #bce9d7; border-radius: 999px; color: #0b6b50; display: inline-flex; font-size: .78rem; font-weight: 800; gap: .42rem; padding: .4rem .75rem; }
.mode-badge.live { background: var(--amber-soft); border-color: #f0d19f; color: #8a550f; }
.mode-badge.demo { background: var(--blue-soft); border-color: #c6d8ff; color: #1f55bd; }
.value-strip, .stage-rail, .fact-grid { display: grid; gap: .75rem; }
.value-strip { grid-template-columns: repeat(auto-fit, minmax(min(100%, 145px), 1fr)); margin: .5rem 0 2.3rem; }
.value-item { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; color: var(--ink); font-size: .84rem; font-weight: 750; padding: .8rem .7rem; text-align: center; }
.value-arrow { color: var(--blue); font-size: 1rem; margin-left: .35rem; }
.section-kicker { color: var(--blue); font-size: .72rem; font-weight: 850; letter-spacing: .15em; margin: 1.8rem 0 .35rem; text-transform: uppercase; }
.section-title { color: var(--ink); font-size: 1.65rem; font-weight: 780; letter-spacing: -.035em; margin: 0; }
.section-subtitle { color: var(--muted); margin: .3rem 0 1.05rem; }
.stage-rail { grid-template-columns: repeat(auto-fit, minmax(min(100%, 145px), 1fr)); margin: 1rem 0 2.15rem; }
.stage-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; min-height: 142px; padding: .9rem; }
.stage-number { color: var(--blue); font-size: .69rem; font-weight: 850; letter-spacing: .1em; }
.stage-icon { font-size: 1.35rem; margin: .55rem 0 .3rem; }
.stage-name { color: var(--ink); font-size: .91rem; font-weight: 800; }
.stage-copy { color: var(--muted); font-size: .75rem; line-height: 1.38; margin-top: .35rem; }
.fact-grid { grid-template-columns: repeat(3, 1fr); margin: .85rem 0 1.2rem; }
.fact-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.15rem; }
.fact-value { color: var(--blue); font-size: 1.75rem; font-weight: 850; letter-spacing: -.04em; }
.fact-label { color: var(--ink); font-size: .84rem; font-weight: 750; margin-top: .15rem; }
.fact-note { color: var(--muted); font-size: .73rem; margin-top: .28rem; }
.empty-state { background: linear-gradient(110deg, #eef4ff, #ffffff 70%); border: 1px solid #cadbff; border-radius: 12px; margin: .4rem 0 1.2rem; padding: 1.4rem 1.5rem; }
.empty-state h3 { color: var(--ink); font-size: 1.35rem; margin: 0 0 .3rem; }
.empty-state p { color: var(--muted); margin: 0; }
.kpi-grid { display: grid; gap: .7rem; grid-template-columns: repeat(4, 1fr); margin: .8rem 0 1.4rem; }
.kpi-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: .85rem 1rem; }
.kpi-label { color: var(--muted); font-size: .72rem; font-weight: 750; line-height: 1.25; }
.kpi-value { color: var(--ink); font-size: 1.65rem; font-weight: 850; margin-top: .35rem; }
.result-banner { align-items: center; background: var(--green-soft); border: 1px solid #bce9d7; border-radius: 10px; color: #0b6b50; display: flex; font-size: .9rem; font-weight: 750; gap: .5rem; margin: .6rem 0 1.2rem; padding: .78rem 1rem; }
.result-banner.partial { background: var(--amber-soft); border-color: #f0d19f; color: #8a550f; }
.evidence-label { color: var(--ink); font-size: .82rem; font-weight: 800; margin: .7rem 0 .25rem; }
.evidence-box { background: var(--surface-soft); border-left: 3px solid var(--blue); border-radius: 0 7px 7px 0; color: var(--ink); font-size: .88rem; line-height: 1.5; padding: .75rem .9rem; }
.footer { border-top: 1px solid var(--line); color: var(--muted); font-size: .78rem; margin-top: 2.8rem; padding: 1.1rem 0 .2rem; text-align: center; }
.footer strong { color: var(--ink); }
@media (max-width: 900px) { .block-container { padding: 1.5rem 1.1rem; } .stage-card { min-height: 128px; padding: .75rem; } .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .hero h1 { font-size: 2.45rem; } .stage-rail, .fact-grid, .value-strip, .kpi-grid { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)


MODE_DEMO = "Demo / Mock"
MODE_LIVE = "Live Agent"
PIPELINE_STAGES = [
    ("01", "🔎", "Discovery", "Finds companies dynamically through web search."),
    ("02", "📄", "Research", "Extracts structured company facts from sources."),
    ("03", "✅", "Qualification", "Checks financial, technology, and geography gates."),
    ("04", "👤", "Founder", "Identifies a CEO or co-founder from evidence."),
    ("05", "✉️", "Email", "Finds an explicitly supported professional email."),
    ("06", "🔐", "Verification", "Checks syntax, MX infrastructure, and association."),
    ("07", "🎯", "Final Leads", "Returns only leads that pass every gate."),
]


def lead_table(leads: List[QualifiedLead]) -> pd.DataFrame:
    """Present final lead models without exposing internal objects."""
    return pd.DataFrame([{
        "Company": item.company_name, "Industry": item.industry or "Not disclosed", "Location": item.location or "Not disclosed", "Founder / CEO": item.founder_name, "Role": item.founder_role, "Verified Email": item.email, "Funding": f"{item.funding_amount:,.0f} {item.funding_currency or ''}" if item.funding_amount else "Not disclosed", "Revenue": f"{item.revenue_amount:,.0f} {item.revenue_currency or ''}" if item.revenue_amount else "Not disclosed", "Verification": item.email_verification_status,
    } for item in leads])


def _save(leads: List[QualifiedLead], stats: Dict[str, int], rejections: Dict[str, int], demo: bool) -> None:
    st.session_state.result = {"leads": leads, "stats": stats, "rejections": rejections, "demo": demo}


def run_live(target: int, iterations: int, candidates: int) -> None:
    """Use production orchestration only after an explicit user action."""
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Live mode requires GROQ_API_KEY and TAVILY_API_KEY in the local environment.")
    from agent.lead_pipeline import LeadPipeline
    pipeline = LeadPipeline(target_leads=target, max_iterations=iterations, max_candidates=candidates)
    _save(pipeline.run(), pipeline.stats, {}, False)


def render_process_overview() -> None:
    stages = "".join(
        f"<div class='stage-card'><div class='stage-number'>{number}</div><div class='stage-icon'>{icon}</div>"
        f"<div class='stage-name'>{name}</div><div class='stage-copy'>{copy}</div></div>"
        for number, icon, name, copy in PIPELINE_STAGES
    )
    st.markdown("<div class='section-kicker'>Agent workflow</div><div class='section-title'>How the agent works</div>", unsafe_allow_html=True)
    st.markdown("<div class='stage-rail'>" + stages + "</div>", unsafe_allow_html=True)


def render_kpis(stats: Dict[str, int], leads: List[QualifiedLead]) -> None:
    fields = [
        ("Companies Discovered", stats.get("discovered", 0)),
        ("Companies Researched", stats.get("researched", 0)),
        ("Qualified Companies", stats.get("qualified", 0)),
        ("Founders Found", stats.get("founders", 0)),
        ("Emails Discovered", stats.get("emails", 0)),
        ("Verified Emails", stats.get("verified", 0)),
        ("Final Leads", len(leads)),
    ]
    cards = "".join(f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>" for label, value in fields)
    st.markdown("<div class='kpi-grid'>" + cards + "</div>", unsafe_allow_html=True)


def render_evidence(lead: QualifiedLead) -> None:
    evidence_tabs = st.tabs(["Overview", "Qualification Evidence", "Founder Evidence", "Email Verification", "Sources"])
    with evidence_tabs[0]:
        st.markdown(f"<div class='section-title'>{escape(lead.company_name)}</div>", unsafe_allow_html=True)
        st.write(lead.description or "No company overview was disclosed.")
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Industry**  \n{lead.industry or 'Not disclosed'}")
            st.markdown(f"**Location**  \n{lead.location or 'Not disclosed'}")
            st.markdown(f"**Founder**  \n{lead.founder_name} ({lead.founder_role})")
        with right:
            funding = f"{lead.funding_amount:,.0f} {lead.funding_currency or ''}".strip() if lead.funding_amount else "Not disclosed"
            revenue = f"{lead.revenue_amount:,.0f} {lead.revenue_currency or ''}".strip() if lead.revenue_amount else "Not disclosed"
            st.markdown(f"**Funding**  \n{funding}")
            st.markdown(f"**Revenue**  \n{revenue}")
            st.markdown(f"**Verification**  \n`{lead.email_verification_status}`")
    qualification_text = "\n".join(
        f"{label}: {lead.evidence[key]}"
        for label, key in (("Financial", "funding"), ("Technology", "technology"), ("Geography", "geography"))
        if lead.evidence.get(key)
    ) or "No qualification evidence available."
    evidence_items = [
        (evidence_tabs[1], "✓ Qualification Evidence", qualification_text),
        (evidence_tabs[2], "✓ Founder Evidence", lead.evidence.get("founder", "No founder evidence available.")),
        (evidence_tabs[3], "✓ Email Verification", lead.evidence.get("email", "No email evidence available.")),
    ]
    for tab, label, text in evidence_items:
        with tab:
            st.markdown(f"<div class='evidence-label'>{label}</div><div class='evidence-box'>{escape(text)}</div>", unsafe_allow_html=True)
            if tab == evidence_tabs[3]:
                st.caption(f"{lead.email} · status: {lead.email_verification_status}")
    with evidence_tabs[4]:
        urls = lead.company_source_urls + lead.founder_source_urls + ([lead.email_source_url] if lead.email_source_url else [])
        if urls:
            for url in dict.fromkeys(urls):
                st.markdown(f"🔗 [{url}]({url})")
        else:
            st.info("No source URLs were returned for this lead.")


if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    st.caption("Tune the bounded discovery run before you start.")
    target = st.number_input("Target Leads", 1, 50, 15)
    iterations = st.number_input("Max Iterations", 1, 30, 5)
    candidates = st.number_input("Max Candidates", 1, 300, 75, step=5)
    mode = st.radio("Execution Mode", [MODE_DEMO, MODE_LIVE], index=0)
    st.divider()
    if mode == MODE_DEMO:
        st.markdown("<span class='mode-badge demo'>● DEMO MODE</span>", unsafe_allow_html=True)
        st.caption("Uses local fictional data and consumes zero API credits.")
    else:
        st.markdown("<span class='mode-badge live'>● LIVE AGENT MODE</span>", unsafe_allow_html=True)
        st.caption("Uses Groq + Tavily and may consume API credits. It runs only after you click the action.")
    st.divider()
    st.caption("Final leads require qualification, founder evidence, source-backed email, and VERIFIED status.")

st.markdown("<section class='hero'><div class='eyebrow'>TVB VENTURE INTELLIGENCE</div><h1>TVB Agentic Lead Discovery</h1><p class='hero-sub'>Autonomously discover, qualify, and verify high-potential technology companies.</p><span class='badge'>● AI Lead Discovery Engine</span></section>", unsafe_allow_html=True)
st.markdown("<div class='value-strip'><div class='value-item'>Discover <span class='value-arrow'>→</span></div><div class='value-item'>Qualify <span class='value-arrow'>→</span></div><div class='value-item'>Find Founder <span class='value-arrow'>→</span></div><div class='value-item'>Verify Email <span class='value-arrow'>→</span></div><div class='value-item'>Deliver Leads</div></div>", unsafe_allow_html=True)
render_process_overview()

result = st.session_state.result
if result:
    current_label = "DEMO MODE · Fictional deterministic records" if result["demo"] else "LIVE AGENT MODE · External services enabled"
    st.markdown(f"<div class='mode-badge {'demo' if result['demo'] else 'live'}'>● {current_label}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='section-kicker'>Run console</div><div class='section-title'>Ready to discover</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='fact-grid'><div class='fact-card'><div class='fact-value'>{int(target)}</div><div class='fact-label'>Target verified leads</div><div class='fact-note'>Current run configuration</div></div><div class='fact-card'><div class='fact-value'>3</div><div class='fact-label'>Core validation gates</div><div class='fact-note'>Company, founder, email</div></div><div class='fact-card'><div class='fact-value'>7</div><div class='fact-label'>Agent pipeline stages</div><div class='fact-note'>Discovery through delivery</div></div></div>", unsafe_allow_html=True)

if st.button("🚀 Discover Qualified Leads", type="primary", width="stretch"):
    valid, message = validate_pipeline_config(int(target), int(iterations), int(candidates))
    if not valid:
        st.error(message)
    else:
        progress = st.progress(0, text="Preparing pipeline")
        try:
            if mode == MODE_DEMO:
                with st.status("Running local deterministic demo", expanded=True) as status:
                    for percent, stage in ((15, "Discovering companies"), (30, "Researching company information"), (45, "Validating qualification"), (60, "Discovering CEO / co-founder"), (75, "Discovering founder email"), (90, "Verifying email")):
                        status.write(stage)
                        progress.progress(percent, text=stage)
                    mock_result = run_mock_pipeline(int(target), int(iterations), int(candidates))
                    _save(mock_result.leads, mock_result.stats, mock_result.rejection_counts, True)
                    status.update(label="Demo lead list complete", state="complete")
            else:
                with st.status("Running existing LeadPipeline", expanded=True) as status:
                    status.write("Live backend started. Progress is provided by the existing bounded pipeline.")
                    run_live(int(target), int(iterations), int(candidates))
                    status.update(label="Live pipeline complete", state="complete")
            progress.progress(100, text="Building final lead list")
            st.rerun()
        except Exception as exc:
            progress.empty()
            st.error(f"The pipeline could not complete: {exc}")

result = st.session_state.result
if not result:
    st.markdown("<div class='empty-state'><h3>Ready when you are.</h3><p>Configure the pipeline in the sidebar and start a discovery run.</p></div>", unsafe_allow_html=True)
else:
    leads, stats = result["leads"], result["stats"]
    render_kpis(stats, leads)
    if len(leads) >= int(target):
        st.markdown(f"<div class='result-banner'>🎯 Target reached · {len(leads)} verified lead{'s' if len(leads) != 1 else ''} are ready.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-banner partial'>Discovery completed with {len(leads)} verified lead{'s' if len(leads) != 1 else ''}.</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>Verified output</div><div class='section-title'>🎯 Qualified Leads</div><div class='section-subtitle'>Companies that passed all qualification gates and have verified founder contact information.</div>", unsafe_allow_html=True)
    leads_tab, analytics_tab, details_tab, config_tab = st.tabs(["Qualified Leads", "📊 Pipeline Analytics", "Lead Details", "Run Configuration"])
    with leads_tab:
        if leads:
            table = lead_table(leads)
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button("Download CSV", table.to_csv(index=False), "final_verified_leads.csv", "text/csv")
        else:
            st.warning("The run completed but no final verified leads were found.")
    with analytics_tab:
        if result["rejections"]:
            st.markdown("<div class='section-title'>Why companies were rejected</div><div class='section-subtitle'>Observed rejection categories from this run.</div>", unsafe_allow_html=True)
            rejection_table = pd.DataFrame({"Reason": list(result["rejections"]), "Count": list(result["rejections"].values())})
            st.dataframe(rejection_table, width="stretch", hide_index=True)
        else:
            st.info("Detailed rejection categories were not returned for this run.")
        if stats.get("failures", 0) or stats.get("duplicates", 0):
            left, right = st.columns(2)
            left.metric("Pipeline failures", stats.get("failures", 0))
            right.metric("Duplicates skipped", stats.get("duplicates", 0))
    with details_tab:
        if leads:
            selected = st.selectbox("Select a company to inspect", [item.company_name for item in leads])
            lead = next(item for item in leads if item.company_name == selected)
            render_evidence(lead)
        else:
            st.info("Evidence appears here after a final verified lead is produced.")
    with config_tab:
        st.write(f"**Mode:** {MODE_DEMO if result['demo'] else MODE_LIVE}")
        st.write(f"**Target:** {target}  |  **Max iterations:** {iterations}  |  **Max candidates:** {candidates}")

st.markdown("<div class='footer'><strong>TVB Agentic Lead Discovery</strong><br>Discover&nbsp; • &nbsp;Qualify&nbsp; • &nbsp;Verify</div>", unsafe_allow_html=True)
