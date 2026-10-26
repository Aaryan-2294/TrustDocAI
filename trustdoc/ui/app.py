"""TrustDoc AI - Streamlit Web Application (Part 7).

Entry point for the browser interface.
Launch via:
    streamlit run trustdoc/ui/app.py
or:
    python -m trustdoc ui
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Ensure repository root is on sys.path when invoked directly via `streamlit run trustdoc/ui/app.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trustdoc.exceptions import (
    FileTooLargeError,
    InvalidQuestionError,
    NotFoundError,
    ParsingError,
    TrustDocError,
    VerificationError,
)
from trustdoc.models import EntityType, SourceType
from trustdoc.ui.components import (
    render_citations_list,
    render_claim_counts,
    render_evidence_drawer,
    render_findings_list,
    render_header,
    render_heuristic_disclaimer,
    render_status_badge,
    render_trust_score_card,
)
from trustdoc.ui.state import (
    create_project,
    delete_project,
    delete_source,
    get_app_context,
    get_project,
    get_project_stats,
    get_services,
    list_entities,
    list_projects,
    list_sources,
)
from trustdoc.workflows.types import AskStatus, GenerationKind, GenerationStatus


def main() -> None:
    st.set_page_config(
        page_title="TrustDoc AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ctx = get_app_context()
    services = get_services(ctx)

    # ------------------------------------------------------------------- Sidebar
    with st.sidebar:
        st.markdown("### 🛡️ TrustDoc AI")
        st.caption("Evidence-Grounded Documentation Assistant")
        st.divider()

        projects = list_projects(ctx.db)
        project_options = {p.id: f"{p.name} (id: {p.id})" for p in projects}
        project_options[0] = "➕ Create New Project"

        # Determine current active project in session state
        if "active_project_id" not in st.session_state:
            st.session_state["active_project_id"] = projects[0].id if projects else 0

        current_choice = st.session_state["active_project_id"]
        if current_choice not in project_options:
            current_choice = projects[0].id if projects else 0
            st.session_state["active_project_id"] = current_choice

        selected_id = st.selectbox(
            "Select Project",
            options=list(project_options.keys()),
            format_func=lambda pid: project_options[pid],
            index=list(project_options.keys()).index(current_choice),
            key="project_selector",
        )
        st.session_state["active_project_id"] = selected_id

        if selected_id == 0:
            st.markdown("#### ➕ Create New Project")
            with st.form("create_project_form", clear_on_submit=True):
                new_name = st.text_input("Project Name *", placeholder="e.g. my-python-lib")
                new_desc = st.text_area("Description", placeholder="Optional summary of this codebase")
                submit = st.form_submit_button("Create Project")
                if submit:
                    if not new_name.strip():
                        st.error("Project name cannot be empty.")
                    else:
                        try:
                            created = create_project(ctx.db, new_name.strip(), new_desc.strip())
                            st.session_state["active_project_id"] = created.id
                            st.success(f"Project '{created.name}' created!")
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed to create project: {exc}")
        else:
            active_proj = get_project(ctx.db, selected_id)
            if active_proj:
                st.markdown(f"**Description:** {active_proj.description or '*(None)*'}")
                stats = get_project_stats(ctx.db, selected_id)
                col1, col2, col3 = st.columns(3)
                col1.metric("Sources", stats["sources"])
                col2.metric("Entities", stats["entities"])
                col3.metric("Chunks", stats["chunks"])

                with st.expander("⚙️ Manage Project"):
                    confirm_delete = st.checkbox("Confirm deletion of this project", key="confirm_del_proj")
                    if st.button("Delete Project", type="secondary", disabled=not confirm_delete):
                        try:
                            delete_project(
                                ctx,
                                selected_id,
                                store=services.ingestion.store,
                                index_manager=services.retriever.index_manager,
                            )
                            st.session_state["active_project_id"] = 0
                            st.success("Project deleted.")
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed to delete project: {exc}")

        st.divider()
        with st.expander("ℹ️ System & Configuration"):
            st.markdown(f"**LLM Provider:** `{ctx.settings.llm.provider}`")
            st.markdown(f"**LLM Model:** `{ctx.settings.llm.model or 'default'}`")
            st.markdown(f"**Embedding Model:** `{ctx.settings.retrieval.embedding_model}`")
            st.markdown(f"**Database:** `{Path(ctx.settings.storage.db_path).name}`")
            st.caption("All verification and retrieval runs CPU-first and offline.")

    # ---------------------------------------------------------------- Main Area
    render_header()

    active_id = st.session_state.get("active_project_id", 0)
    if active_id == 0 or not projects:
        st.info("👈 Please create or select a project in the sidebar to start.")
        st.markdown(
            """
            ### Welcome to TrustDoc AI
            TrustDoc AI provides grounded, hallucination-resistant workflows for your Python codebases:
            - **📁 Ingestion**: Parse Python AST, Markdown documentation, and PDF files.
            - **💬 Ask**: Question answering strictly grounded in project evidence with validated citations.
            - **📝 Generate**: Automated generation of READMEs and API references passed through static AST verification.
            - **🔍 Verify**: Deterministic verification of claims against code signatures with transparent Trust Scoring.
            """
        )
        return

    active_project = get_project(ctx.db, active_id)
    if not active_project:
        st.warning("Selected project not found. Please select a valid project.")
        return

    # Tabs for the 4 core workflows
    tab_sources, tab_ask, tab_generate, tab_verify = st.tabs(
        ["📁 Sources & Ingestion", "💬 Ask", "📝 Generate", "🔍 Verify"]
    )

    # ---------------------------------------------------------------- TAB 1: Sources & Ingestion
    with tab_sources:
        st.subheader("📁 Project Sources")
        st.markdown(f"Manage code and documentation sources for **{active_project.name}**.")

        # File uploader
        st.markdown("#### Upload & Ingest Files")
        uploaded_files = st.file_uploader(
            "Upload Python (`.py`), Markdown (`.md`, `.markdown`), or text PDF (`.pdf`) files",
            type=["py", "md", "markdown", "pdf"],
            accept_multiple_files=True,
            key="source_file_uploader",
        )
        force_reingest = st.checkbox("Force re-ingestion if file content is unchanged", value=False)

        if st.button("Ingest Uploaded Files", type="primary", disabled=not bool(uploaded_files)):
            progress_bar = st.progress(0)
            status_holder = st.empty()
            success_count = 0

            for idx, uf in enumerate(uploaded_files or []):
                status_holder.text(f"Ingesting {uf.name}...")
                data = uf.getvalue()
                try:
                    res = services.ingestion.ingest_bytes(
                        active_project.id, uf.name, data, force=force_reingest
                    )
                    success_count += 1
                    st.success(
                        f"**{uf.name}**: {res.status.value.capitalize()} "
                        f"({res.entity_count} code entities, {res.chunk_count} chunks)"
                    )
                    for w in res.warnings:
                        st.warning(f"{uf.name}: {w}")
                except (FileTooLargeError, ParsingError, TrustDocError) as exc:
                    st.error(f"Failed to ingest **{uf.name}**: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected error ingesting **{uf.name}**: {exc}")
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_holder.empty()
            if success_count > 0:
                st.rerun()

        st.divider()
        st.markdown("#### Indexed Project Sources")
        sources = list_sources(ctx.db, active_project.id)

        if not sources:
            st.info("No sources indexed in this project yet. Upload a `.py` or `.md` file above.")
        else:
            for s in sources:
                col_name, col_type, col_size, col_action = st.columns([3, 1, 1, 1])
                col_name.markdown(f"📄 **`{s.name}`**")
                col_type.markdown(f"`{s.source_type.value}`")
                col_size.markdown(f"{s.size_bytes / 1024:.1f} KB")
                if col_action.button("🗑️ Delete", key=f"del_src_{s.id}"):
                    try:
                        delete_source(
                            ctx,
                            s.id,
                            store=services.ingestion.store,
                            index_manager=services.retriever.index_manager,
                        )
                        st.success(f"Deleted {s.name}")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to delete source: {exc}")

    # ---------------------------------------------------------------- TAB 2: Ask
    with tab_ask:
        st.subheader("💬 Ask Grounded Questions")
        st.markdown(
            "Ask a question about the project. Answers are strictly grounded in retrieved evidence; "
            "if evidence is insufficient, the system abstains honestly rather than inventing facts."
        )

        question = st.text_area(
            "Your Question:",
            placeholder="e.g. How do I configure the SQLite database connection timeout?",
            key="ask_question_input",
            height=90,
        )

        with st.expander("⚙️ Retrieval Settings"):
            top_k_ask = st.slider("Evidence items to retrieve (top_k)", min_value=1, max_value=20, value=5)

        if st.button("Ask Question", type="primary", key="btn_ask"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving evidence and formulating answer..."):
                    try:
                        result = services.ask.ask(active_project.id, question.strip(), top_k=top_k_ask)
                        st.session_state["last_ask_result"] = result
                    except (InvalidQuestionError, NotFoundError) as exc:
                        st.error(str(exc))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Operational error running Ask workflow: {exc}")

        last_ask = st.session_state.get("last_ask_result")
        if last_ask and last_ask.project_id == active_project.id:
            st.divider()
            render_status_badge(last_ask.status, error=last_ask.error)

            if last_ask.status == AskStatus.ANSWERED:
                st.markdown("### Answer")
                st.markdown(last_ask.answer)
                if last_ask.grounding:
                    st.caption(f"Grounding Level: `{last_ask.grounding.value}`")

                render_citations_list(last_ask.citations)
                render_evidence_drawer(last_ask.evidence)
            elif last_ask.status == AskStatus.INSUFFICIENT_EVIDENCE:
                st.info(f"**Abstention Message:** {last_ask.answer}")
                if last_ask.abstention_reason:
                    st.caption(f"Reason: `{last_ask.abstention_reason}`")
                render_evidence_drawer(last_ask.evidence)

            for w in last_ask.warnings:
                st.warning(f"⚠️ {w}")

    # ---------------------------------------------------------------- TAB 3: Generate
    with tab_generate:
        st.subheader("📝 Grounded Documentation Generation")
        st.markdown(
            "Generate README or Python API reference documentation grounded in retrieved code evidence. "
            "Generated content automatically passes through automated static AST verification."
        )

        gen_kind = st.radio(
            "Documentation Target",
            options=["Project README (README.md)", "Python API Documentation"],
            key="gen_kind_choice",
        )

        selected_entity_name: str | None = None
        if gen_kind == "Python API Documentation":
            entities = list_entities(ctx.db, active_project.id)
            # Filter to documentable entities (functions, classes, methods)
            target_entities = [
                e for e in entities if e.entity_type in (EntityType.FUNCTION, EntityType.CLASS, EntityType.METHOD)
            ]
            if not target_entities:
                st.warning("No Python functions, classes, or methods indexed in this project yet. Upload a `.py` file first.")
            else:
                entity_choices = {e.qualified_name: f"{e.qualified_name} ({e.entity_type.value}) - line {e.start_line}" for e in target_entities}
                selected_entity_name = st.selectbox(
                    "Select Python Entity to Document",
                    options=list(entity_choices.keys()),
                    format_func=lambda qn: entity_choices[qn],
                    key="gen_entity_select",
                )
                st.caption("💡 Selecting a class entity documents the class and its constructor (`__init__`) and methods.")

        with st.expander("⚙️ Generation & Verification Options"):
            top_k_gen = st.slider("Evidence items to retrieve (top_k)", min_value=1, max_value=20, value=5, key="top_k_gen")
            semantic_gen = st.checkbox(
                "Run Tier 2 semantic verification (requires local LLM)",
                value=False,
                key="semantic_gen_check",
            )
            skip_verify = st.checkbox(
                "Deliberately skip verification (run_verify=False)",
                value=False,
                key="skip_verify_check",
            )

        can_generate = gen_kind == "Project README (README.md)" or bool(selected_entity_name)
        if st.button("Generate Documentation", type="primary", disabled=not can_generate, key="btn_gen"):
            with st.spinner("Generating documentation and running verification pipeline..."):
                try:
                    if gen_kind == "Project README (README.md)":
                        res = services.generation.generate_readme(
                            active_project.id,
                            top_k=top_k_gen,
                            semantic=semantic_gen,
                            run_verify=not skip_verify,
                        )
                    else:
                        assert selected_entity_name is not None
                        res = services.generation.generate_api(
                            active_project.id,
                            selected_entity_name,
                            top_k=top_k_gen,
                            semantic=semantic_gen,
                            run_verify=not skip_verify,
                        )
                    st.session_state["last_gen_result"] = res
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Generation pipeline failed: {exc}")

        last_gen = st.session_state.get("last_gen_result")
        if last_gen and last_gen.project_id == active_project.id:
            st.divider()
            render_status_badge(
                last_gen.status,
                is_verified=last_gen.is_verified,
                error=last_gen.error,
            )

            if last_gen.markdown:
                target_fname = "README.md" if last_gen.kind == GenerationKind.README else f"api_{last_gen.target_entity or 'doc'}.md"
                st.download_button(
                    label="📥 Download Generated Markdown",
                    data=last_gen.markdown,
                    file_name=target_fname,
                    mime="text/markdown",
                    key="dl_markdown_btn",
                )

                subtab_preview, subtab_verification = st.tabs(["📄 Markdown Preview", "🛡️ Verification & Trust Score"])

                with subtab_preview:
                    st.markdown(last_gen.markdown)
                    with st.expander("Show Raw Markdown Source"):
                        st.code(last_gen.markdown, language="markdown")

                with subtab_verification:
                    if last_gen.verification_report:
                        rep = last_gen.verification_report
                        render_trust_score_card(rep.score)
                        render_claim_counts(rep.counts)
                        render_findings_list(rep.findings)
                    elif last_gen.status == GenerationStatus.VERIFICATION_SKIPPED:
                        st.info("Verification was deliberately skipped during generation.")
                    elif last_gen.status == GenerationStatus.VERIFICATION_ERROR:
                        st.error(f"Verification encountered an operational error: {last_gen.error}")

                render_citations_list(last_gen.citations)
                render_evidence_drawer(last_gen.evidence)

            for w in last_gen.warnings:
                st.warning(f"⚠️ {w}")

    # ---------------------------------------------------------------- TAB 4: Verify
    with tab_verify:
        st.subheader("🔍 Documentation Verification & Trust Score")
        st.markdown(
            "Verify documentation against code AST. Detects API signature mismatches, missing parameters, "
            "wrong types, and broken references."
        )

        verify_target_mode = st.radio(
            "Verification Target",
            options=["Verify Indexed Project Document", "Verify Custom Markdown Text"],
            key="verify_target_mode_radio",
        )

        custom_text: str | None = None
        selected_doc_name: str | None = None

        if verify_target_mode == "Verify Indexed Project Document":
            md_sources = list_sources(ctx.db, active_project.id, source_type=SourceType.MARKDOWN)
            if not md_sources:
                st.warning("No Markdown documentation indexed in this project. Ingest a `.md` file in Sources tab first.")
            else:
                doc_options = [s.name for s in md_sources]
                selected_doc_name = st.selectbox("Select Document", options=doc_options, key="verify_doc_select")
        else:
            custom_text = st.text_area(
                "Enter Markdown text to verify:",
                height=180,
                placeholder="# API Documentation\n\n`my_func(x, y)` takes two numbers and returns their sum.",
                key="verify_custom_text",
            )

        with st.expander("⚙️ Verification Options"):
            include_docstrings = st.checkbox("Include Python docstrings in verification", value=True, key="chk_docstrings")
            semantic_verify = st.checkbox("Run Tier 2 semantic checks (requires local LLM)", value=False, key="chk_semantic_v")

        can_verify = (verify_target_mode == "Verify Indexed Project Document" and bool(selected_doc_name)) or (
            verify_target_mode == "Verify Custom Markdown Text" and bool(custom_text and custom_text.strip())
        )

        if st.button("Run Verification", type="primary", disabled=not can_verify, key="btn_run_verify"):
            with st.spinner("Analyzing code AST, extracting claims, and verifying..."):
                try:
                    if verify_target_mode == "Verify Indexed Project Document":
                        assert selected_doc_name is not None
                        report = services.verification.verify_project(
                            active_project.id,
                            documents=[selected_doc_name],
                            include_docstrings=include_docstrings,
                            semantic=semantic_verify,
                        )
                    else:
                        assert custom_text is not None
                        report = services.verification.verify_text(
                            active_project.id,
                            custom_text,
                            name="custom_input.md",
                            semantic=semantic_verify,
                        )
                    st.session_state["last_verification_report"] = report
                except VerificationError as exc:
                    st.error(f"Verification error: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Operational error: {exc}")

        last_report = st.session_state.get("last_verification_report")
        if last_report and last_report.project_id == active_project.id:
            st.divider()
            render_trust_score_card(last_report.score)
            render_claim_counts(last_report.counts)
            render_findings_list(last_report.findings)

            if last_report.claims:
                with st.expander(f"📋 Extracted Claims ({len(last_report.claims)} claims)"):
                    for c in last_report.claims:
                        verdict_emoji = "✅" if c.status.value == "supported" else ("❌" if c.status.value == "contradicted" else "❓")
                        st.markdown(
                            f"- {verdict_emoji} **{c.category.value.upper()}** (`{c.status.value}` | Tier {c.tier.value}): *\"{c.text}\"* — {c.explanation}"
                        )

            for w in last_report.warnings:
                st.warning(f"⚠️ {w}")


if __name__ == "__main__":
    main()
