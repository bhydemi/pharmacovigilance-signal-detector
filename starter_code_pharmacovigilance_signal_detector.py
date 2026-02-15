"""
Capstone: Pharmacovigilance Signal Detector with Feedback Loops

DO NOT MODIFY:
- pharma_tools.py   → FAERS + PubMed + plausibility + regulatory + QC tools
- audit_logger.py   → audit trail utilities

IN THIS FILE YOU WILL:
1. Write system prompts for your agents (statistical & clinical).
2. Assign pharma_tools functions to each agent.
3. Orchestrate the agents (statistical, clinical, regulatory, QC).
4. Implement a TRUE feedback loop driven by the QC tool.
5. Add minimal audit logging inside the loop (regulatory-style requirement).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import pharma_tools
from pharma_tools import fetch_literature_evidence
from audit_logger import AuditLogger


# =============================================================================
# 1. ENVIRONMENT / LLM SETUP (GIVEN)
# =============================================================================


def setup_environment() -> ChatOpenAI:
    """
    Initialize the OpenAI client.

    You do NOT need to change this.
    Just make sure you have a .env with OPENAI_API_KEY (and optional BASE_URL).
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")  # optional

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Add it to your .env file before running."
        )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_completion_tokens=2000,
        base_url=base_url,
    )
    return llm


def configure_faers_mode(use_live_api: bool = True) -> None:
    """
    Switch between live FAERS API and any local fallback data.

    For this course we recommend using the LIVE API (default=True).
    """
    pharma_tools.set_use_live_api(use_live_api)


# =============================================================================
# 2. SYSTEM PROMPTS & TOOL ASSIGNMENT  (TODO 0)
# =============================================================================

# ------------------------
# TODO 0A: SYSTEM PROMPTS
# ------------------------
#
# Write effective system prompts for the two LLM-based agents.
# Use your own words, but make sure you cover at least the points in the comments.


STATISTICAL_SYSTEM_PROMPT = """You are a Senior FDA Biostatistician specializing in pharmacovigilance signal detection.

ROLE: Senior Pharmacovigilance Biostatistician with expertise in disproportionality analysis.

INPUTS YOU WILL RECEIVE:
- FAERS statistical data: ROR, PRR, 95% confidence intervals, chi-square, p-values
- Contingency table counts (drug-event, drug-no-event, etc.)
- Validation report assessing data quality
- Literature evidence summary (article count, evidence strength)

YOUR RESPONSIBILITIES:
1. SIGNAL ASSESSMENT: Evaluate whether a disproportionality signal exists using EMA criteria:
   - Minimum 3 cases (n ≥ 3)
   - ROR ≥ 2.0
   - Lower bound of 95% CI > 1.0 (statistical significance)

2. QUANTITATIVE REPORTING: You MUST include these exact numbers from the tool outputs:
   - ROR value and 95% CI (lower-upper bounds)
   - PRR value
   - Case count (drug-event occurrences)
   - P-value and chi-square statistic

3. DATA QUALITY ASSESSMENT: Identify and discuss limitations:
   - Low case counts (n < 10 warrants caution)
   - Wide confidence intervals (uncertainty)
   - Data quality flags from validation

4. INTERPRETATION: Provide clear conclusion on signal status (DETECTED/NOT DETECTED)

SCOPE LIMITATIONS:
- DO NOT make clinical recommendations or treatment advice
- DO NOT assess biological plausibility (clinical agent's role)
- DO NOT recommend regulatory actions (regulatory agent's role)
- ONLY provide statistical interpretation

QC FEEDBACK HANDLING:
If QC feedback is provided, you MUST:
1. Acknowledge the specific issue raised
2. Explain what you are changing in response
3. Provide the revised interpretation with corrections

Always cite specific numbers from the tool outputs. Never fabricate statistics."""


CLINICAL_SYSTEM_PROMPT = """You are a Clinical Pharmacologist and Director of Pharmacovigilance with expertise in drug safety assessment.

ROLE: Clinical Drug Safety Physician specializing in adverse event causality assessment.

INPUTS YOU WILL RECEIVE:
- Statistical summary from the biostatistician (ROR, signal status)
- PubMed literature search results (article count, titles, years)
- Biological plausibility assessment (mechanism, pathway, confidence)

YOUR RESPONSIBILITIES:
1. CLINICAL EVIDENCE SYNTHESIS:
   - Summarize relevant literature findings
   - Match evidence strength to PubMed article count:
     * 0 articles = "none"
     * 1-4 articles = "limited"
     * 5-19 articles = "moderate"
     * 20+ articles = "strong"
   - DO NOT overclaim evidence strength beyond what PubMed data supports

2. MECHANISTIC PLAUSIBILITY ASSESSMENT:
   - Evaluate proposed biological mechanisms
   - Consider pharmacological pathways (receptor binding, metabolism, etc.)
   - Assess whether mechanism is biologically credible
   - If mechanism is UNCERTAIN, investigate potential hypotheses:
     * Propose plausible mechanistic pathways with literature support
     * Identify potential confounders that could explain spurious signals
     * Document why mechanism remains uncertain and what research is needed

3. CAUSALITY CLASSIFICATION: Provide final assessment as one of:
   - PLAUSIBLE: Known mechanism, consistent evidence, biological rationale
   - IMPLAUSIBLE: Contradicts known pharmacology, no credible mechanism
   - UNCERTAIN: Insufficient evidence, conflicting data, or unknown mechanism

4. ASSOCIATION VS CAUSATION:
   - Consider Bradford-Hill criteria when applicable
   - Identify potential confounders
   - Distinguish correlation from true causal relationship

SCOPE LIMITATIONS:
- DO NOT modify or recalculate statistical figures
- DO NOT make regulatory recommendations
- Focus on clinical/scientific interpretation only

QC FEEDBACK HANDLING:
If QC feedback is provided, you MUST:
1. Acknowledge the specific issue raised
2. Explicitly state "In response to QC feedback, I am revising..."
3. Address the specific evidence gaps or misalignments identified
4. Provide corrected assessment with clear reasoning

Always align your evidence claims with actual PubMed article counts."""


# -----------------------------
# TODO 0B: TOOL ASSIGNMENT
# -----------------------------
#
# Map each agent to the tools it should use from pharma_tools.py.
#
# The tools you have available (from pharma_tools):
#   - calculate_drug_event_statistics(drug_name, adverse_event)
#   - validate_statistical_results(statistical_results_json)
#   - search_clinical_literature(drug_name, adverse_event)
#   - assess_biological_plausibility(drug_name, adverse_event, literature_context=None)
#   - generate_regulatory_report(...)
#   - quality_review_analysis(...)
#
# Fill in the correct pharma_tools.<function_name> below.
# Code will crash until these are NOT None.


STATISTICAL_TOOLS = {
    "calculate_stats": pharma_tools.calculate_drug_event_statistics,
    "validate_stats": pharma_tools.validate_statistical_results,
}

CLINICAL_TOOLS = {
    "search_literature": pharma_tools.search_clinical_literature,
    "assess_plausibility": pharma_tools.assess_biological_plausibility,
}

REGULATORY_TOOL = pharma_tools.generate_regulatory_report
QC_TOOL = pharma_tools.quality_review_analysis


# =============================================================================
# 3. AGENT RUNNERS  (TODO 1)
# =============================================================================
#
# You are now building thin “agent wrappers” around:
#   - FAERS tools (stats)
#   - PubMed + plausibility tools (clinical)
#   - Regulatory report tool
#
# Design choice (recommended, but up to you):
#   • Call pharma_tools.* directly to get JSON.
#   • Then ask the LLM (with your system prompt) to summarize / interpret.
#
# Returned dicts SHOULD roughly follow this shape (so QC + feedback loop can use them):
#   {
#      "agent_response": "<narrative text>",
#      "tool_usage": ["tool_name_1", ...],
#      "raw_data": <JSON string or dict used by downstream tools>,
#      "status": "completed" or "revised",
#      "revision_number": int,
#      "feedback_addressed": Optional[str]
#   }
#
# Feel free to extend with extra fields if useful.


def run_statistical_agent(
    llm: ChatOpenAI,
    drug_name: str,
    adverse_event: str,
    literature_evidence: Dict[str, Any],
    feedback_context: Optional[str] = None,
    revision_number: int = 0,
) -> Dict[str, Any]:
    """
    Statistical agent: analyzes FAERS data for disproportionality signals.

    Uses calculate_drug_event_statistics and validate_statistical_results tools.
    """
    tool_usage = []

    # Step 1: Calculate FAERS statistics
    stats_json = STATISTICAL_TOOLS["calculate_stats"].invoke({
        "drug_name": drug_name,
        "adverse_event": adverse_event
    })
    tool_usage.append("calculate_drug_event_statistics")

    # Step 2: Validate the statistical results
    validation_json = STATISTICAL_TOOLS["validate_stats"].invoke({
        "statistical_results_json": stats_json
    })
    tool_usage.append("validate_statistical_results")

    # Step 3: Build prompt with all context
    lit_summary = (
        f"Literature context: {literature_evidence.get('n_articles', 0)} PubMed articles found, "
        f"evidence strength: {literature_evidence.get('evidence_strength', 'unknown')}, "
        f"year range: {literature_evidence.get('year_range', 'N/A')}"
    )

    prompt = f"""Analyze the following drug-adverse event pair for statistical signal detection:

DRUG: {drug_name}
ADVERSE EVENT: {adverse_event}

=== FAERS STATISTICAL DATA ===
{stats_json}

=== VALIDATION REPORT ===
{validation_json}

=== LITERATURE CONTEXT ===
{lit_summary}
"""

    # Add QC feedback if this is a revision
    if feedback_context:
        prompt += f"""
=== QC FEEDBACK (MUST ADDRESS) ===
{feedback_context}

Please revise your analysis to address the QC feedback above. Explicitly state what you are changing.
"""

    prompt += """
Provide your statistical interpretation including:
1. Signal status (DETECTED or NOT DETECTED) with EMA criteria assessment
2. Key statistics (ROR, CI, PRR, p-value, case count)
3. Data quality assessment
4. Limitations and caveats
"""

    # Step 4: Call LLM for interpretation
    messages = [
        SystemMessage(content=STATISTICAL_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)

    # Step 5: Return structured result
    return {
        "agent_response": response.content,
        "tool_usage": tool_usage,
        "raw_data": stats_json,
        "validation_data": validation_json,
        "status": "revised" if revision_number > 0 else "completed",
        "revision_number": revision_number,
        "feedback_addressed": feedback_context if feedback_context else None
    }


def run_clinical_agent(
    llm: ChatOpenAI,
    drug_name: str,
    adverse_event: str,
    literature_evidence: Dict[str, Any],
    statistical_context: str,
    feedback_context: Optional[str] = None,
    revision_number: int = 0,
) -> Dict[str, Any]:
    """
    Clinical agent: evaluates biological plausibility and literature evidence.

    Uses search_clinical_literature and assess_biological_plausibility tools.
    """
    tool_usage = []

    # Step 1: Search clinical literature
    literature_json = CLINICAL_TOOLS["search_literature"].invoke({
        "drug_name": drug_name,
        "adverse_event": adverse_event
    })
    tool_usage.append("search_clinical_literature")

    # Step 2: Assess biological plausibility (with literature context)
    plausibility_json = CLINICAL_TOOLS["assess_plausibility"].invoke({
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "literature_context": json.dumps(literature_evidence)
    })
    tool_usage.append("assess_biological_plausibility")

    # Step 3: Parse and combine for downstream use
    try:
        literature_data = json.loads(literature_json)
    except json.JSONDecodeError:
        literature_data = {"error": "Failed to parse literature JSON"}

    try:
        plausibility_data = json.loads(plausibility_json)
    except json.JSONDecodeError:
        plausibility_data = {"error": "Failed to parse plausibility JSON"}

    # Combined data for QC and regulatory
    combined_data = {
        "literature": literature_data,
        "plausibility": plausibility_data,
        "mechanism_assessment": plausibility_data.get("mechanism_assessment", plausibility_data),
        "evidence_summary": literature_data.get("evidence_summary", {}),
        "_revision_number": revision_number  # Track for QC penalty logic
    }

    # Step 4: Build prompt
    # Truncate statistical context if very long
    stat_context_truncated = statistical_context[:2000] if len(statistical_context) > 2000 else statistical_context

    prompt = f"""Evaluate the clinical evidence for the following drug-adverse event association:

DRUG: {drug_name}
ADVERSE EVENT: {adverse_event}

=== STATISTICAL SUMMARY ===
{stat_context_truncated}

=== PUBMED LITERATURE SEARCH RESULTS ===
{literature_json}

=== BIOLOGICAL PLAUSIBILITY ASSESSMENT ===
{plausibility_json}
"""

    # Add QC feedback if this is a revision
    if feedback_context:
        prompt += f"""
=== QC FEEDBACK (MUST ADDRESS) ===
{feedback_context}

In response to QC feedback, you MUST:
1. Explicitly acknowledge the issue raised
2. Investigate the mechanism more thoroughly if requested
3. Revise your assessment with clear reasoning for any changes
"""

    prompt += """
Provide your clinical assessment including:
1. Evidence strength assessment (aligned with PubMed article count)
2. Biological plausibility classification (PLAUSIBLE / IMPLAUSIBLE / UNCERTAIN)
3. Mechanistic pathway discussion
4. Association vs causation considerations
5. Key limitations and uncertainties
"""

    # Step 5: Call LLM for interpretation
    messages = [
        SystemMessage(content=CLINICAL_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)

    # Add narrative to combined data for QC mechanism detection
    combined_data["narrative_response"] = response.content

    # Step 6: Return structured result
    return {
        "agent_response": response.content,
        "tool_usage": tool_usage,
        "raw_data": combined_data,
        "literature_json": literature_json,
        "plausibility_json": plausibility_json,
        "status": "revised" if revision_number > 0 else "completed",
        "revision_number": revision_number,
        "feedback_addressed": feedback_context if feedback_context else None
    }


def run_regulatory_agent(
    drug_name: str,
    adverse_event: str,
    statistical_data_json: str,
    clinical_combined_json: str,
    literature_evidence: Dict[str, Any],
    feedback_loop_summary: Optional[Dict[str, Any]] = None,
    revision_number: int = 0,
) -> Dict[str, Any]:
    """
    Regulatory agent: generates FDA-style safety report.

    This is purely tool-based (no LLM needed).
    Uses generate_regulatory_report tool.
    """
    tool_usage = []

    # Prepare arguments for the regulatory report tool
    tool_args = {
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "statistical_data": statistical_data_json,
        "clinical_assessment": clinical_combined_json,
        "literature_evidence": json.dumps(literature_evidence),
        "feedback_loop_summary": json.dumps(feedback_loop_summary) if feedback_loop_summary else None
    }

    # Generate the regulatory report
    report_json = REGULATORY_TOOL.invoke(tool_args)
    tool_usage.append("generate_regulatory_report")

    # Return structured result
    return {
        "agent_response": report_json,  # The report is the response
        "tool_usage": tool_usage,
        "raw_data": report_json,
        "status": "revised" if revision_number > 0 else "completed",
        "revision_number": revision_number,
        "feedback_addressed": None  # Regulatory doesn't directly address feedback
    }


# =============================================================================
# 4. QUALITY CONTROL & FEEDBACK HELPERS  (TODO 2)
# =============================================================================

def run_quality_control(
    statistical_data_json: str,
    clinical_combined_json: str,
    regulatory_report_json: str,
    literature_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    QC runner: performs quality review of the multi-agent analysis.

    Uses quality_review_analysis tool.
    """
    # Prepare arguments for QC tool
    tool_args = {
        "statistical_findings": statistical_data_json,
        "clinical_findings": clinical_combined_json,
        "regulatory_report": regulatory_report_json,
        "literature_evidence": json.dumps(literature_evidence)
    }

    # Call QC tool
    qc_json = QC_TOOL.invoke(tool_args)

    # Parse the JSON response
    try:
        qc_result = json.loads(qc_json)
    except json.JSONDecodeError:
        qc_result = {
            "error": "Failed to parse QC response",
            "quality_scores": {"overall_score": 0},
            "target_agents": [],
            "issues": [],
            "approval_status": "revision_required"
        }

    # Keep raw output for debugging
    qc_result["raw_tool_output"] = qc_json

    return qc_result


def get_feedback_for_agent(
    agent_name: str, issues: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Extract QC feedback relevant to a specific agent.

    Filters issues by responsible_agent and builds actionable feedback string.
    """
    # Filter issues for this specific agent
    agent_issues = [
        issue for issue in issues
        if issue.get("responsible_agent") == agent_name
    ]

    if not agent_issues:
        return None

    # Build feedback string
    feedback_parts = [f"QC FEEDBACK FOR {agent_name.upper()} AGENT:"]

    for i, issue in enumerate(agent_issues, 1):
        issue_text = f"""
Issue {i}:
- Type: {issue.get('type', 'unknown')}
- Severity: {issue.get('severity', 'unknown')}
- Description: {issue.get('description', 'No description')}
- Required Action: {issue.get('specific_action', 'Review and revise')}
- Evidence Needed: {', '.join(issue.get('evidence_needed', [])) if issue.get('evidence_needed') else 'N/A'}
"""
        feedback_parts.append(issue_text)

    feedback_parts.append("\nYou MUST address each issue above in your revised response.")

    return "\n".join(feedback_parts)


# =============================================================================
# 5. FEEDBACK-DRIVEN RE-RUNNING  (TODO 3)
# =============================================================================

def rerun_agent_with_feedback(
    llm: ChatOpenAI,
    agent_name: str,
    feedback: str,
    drug_name: str,
    adverse_event: str,
    literature_evidence: Dict[str, Any],
    current_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Re-run a specific agent with QC feedback incorporated.

    Returns the revised agent result.
    """
    # Map agent names to result keys
    agent_result_keys = {
        "statistical": "statistical_analysis",
        "clinical": "clinical_assessment",
        "regulatory": "regulatory_report"
    }

    # Get previous result and increment revision number
    result_key = agent_result_keys.get(agent_name)
    if not result_key or result_key not in current_results:
        raise ValueError(f"Unknown agent or missing results: {agent_name}")

    previous_result = current_results[result_key]
    old_revision = previous_result.get("revision_number", 0)
    new_revision = old_revision + 1

    # Re-run the appropriate agent
    if agent_name == "statistical":
        revised_result = run_statistical_agent(
            llm=llm,
            drug_name=drug_name,
            adverse_event=adverse_event,
            literature_evidence=literature_evidence,
            feedback_context=feedback,
            revision_number=new_revision
        )

    elif agent_name == "clinical":
        # Get the latest statistical agent response for context
        stat_result = current_results.get("statistical_analysis", {})
        statistical_context = stat_result.get("agent_response", "")

        revised_result = run_clinical_agent(
            llm=llm,
            drug_name=drug_name,
            adverse_event=adverse_event,
            literature_evidence=literature_evidence,
            statistical_context=statistical_context,
            feedback_context=feedback,
            revision_number=new_revision
        )

    elif agent_name == "regulatory":
        # Get latest statistical and clinical data
        stat_result = current_results.get("statistical_analysis", {})
        clinical_result = current_results.get("clinical_assessment", {})

        statistical_data_json = stat_result.get("raw_data", "{}")
        clinical_combined_json = json.dumps(clinical_result.get("raw_data", {}))

        revised_result = run_regulatory_agent(
            drug_name=drug_name,
            adverse_event=adverse_event,
            statistical_data_json=statistical_data_json,
            clinical_combined_json=clinical_combined_json,
            literature_evidence=literature_evidence,
            revision_number=new_revision
        )

    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    return revised_result


# =============================================================================
# 6. MAIN ORCHESTRATION: FEEDBACK LOOP  (TODO 4)
# =============================================================================

def analyze_with_feedback_loop(
    drug_name: str,
    adverse_event: str,
    max_iterations: int = 3,
    quality_threshold: float = 90.0,
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, Any]:
    """
    Full multi-agent pharmacovigilance analysis with QC-driven feedback loop.

    Runs statistical, clinical, and regulatory agents, then iteratively
    improves based on QC feedback until quality threshold is met.
    """
    # =========================================================================
    # Step 1: Setup
    # =========================================================================
    if llm is None:
        llm = setup_environment()

    audit = AuditLogger()
    run_id = audit.start_run(drug_name, adverse_event, {
        "max_iterations": max_iterations,
        "quality_threshold": quality_threshold
    })

    print(f"\n{'='*60}")
    print(f"PHARMACOVIGILANCE SIGNAL ANALYSIS")
    print(f"Drug: {drug_name} | Adverse Event: {adverse_event}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Step 2: Fetch literature evidence ONCE (shared across all agents)
    # =========================================================================
    print("📚 Fetching literature evidence from PubMed...")
    literature_evidence = fetch_literature_evidence(drug_name, adverse_event)
    audit.log_literature_fetch(literature_evidence)
    print(f"   Found {literature_evidence.get('n_articles', 0)} articles, "
          f"strength: {literature_evidence.get('evidence_strength', 'unknown')}\n")

    # Initialize results container
    current_results: Dict[str, Any] = {}

    # =========================================================================
    # Step 3: Initial pass - run all agents
    # =========================================================================
    print("🔬 Running STATISTICAL agent...")
    stat_result = run_statistical_agent(
        llm=llm,
        drug_name=drug_name,
        adverse_event=adverse_event,
        literature_evidence=literature_evidence
    )
    current_results["statistical_analysis"] = stat_result
    audit.log_agent_execution(
        agent_name="statistical",
        tool_calls=stat_result.get("tool_usage", []),
        success=True,
        revision_number=0
    )
    print(f"   ✓ Statistical analysis complete\n")

    print("🏥 Running CLINICAL agent...")
    clinical_result = run_clinical_agent(
        llm=llm,
        drug_name=drug_name,
        adverse_event=adverse_event,
        literature_evidence=literature_evidence,
        statistical_context=stat_result.get("agent_response", "")
    )
    current_results["clinical_assessment"] = clinical_result
    audit.log_agent_execution(
        agent_name="clinical",
        tool_calls=clinical_result.get("tool_usage", []),
        success=True,
        revision_number=0
    )
    print(f"   ✓ Clinical assessment complete\n")

    print("📋 Running REGULATORY agent...")
    reg_result = run_regulatory_agent(
        drug_name=drug_name,
        adverse_event=adverse_event,
        statistical_data_json=stat_result.get("raw_data", "{}"),
        clinical_combined_json=json.dumps(clinical_result.get("raw_data", {})),
        literature_evidence=literature_evidence
    )
    current_results["regulatory_report"] = reg_result
    audit.log_agent_execution(
        agent_name="regulatory",
        tool_calls=reg_result.get("tool_usage", []),
        success=True,
        revision_number=0
    )
    print(f"   ✓ Regulatory report complete\n")

    # =========================================================================
    # Step 4: First QC pass
    # =========================================================================
    print("🔍 Running QUALITY CONTROL review...")
    qc_result = run_quality_control(
        statistical_data_json=stat_result.get("raw_data", "{}"),
        clinical_combined_json=json.dumps(clinical_result.get("raw_data", {})),
        regulatory_report_json=reg_result.get("raw_data", "{}"),
        literature_evidence=literature_evidence
    )
    current_results["quality_review"] = qc_result

    quality_score = qc_result.get("quality_scores", {}).get("overall_score", 0)
    target_agents = qc_result.get("target_agents", [])
    issues = qc_result.get("issues", [])

    audit.log_quality_review(
        scores=qc_result.get("quality_scores", {}),
        issues=issues,
        approval_status=qc_result.get("approval_status", "unknown"),
        target_agents=target_agents
    )
    print(f"   Quality Score: {quality_score:.1f}")
    print(f"   Status: {qc_result.get('approval_status', 'unknown')}")
    print(f"   Target agents for revision: {target_agents}\n")

    # =========================================================================
    # Step 5: Feedback loop
    # =========================================================================
    iteration = 1
    feedback_loop_metadata = {
        "iterations_used": 1,
        "final_quality_score": quality_score,
        "converged": False,
        "analysis_history": []
    }

    # Record initial state in history
    feedback_loop_metadata["analysis_history"].append({
        "iteration": 1,
        "quality_score": quality_score,
        "issues_count": len(issues),
        "target_agents": target_agents,
        "actions_taken": ["initial_pass"]
    })

    # Feedback loop
    while iteration < max_iterations and quality_score < quality_threshold:
        # Check if we've converged (no more agents to revise)
        if not target_agents:
            print(f"✅ No agents targeted for revision - converged!\n")
            feedback_loop_metadata["converged"] = True
            break

        iteration += 1
        print(f"\n{'─'*40}")
        print(f"ITERATION {iteration}")
        print(f"{'─'*40}")

        actions_taken = []
        upstream_changed = False

        # Re-run targeted agents with feedback
        for agent_name in target_agents:
            feedback = get_feedback_for_agent(agent_name, issues)
            if not feedback:
                continue

            print(f"🔄 Re-running {agent_name.upper()} agent with QC feedback...")

            revised_result = rerun_agent_with_feedback(
                llm=llm,
                agent_name=agent_name,
                feedback=feedback,
                drug_name=drug_name,
                adverse_event=adverse_event,
                literature_evidence=literature_evidence,
                current_results=current_results
            )

            # Update results
            result_key = {
                "statistical": "statistical_analysis",
                "clinical": "clinical_assessment",
                "regulatory": "regulatory_report"
            }[agent_name]
            current_results[result_key] = revised_result

            # Log the revision
            audit.log_agent_execution(
                agent_name=agent_name,
                tool_calls=revised_result.get("tool_usage", []),
                success=True,
                revision_number=revised_result.get("revision_number", 1),
                feedback_context=feedback[:200]
            )

            actions_taken.append(f"revised_{agent_name}")
            print(f"   ✓ {agent_name.capitalize()} revised (revision #{revised_result.get('revision_number', 1)})")

            # Track if upstream agents changed
            if agent_name in ["statistical", "clinical"]:
                upstream_changed = True

        # If statistical or clinical changed, regenerate regulatory report
        if upstream_changed and "regulatory" not in target_agents:
            print(f"📋 Regenerating REGULATORY report with updated inputs...")
            stat_result = current_results["statistical_analysis"]
            clinical_result = current_results["clinical_assessment"]

            reg_result = run_regulatory_agent(
                drug_name=drug_name,
                adverse_event=adverse_event,
                statistical_data_json=stat_result.get("raw_data", "{}"),
                clinical_combined_json=json.dumps(clinical_result.get("raw_data", {})),
                literature_evidence=literature_evidence,
                feedback_loop_summary=feedback_loop_metadata
            )
            current_results["regulatory_report"] = reg_result
            audit.log_agent_execution(
                agent_name="regulatory",
                tool_calls=reg_result.get("tool_usage", []),
                success=True,
                revision_number=reg_result.get("revision_number", 0)
            )
            actions_taken.append("regenerated_regulatory")
            print(f"   ✓ Regulatory report regenerated")

        # Re-run QC with updated results
        print(f"\n🔍 Re-running QUALITY CONTROL...")
        stat_result = current_results["statistical_analysis"]
        clinical_result = current_results["clinical_assessment"]
        reg_result = current_results["regulatory_report"]

        qc_result = run_quality_control(
            statistical_data_json=stat_result.get("raw_data", "{}"),
            clinical_combined_json=json.dumps(clinical_result.get("raw_data", {})),
            regulatory_report_json=reg_result.get("raw_data", "{}"),
            literature_evidence=literature_evidence
        )
        current_results["quality_review"] = qc_result

        quality_score = qc_result.get("quality_scores", {}).get("overall_score", 0)
        target_agents = qc_result.get("target_agents", [])
        issues = qc_result.get("issues", [])

        audit.log_quality_review(
            scores=qc_result.get("quality_scores", {}),
            issues=issues,
            approval_status=qc_result.get("approval_status", "unknown"),
            target_agents=target_agents
        )

        # Log iteration in audit
        audit.log_iteration(
            iteration=iteration,
            quality_score=quality_score,
            issues=issues,
            actions=[{"action": a} for a in actions_taken]
        )

        print(f"   Quality Score: {quality_score:.1f}")
        print(f"   Status: {qc_result.get('approval_status', 'unknown')}")
        print(f"   Remaining targets: {target_agents}")

        # Record in history
        feedback_loop_metadata["analysis_history"].append({
            "iteration": iteration,
            "quality_score": quality_score,
            "issues_count": len(issues),
            "target_agents": target_agents,
            "actions_taken": actions_taken
        })

    # =========================================================================
    # Step 6: Finalize results
    # =========================================================================
    feedback_loop_metadata["iterations_used"] = iteration
    feedback_loop_metadata["final_quality_score"] = quality_score
    feedback_loop_metadata["converged"] = (
        quality_score >= quality_threshold or not target_agents
    )

    # End audit run
    final_results = {
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "timestamp": datetime.now().isoformat(),
        "literature_evidence": literature_evidence,
        "statistical_analysis": current_results.get("statistical_analysis", {}),
        "clinical_assessment": current_results.get("clinical_assessment", {}),
        "regulatory_report": current_results.get("regulatory_report", {}),
        "quality_review": current_results.get("quality_review", {}),
        "feedback_loop_metadata": feedback_loop_metadata
    }

    audit_log_path = audit.end_run(final_results, status="completed")
    final_results["audit_log_path"] = audit_log_path

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Final Quality Score: {quality_score:.1f}")
    print(f"Iterations Used: {iteration}")
    print(f"Converged: {feedback_loop_metadata['converged']}")
    print(f"Audit Log: {audit_log_path}")

    return final_results


# =============================================================================
# 7. SIMPLE EXAMPLE ENTRY POINT  (GIVEN)
# =============================================================================

def run_example() -> None:
    """
    Run a single example analysis.

    You can modify the drug / adverse_event to experiment.

    Try:
      - Clear-cut:    METFORMIN + LACTIC ACIDOSIS
      - Ambiguous:    MONTELUKAST + AGGRESSION   (good for seeing iterations)
    """
    configure_faers_mode(use_live_api=True)
    llm = setup_environment()

    results = analyze_with_feedback_loop(
        drug_name="MONTELUKAST",
        adverse_event="AGGRESSION",
        max_iterations=3,
        quality_threshold=95.0,
        llm=llm,
    )

    meta = results.get("feedback_loop_metadata", {})
    print("\n=== SUMMARY ===")
    print(f"Drug: {results['drug_name']}")
    print(f"Adverse event: {results['adverse_event']}")
    print(f"Final quality score: {meta.get('final_quality_score')}")
    print(f"Iterations used: {meta.get('iterations_used')}")
    print(f"Converged: {meta.get('converged')}")
    print(f"Audit log: {results.get('audit_log_path')}")


if __name__ == "__main__":
    # You can comment this out while you are working on TODOs.
    run_example()

