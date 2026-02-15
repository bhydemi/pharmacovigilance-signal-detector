# Pharmacovigilance Signal Detector

A multi-agent AI system for real-world drug safety signal detection using FDA FAERS data.

## Overview

This project implements a pharmacovigilance workflow that:
- Analyzes FDA Adverse Event Reporting System (FAERS) data via live openFDA API
- Uses disproportionality statistics (ROR, PRR, confidence intervals) to detect signals
- Integrates PubMed literature search for clinical evidence
- Assesses biological plausibility of drug-adverse event associations
- Generates FDA-style regulatory safety reports
- Implements QC-driven feedback loops for iterative quality improvement
- Maintains audit trails for regulatory compliance

## Architecture

### Agents

1. **Statistical Agent** - Senior FDA Biostatistician
   - Calculates ROR, PRR, chi-square statistics from FAERS data
   - Validates statistical results
   - Determines signal detection status using EMA criteria

2. **Clinical Agent** - Clinical Pharmacologist
   - Searches PubMed for literature evidence
   - Assesses biological plausibility
   - Classifies associations as PLAUSIBLE/IMPLAUSIBLE/UNCERTAIN

3. **Regulatory Agent** - FDA Regulatory Affairs Specialist
   - Generates FDA-compliant safety reports
   - Determines risk levels and recommended actions

4. **Quality Control Agent** - QA Director
   - Scores analysis across multiple dimensions
   - Identifies issues and routes feedback to responsible agents
   - Drives iterative improvement loop

## Files

- `starter_code_pharmacovigilance_signal_detector.py` - Main orchestration code
- `pharma_tools.py` - Tool implementations (FAERS API, PubMed, plausibility, etc.)
- `audit_logger.py` - Audit trail utilities for regulatory compliance
- `.env.example` - Environment variable template

## Setup

1. Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
BASE_URL=https://openai.vocareum.com/v1
```

2. Install dependencies:
```bash
pip install langchain langchain-openai python-dotenv requests
```

3. Run the analysis:
```bash
python starter_code_pharmacovigilance_signal_detector.py
```

## Example Output

```
PHARMACOVIGILANCE SIGNAL ANALYSIS
Drug: MONTELUKAST | Adverse Event: AGGRESSION

📚 Fetching literature evidence from PubMed...
🔬 Running STATISTICAL agent...
🏥 Running CLINICAL agent...
📋 Running REGULATORY agent...
🔍 Running QUALITY CONTROL review...

Final Quality Score: 97.0
Iterations Used: 2
Converged: True
```

## Rubric Compliance

- Data handling with FAERS API integration
- ROR/PRR calculation and validation
- Role-based system prompts for Statistical and Clinical agents
- Correct tool mapping to agents
- Agent runners producing usable outputs
- QC integration with feedback routing
- Feedback loop with targeted re-execution
- Audit logging for traceability

## License

Educational project for Udacity AI course.
