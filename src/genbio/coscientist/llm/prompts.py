"""Prompt templates for LLM interactions."""

SYSTEM_ADVISOR = (
    "You are a bioinformatics ML co-scientist at GenBio AI. "
    "Help users design experiments, select models, and interpret results. "
    "Be concise, precise, and scientifically rigorous."
)

SYSTEM_REPORT_WRITER = (
    "You are a scientific report writer. "
    "Polish ML experiment reports for clarity, accuracy, and scientific rigor. "
    "Maintain all data and markdown formatting."
)

INTERPRET_PROFILE = """Analyze this dataset profile and provide a concise summary:

{profile_json}

Include: data type, task type, key characteristics, potential challenges, and recommended approach."""

SUGGEST_APPROACH = """Given this dataset and available pipelines, recommend an experiment plan:

Dataset Profile:
{profile_json}

Available Pipelines: {pipeline_names}
Budget: {budget}

Provide a concise, actionable recommendation."""

INTERPRET_RESULTS = """Interpret these experiment results:

Dataset: {data_type}, {task_type}
Primary Metric: {primary_metric}

Results:
{results_json}

Provide key insights and recommendations for improvement."""
