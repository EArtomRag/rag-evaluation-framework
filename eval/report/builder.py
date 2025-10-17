"""Builds the final HTML report from evaluation results."""

from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import json


def build_html_report(
    run_id: str,
    suite_name: str,
    timestamp: str,
    aggregated_results: pd.DataFrame,
    output_dir: Path,
):
    """
    Renders the Jinja2 template with evaluation results and saves it as an HTML file.

    Args:
        run_id: The ID of the evaluation run.
        suite_name: The name of the evaluation suite.
        timestamp: The timestamp when the run was started.
        aggregated_results: A DataFrame with the aggregated metric scores.
        output_dir: The directory where the report.html file will be saved.
    """
    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html.j2")

    # Convert DataFrame to HTML table for rendering
    results_html = aggregated_results.to_html(
        classes="table",
        float_format="{:.4f}".format,
    ) if not aggregated_results.empty else None

    # Load detailed results from the 'raw' directory
    detailed_query_results = []
    detailed_conversation_results = []
    raw_dir = output_dir / "raw"
    if raw_dir.exists():
        for f in sorted(raw_dir.glob("*.json")):
            with open(f, "r", encoding="utf-8") as raw_file:
                data = json.load(raw_file)
                if data.get("type") == "conversation":
                    detailed_conversation_results.append(data)
                else:
                    detailed_query_results.append(data)

    # Render the template with the provided data
    html_content = template.render(
        run_id=run_id,
        suite_name=suite_name,
        timestamp=timestamp,
        aggregated_results=results_html,
        detailed_query_results=detailed_query_results,
        detailed_conversation_results=detailed_conversation_results,
    )

    # Save the rendered HTML to a file
    report_path = output_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {report_path}")

