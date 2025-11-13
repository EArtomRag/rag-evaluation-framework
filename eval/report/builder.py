"""Builds the final HTML report from evaluation results."""

from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import json


def _create_detailed_csv_report(output_dir: Path, detailed_query_results: list):
    """Creates a CSV file with detailed metrics for each query."""
    if not detailed_query_results:
        return

    records = []
    for result in detailed_query_results:
        query_info = result.get("query", {})
        rag_output = result.get("rag_output", {})
        metrics = result.get("metrics", {})

        record = {
            "query_id": query_info.get("id"),
            "tipo_domanda": query_info.get("tipo_domanda"),
            "domanda": query_info.get("domanda"),
            "risposta_generata": rag_output.get("answer"),
        }

        for metric_name, value in metrics.items():
            if metric_name == 'query_id':
                continue
            
            if isinstance(value, dict):
                record[f"{metric_name}_score"] = value.get("score")
                record[f"{metric_name}_reason"] = value.get("reason")
            else:
                record[metric_name] = value
        
        records.append(record)

    df = pd.DataFrame(records)
    output_path = output_dir / "detailed_metrics.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig", decimal=".")
    print(f"Detailed metrics report saved to: {output_path}")


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

    # Create the detailed CSV report for single-turn queries
    _create_detailed_csv_report(output_dir, detailed_query_results)

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

