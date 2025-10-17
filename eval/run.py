"""Main CLI entry point for the RAG Evaluation Runner."""

import json
from pathlib import Path
import time
from typing import Dict, List, Union

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
import pandas as pd

# Carica le variabili d'ambiente da un file .env se esiste
# Questo permette di non dover impostare a mano la OPENAI_API_KEY
load_dotenv()

from eval.schemas.data_models import (
    Conversation,
    EvaluationSuite,
    Query,
    RAGOutput,
    Turn,
)
from eval.data_loader import stream_dataset
from eval.adapters.base import BaseRAGAdapter
from eval.adapters.rag_client import MockRAGAdapter
from eval.adapters.rag_http_client import RAGHttpAdapter
from eval.metrics.manager import MetricManager
from eval.report.builder import build_html_report
from eval.comparison import compare_runs
# Rinomino la funzione importata per evitare il conflitto di nomi con il comando
from eval.human_in_the_loop import create_review_sample, analyze_review as analyze_review_logic


app = typer.Typer(help="RAG Evaluation Runner CLI")


def load_suite(suite_path: Path) -> EvaluationSuite:
    """
    Loads and validates an evaluation suite configuration file.

    Args:
        suite_path: The path to the suite JSON file.

    Returns:
        A validated EvaluationSuite object.
    
    Raises:
        typer.Exit: If the file is not found, cannot be parsed, or fails validation.
    """
    if not suite_path.is_file():
        print(f"Error: Suite file not found at '{suite_path}'")
        raise typer.Exit(code=1)

    try:
        with open(suite_path, "r", encoding="utf-8") as f:
            suite_data = json.load(f)
        return EvaluationSuite.model_validate(suite_data)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{suite_path}'")
        raise typer.Exit(code=1)
    except ValidationError as e:
        print(f"Error: Suite configuration is invalid in '{suite_path}':\n{e}")
        raise typer.Exit(code=1)


def get_adapter(suite: EvaluationSuite) -> BaseRAGAdapter:
    """
    Factory function to get an instance of a RAG adapter based on suite config.

    Args:
        suite: The full evaluation suite configuration object.

    Returns:
        An instance of a class that implements BaseRAGAdapter.

    Raises:
        typer.Exit: If the adapter type is not recognized or config is missing.
    """
    adapter_type = suite.adapter.type.lower()
    
    if adapter_type == "inproc":
        return MockRAGAdapter()
    
    if adapter_type == "http":
        if not suite.adapter.url:
            print("Error: 'url' must be specified for the 'http' adapter in the suite file.")
            raise typer.Exit(code=1)
        return RAGHttpAdapter(base_url=suite.adapter.url)

    print(f"Error: Unknown adapter type '{adapter_type}'. Available: ['inproc', 'http']")
    raise typer.Exit(code=1)


def save_raw_result(
    run_dir: Path,
    item: Union[Query, Conversation],
    outputs: Union[RAGOutput, List[RAGOutput]],
    metrics: Dict,
) -> None:
    """Saves the raw output and metrics for a single item (query or conversation)."""
    raw_results_dir = run_dir / "raw"
    raw_results_dir.mkdir(parents=True, exist_ok=True)

    # Determine the type of item and structure the output accordingly
    if isinstance(item, Query) and isinstance(outputs, RAGOutput):
        result_content = {
            "type": "query",
            "query": item.model_dump(),
            "rag_output": outputs.model_dump(),
            "metrics": metrics,
        }
        file_name = f"query-{item.id}.json"
    elif isinstance(item, Conversation) and isinstance(outputs, list):
        result_content = {
            "type": "conversation",
            "conversation": item.model_dump(),
            "rag_outputs": [o.model_dump() for o in outputs],
            "metrics": metrics,
        }
        file_name = f"conversation-{item.id}.json"
    else:
        # Skip saving if the data types are mismatched
        return

    file_path = raw_results_dir / file_name
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result_content, f, indent=2, ensure_ascii=False)


def _process_query(
    query: Query,
    rag_adapter: BaseRAGAdapter,
    metric_manager: MetricManager,
    run_dir: Path,
) -> None:
    """Processes a single query evaluation."""
    print(f"  - Processing query '{query.id}': '{query.domanda[:50]}...'")
    
    start_time = time.perf_counter()
    rag_output = rag_adapter.get_response(query)
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    item_results = metric_manager.compute_all_metrics(query, rag_output)
    item_results["latency"] = latency
    metric_manager.add_result(item_results)

    scores_str = ", ".join(
        f"{k}: {v['score']:.2f}" if isinstance(v, dict) else f"{k}: {v:.2f}"
        for k, v in item_results.items()
        if k != "query_id"
    )
    print(f"    Scores -> {scores_str}")
    
    save_raw_result(run_dir, query, rag_output, item_results)


def _process_conversation(
    conversation: Conversation,
    rag_adapter: BaseRAGAdapter,
    metric_manager: MetricManager,
    run_dir: Path,
) -> None:
    """Processes a multi-turn conversation evaluation."""
    print(f"  - Processing conversation '{conversation.id}'...")
    
    conversation_history: List[Turn] = []
    assistant_outputs: List[RAGOutput] = []
    total_latency = 0.0

    for i, turn in enumerate(conversation.turns):
        conversation_history.append(turn)
        if turn.role == "user":
            print(f"    - Turn {i+1} (user): '{turn.content[:50]}...'")
            start_time = time.perf_counter()
            rag_output = rag_adapter.get_conversation_response(
                turns=conversation_history
            )
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            total_latency += latency
            
            # The actual assistant response is in rag_output.answer.
            # We create a Turn object to add to history for the *next* call.
            assistant_turn_for_history = Turn(role="assistant", content=rag_output.answer)
            conversation_history.append(assistant_turn_for_history)
            assistant_outputs.append(rag_output)


    # After the conversation is complete, compute conversation-level metrics
    conv_results = metric_manager.compute_all_conversation_metrics(
        conversation, assistant_outputs
    )
    conv_results["total_latency"] = total_latency
    metric_manager.add_result(conv_results)

    scores_str = ", ".join(
        f"{k}: {v['score']:.2f}"
        for k, v in conv_results.items()
        if k != "conversation_id" and isinstance(v, dict)
    )
    print(f"    Conversation Scores -> {scores_str}")

    save_raw_result(run_dir, conversation, assistant_outputs, conv_results)


@app.command()
def compare(
    baseline_run_dir: Path = typer.Argument(
        ..., help="Path to the baseline run directory.", exists=True, file_okay=False, dir_okay=True
    ),
    candidate_run_dir: Path = typer.Argument(
        ..., help="Path to the candidate run directory.", exists=True, file_okay=False, dir_okay=True
    ),
) -> None:
    """
    Compare the metrics from two different evaluation runs.
    """
    print("--- Comparing Two Runs ---")
    print(f"  - Baseline: {baseline_run_dir.name}")
    print(f"  - Candidate: {candidate_run_dir.name}")
    
    comparison_df = compare_runs(baseline_run_dir, candidate_run_dir)

    if not comparison_df.empty:
        print("\nComparison Results:")
        # Format output for better readability
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(comparison_df)


@app.command()
def export_for_review(
    run_dir: Path = typer.Argument(
        ..., help="Path to the run directory to sample from.", exists=True, file_okay=False, dir_okay=True
    ),
    sample_size: float = typer.Option(
        0.15, "--sample", "-s", help="The fraction of items to sample for review (e.g., 0.15 for 15%).", min=0.0, max=1.0
    ),
) -> None:
    """
    Export a random sample of results from a run for human-in-the-loop review.
    """
    print(f"--- Exporting Sample for Human Review ---")
    print(f"  - Run Directory: {run_dir.name}")
    print(f"  - Sample Size: {sample_size:.0%}")

    try:
        output_file = create_review_sample(run_dir, sample_size)
        if output_file:
            print(f"\nSuccessfully created review sample: {output_file}")
            print("You can now open this CSV file, fill in the human scores, and use the 'analyze-review' command.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)


@app.command()
def analyze_review(
    review_csv_path: Path = typer.Argument(
        ..., help="Path to the completed review CSV file.", exists=True, file_okay=True, dir_okay=False
    ),
) -> None:
    """
    Analyze a completed review file to compare human and LLM judgments.
    """
    print(f"--- Analyzing Human Review File ---")
    print(f"  - File: {review_csv_path.name}")
    
    try:
        # Uso la funzione rinominata per chiamare la logica corretta
        analyze_review_logic(review_csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)
        

@app.command()
def run(
    suite: Path = typer.Option(
        ...,
        "--suite",
        "-s",
        help="Path to the evaluation suite JSON file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    limit: int = typer.Option(
        -1, "--limit", "-l", help="Limit the number of queries to process (-1 for all)."
    )
) -> None:
    """
    Run a RAG evaluation based on a suite configuration file.
    """
    print("--- RAG Evaluation Runner ---")
    
    # 1. Setup Run Directory
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_id = f"run_{timestamp}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting run '{run_id}'. Results will be saved in '{run_dir}'")

    # 2. Load and validate the evaluation suite
    print(f"Loading suite from: {suite}")
    evaluation_suite = load_suite(suite)
    print(f"Successfully loaded and validated suite: '{evaluation_suite.name}'")

    # 3. Load Dataset
    print(f"Loading dataset from: {evaluation_suite.dataset_path}")
    try:
        dataset = stream_dataset(Path(evaluation_suite.dataset_path))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: Failed to load dataset. {e}")
        raise typer.Exit(code=1)

    # 4. Initialize RAG Adapter and Metric Manager
    print(f"Initializing RAG adapter: '{evaluation_suite.adapter.type}'")
    rag_adapter = get_adapter(evaluation_suite)
    metric_manager = MetricManager(evaluation_suite.metrics)

    # 5. Run Evaluation Loop
    print("\nProcessing dataset...")
    processed_count = 0
    for item in dataset:
        if 0 < limit <= processed_count:
            print(f"Reached item limit of {limit}. Stopping.")
            break
        
        if isinstance(item, Query):
            _process_query(item, rag_adapter, metric_manager, run_dir)
        elif isinstance(item, Conversation):
            _process_conversation(item, rag_adapter, metric_manager, run_dir)
        
        processed_count += 1
    
    print(f"\nProcessing complete. Processed {processed_count} items.")

    # 6. Aggregate results and generate reports
    print("Aggregating results and generating reports...")
    aggregated_results = metric_manager.get_aggregated_results()

    # Save aggregated results to CSV and JSON
    aggregated_results.to_csv(run_dir / "metrics.csv")
    aggregated_results.to_json(run_dir / "metrics.json", orient="index", indent=2)

    # Build HTML report
    build_html_report(
        run_id=run_id,
        suite_name=evaluation_suite.name,
        timestamp=timestamp,
        aggregated_results=aggregated_results,
        output_dir=run_dir,
    )
    
    print("\n--- Evaluation Run Finished ---")


if __name__ == "__main__":
    app()
