"""Functions to support the Human-in-the-Loop (HITL) workflow."""

import json
from pathlib import Path
import random
import pandas as pd
import yaml


def create_review_sample(run_dir: Path, sample_size: float) -> Path:
    """
    Creates a sample of results for human review.

    For single-turn evaluations, it creates a CSV file.
    For multi-turn conversations, it creates a directory with Markdown files.

    Args:
        run_dir: The path to the evaluation run directory.
        sample_size: The fraction of items to sample (e.g., 0.15 for 15%).

    Returns:
        The path to the generated CSV file or the review directory.

    Raises:
        FileNotFoundError: If the 'raw' directory is not found.
        ValueError: If the sample size is not between 0 and 1.
    """
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    if not 0 < sample_size <= 1:
        raise ValueError("Sample size must be between 0 and 1.")

    json_files = list(raw_dir.glob("*.json"))
    
    num_to_sample = int(len(json_files) * sample_size)
    if num_to_sample == 0 and len(json_files) > 0:
        num_to_sample = 1
        
    if not json_files or num_to_sample == 0:
        print("Warning: No results found to sample.")
        return None
        
    sampled_files = random.sample(json_files, k=num_to_sample)

    # Check the type of evaluation from the first sampled file
    with open(sampled_files[0], "r", encoding="utf-8") as f:
        first_data = json.load(f)
    
    is_conversation = first_data.get("type") == "conversation"

    if is_conversation:
        return _create_conversation_review_files(run_dir, sampled_files)
    else:
        return _create_single_turn_review_csv(run_dir, sampled_files)


def _create_single_turn_review_csv(run_dir: Path, files: list[Path]) -> Path:
    """Creates a CSV file for reviewing single-turn results."""
    records = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            metrics = data.get("metrics", {})
            
            record = {
                "query_id": data.get("query", {}).get("id"),
                "tipo_domanda": data.get("query", {}).get("tipo_domanda"),
                "domanda": data.get("query", {}).get("domanda"),
                "risposta_generata": data.get("rag_output", {}).get("answer"),
                "contesti": json.dumps([ctx['text'] for ctx in data.get("rag_output", {}).get("contexts", [])], ensure_ascii=False, indent=2),
                "punteggio_faithfulness_llm": metrics.get("faithfulness", {}).get("score"),
                "motivazione_faithfulness_llm": metrics.get("faithfulness", {}).get("reason"),
                "punteggio_answer_relevancy_llm": metrics.get("answer_relevancy", {}).get("score"),
                "motivazione_answer_relevancy_llm": metrics.get("answer_relevancy", {}).get("reason"),
                "punteggio_faithfulness_umano": "",
                "punteggio_answer_relevancy_umano": "",
                "note_revisore": "",
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    output_path = run_dir / "review_sample.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig", decimal=".")
    return output_path


def _create_conversation_review_files(run_dir: Path, files: list[Path]) -> Path:
    """Creates Markdown files for reviewing multi-turn conversations."""
    review_dir = run_dir / "review"
    review_dir.mkdir(exist_ok=True)

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        conversation = data.get("conversation", {})
        conv_id = conversation.get("id")
        rag_outputs = data.get("rag_outputs", [])
        metrics = data.get("metrics", {})

        review_content = []

        # --- YAML Front Matter for Human Review ---
        turn_reviews = []
        assistant_turn_idx = 0
        for i, turn in enumerate(conversation.get("turns", [])):
            if turn.get("role") == "assistant":
                turn_reviews.append({
                    "turn_index": assistant_turn_idx + 1,
                    "faithfulness_score": None,
                    "relevancy_score": None,
                    "notes": ""
                })
                assistant_turn_idx += 1

        review_metadata = {
            "conversation_id": conv_id,
            "human_review": {
                "overall_quality_score": None,
                "notes": "Inserisci qui le tue note generali sulla conversazione.",
                "turn_reviews": turn_reviews
            }
        }
        review_content.append("---")
        review_content.append(yaml.dump(review_metadata, allow_unicode=True, sort_keys=False))
        review_content.append("---")

        # --- Conversation Transcript ---
        review_content.append(f"\n# Revisione Conversazione: {conv_id}")
        if "metadata" in conversation:
            review_content.append(f"**Metadati:** `{json.dumps(conversation['metadata'])}`")

        assistant_turn_idx = 0
        for turn in conversation.get("turns", []):
            role = turn.get("role")
            content = turn.get("content")
            
            if role == "user":
                review_content.append(f"\n---\n\n### 👤 Utente\n- **Domanda:** {content}")
            elif role == "assistant":
                review_content.append(f"\n### 🤖 Assistente (Turno {assistant_turn_idx + 1})")
                review_content.append(f"- **Risposta:** {content}")

                if assistant_turn_idx < len(rag_outputs):
                    rag_output = rag_outputs[assistant_turn_idx]
                    contexts = rag_output.get("contexts", [])
                    if contexts:
                        review_content.append("\n**Contesti Recuperati:**")
                        for i, ctx in enumerate(contexts):
                            review_content.append(f"  - **Contesto {i+1} (Doc ID: `{ctx.get('doc_id')}`):**")
                            review_content.append(f"    ```\n    {ctx.get('text')}\n    ```")
                
                assistant_turn_idx += 1

        # --- LLM as a Judge Metrics ---
        review_content.append("\n---\n\n## 📊 Metriche LLM-as-a-Judge")
        if "turn_answer_relevancy" in metrics:
            review_content.append(f"- **Rilevanza Risposte (media):** `{metrics['turn_answer_relevancy'].get('score')}`")
            review_content.append(f"  - **Motivazione:** {metrics['turn_answer_relevancy'].get('explanation')}")
        if "turn_faithfulness" in metrics:
            review_content.append(f"- **Fedeltà Risposte (media):** `{metrics['turn_faithfulness'].get('score')}`")
            review_content.append(f"  - **Motivazione:** {metrics['turn_faithfulness'].get('explanation')}")
        
        # Write to file
        output_path = review_dir / f"{conv_id}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(review_content))
            
    return review_dir


def analyze_review(review_path: Path) -> None:
    """
    Analyzes a completed review file or directory.

    It dispatches to the correct analysis function based on whether the
    path is a CSV file (single-turn) or a directory (multi-turn).

    Args:
        review_path: The path to the completed review CSV file or directory.
    """
    if review_path.is_dir():
        _analyze_multi_turn_review(review_path)
    elif review_path.is_file() and review_path.suffix == ".csv":
        _analyze_single_turn_review(review_path)
    else:
        raise ValueError(
            "Invalid review path. Must be a 'review_sample.csv' file or a 'review' directory."
        )


def _analyze_single_turn_review(review_csv_path: Path) -> None:
    """Analyzes a completed single-turn review CSV file."""
    if not review_csv_path.exists():
        raise FileNotFoundError(f"Review file not found: {review_csv_path}")

    df = pd.read_csv(review_csv_path, decimal=".")

    required_cols = [
        "punteggio_faithfulness_llm", "punteggio_faithfulness_umano",
        "punteggio_answer_relevancy_llm", "punteggio_answer_relevancy_umano"
    ]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("CSV is missing one or more required score columns.")
    
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    human_score_cols = ["punteggio_faithfulness_umano", "punteggio_answer_relevancy_umano"]
    df.dropna(subset=human_score_cols, inplace=True)

    if df.empty:
        raise ValueError("No completed human judgments found in the review file.")

    print("\n--- Human vs. LLM Judgment Analysis (Single Turn) ---")
    
    metrics_to_analyze = ["faithfulness", "answer_relevancy"]
    
    for metric in metrics_to_analyze:
        llm_col = f"punteggio_{metric}_llm"
        human_col = f"punteggio_{metric}_umano"
        
        print(f"\n----- Analysis for: {metric.capitalize()} -----")

        llm_mean = df[llm_col].mean()
        human_mean = df[human_col].mean()
        print(f"  - Average LLM Score:   {llm_mean:.4f}")
        print(f"  - Average Human Score: {human_mean:.4f}")

        correlation = df[llm_col].corr(df[human_col])
        print(f"  - Pearson Correlation: {correlation:.4f}")

        disagreements = (df[llm_col].round() != df[human_col].round()).sum()
        agreement_rate = 1 - (disagreements / len(df))
        print(f"  - Agreement Rate:      {agreement_rate:.2%}")
        
    print("\n----- Cases of Major Disagreement (es. LLM=1, Human=0) -----")
    disagreement_df = df[
        ((df["punteggio_faithfulness_llm"].round() == 1) & (df["punteggio_faithfulness_umano"].round() == 0)) |
        ((df["punteggio_answer_relevancy_llm"].round() == 1) & (df["punteggio_answer_relevancy_umano"].round() == 0))
    ].copy()
    
    if disagreement_df.empty:
        print("  - No major disagreements found. Great alignment!")
    else:
        disagreement_df['faithfulness_disagreement'] = disagreement_df.apply(
            lambda row: abs(row['punteggio_faithfulness_llm'] - row['punteggio_faithfulness_umano']) > 0.5, axis=1
        )
        disagreement_df['relevancy_disagreement'] = disagreement_df.apply(
            lambda row: abs(row['punteggio_answer_relevancy_llm'] - row['punteggio_answer_relevancy_umano']) > 0.5, axis=1
        )

        for _, row in disagreement_df.iterrows():
            print(f"\n  - Query ID: {row['query_id']}")
            print(f"    - Domanda: {row['domanda'][:100]}...")
            if row['faithfulness_disagreement']:
                print(f"    - DISACCORDO su Faithfulness (LLM vs Human): {row['punteggio_faithfulness_llm']:.2f} vs {row['punteggio_faithfulness_umano']:.2f}")
                print(f"      - Motivazione LLM: {row['motivazione_faithfulness_llm']}")
            if row['relevancy_disagreement']:
                print(f"    - DISACCORDO su Relevancy (LLM vs Human): {row['punteggio_answer_relevancy_llm']:.2f} vs {row['punteggio_answer_relevancy_umano']:.2f}")
                print(f"      - Motivazione LLM: {row['motivazione_answer_relevancy_llm']}")
            if pd.notna(row['note_revisore']):
                print(f"      - Note Revisore: {row['note_revisore']}")


def _analyze_multi_turn_review(review_dir: Path) -> None:
    """Analyzes a directory of completed multi-turn review Markdown files."""
    review_files = list(review_dir.glob("*.md"))
    if not review_files:
        raise FileNotFoundError(f"No review files found in directory: {review_dir}")

    all_turn_reviews = []
    
    # Path to the original raw results
    raw_dir = review_dir.parent / "raw"

    for review_file in review_files:
        conv_id = review_file.stem
        raw_json_path = raw_dir / f"{conv_id}.json"
        
        if not raw_json_path.exists():
            print(f"Warning: Raw JSON for '{conv_id}' not found. Skipping analysis.")
            continue

        with open(review_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Extract YAML front matter
            try:
                _, front_matter, _ = content.split("---", 2)
                review_data = yaml.safe_load(front_matter)
            except (ValueError, yaml.YAMLError) as e:
                print(f"Warning: Could not parse YAML from '{review_file.name}'. Skipping. Error: {e}")
                continue
        
        with open(raw_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        human_reviews = review_data.get("human_review", {}).get("turn_reviews", [])
        
        # Simple LLM metric extraction (assumes metrics per turn are not in raw JSON)
        # We use the average conversation scores for comparison
        llm_faith_score = raw_data.get("metrics", {}).get("turn_faithfulness", {}).get("score")
        llm_relev_score = raw_data.get("metrics", {}).get("turn_answer_relevancy", {}).get("score")

        for review in human_reviews:
            if review.get("faithfulness_score") is not None and review.get("relevancy_score") is not None:
                all_turn_reviews.append({
                    "conversation_id": conv_id,
                    "turn_index": review.get("turn_index"),
                    "human_faithfulness": float(review["faithfulness_score"]),
                    "human_relevancy": float(review["relevancy_score"]),
                    "llm_faithfulness_avg": llm_faith_score, # Comparing turn-level human score to conv-level LLM score
                    "llm_relevancy_avg": llm_relev_score,
                    "notes": review.get("notes", "")
                })

    if not all_turn_reviews:
        raise ValueError("No completed human judgments found in any review files.")

    df = pd.DataFrame(all_turn_reviews)
    
    print("\n--- Human vs. LLM Judgment Analysis (Multi-Turn) ---")
    
    # --- Overall Score Analysis ---
    print("\n----- Analysis for: Overall Averages -----")
    human_faith_mean = df["human_faithfulness"].mean()
    human_relev_mean = df["human_relevancy"].mean()
    llm_faith_mean = df["llm_faithfulness_avg"].mean() # This is an average of averages
    llm_relev_mean = df["llm_relevancy_avg"].mean()

    print(f"  - Avg Human Faithfulness: {human_faith_mean:.4f}")
    print(f"  - Avg LLM Faithfulness:   {llm_faith_mean:.4f}")
    print(f"  - Avg Human Relevancy:    {human_relev_mean:.4f}")
    print(f"  - Avg LLM Relevancy:      {llm_relev_mean:.4f}")

    # --- Correlation Analysis ---
    print("\n----- Correlation Analysis -----")
    faith_corr = df["human_faithfulness"].corr(df["llm_faithfulness_avg"])
    relev_corr = df["human_relevancy"].corr(df["llm_relevancy_avg"])
    print(f"  - Pearson Correlation (Faithfulness): {faith_corr:.4f}")
    print(f"  - Pearson Correlation (Relevancy):    {relev_corr:.4f}")

    # --- Disagreements ---
    print("\n----- Turns with Major Disagreements -----")
    df['faith_disagreement'] = (df['human_faithfulness'].round() != df['llm_faithfulness_avg'].round())
    df['relev_disagreement'] = (df['human_relevancy'].round() != df['llm_relevancy_avg'].round())

    disagreement_df = df[df['faith_disagreement'] | df['relev_disagreement']]

    if disagreement_df.empty:
        print("  - No major disagreements found. Great alignment!")
    else:
        for _, row in disagreement_df.iterrows():
            print(f"\n  - Conversation: {row['conversation_id']}, Turn: {row['turn_index']}")
            if row['faith_disagreement']:
                print(f"    - DISACCORDO su Faithfulness (LLM vs Human): {row['llm_faithfulness_avg']:.2f} vs {row['human_faithfulness']:.2f}")
            if row['relev_disagreement']:
                print(f"    - DISACCORDO su Relevancy (LLM vs Human): {row['llm_relevancy_avg']:.2f} vs {row['human_relevancy']:.2f}")
            if pd.notna(row['notes']):
                print(f"    - Note Revisore: {row['notes']}")
