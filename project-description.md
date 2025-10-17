# Descrizione Tecnica del Progetto: RAG Evaluation Runner

## 1. Panoramica Generale

Questo progetto fornisce una Command Line Interface (CLI) in Python per la valutazione strutturata e riproducibile di sistemi RAG (Retrieval-Augmented Generation). Il runner è progettato per essere modulare e supporta due modalità di valutazione distinte ma integrate: **single-turn** (domanda-risposta singola) e **multi-turn** (conversazioni complesse).

L'architettura si basa su principi di

- **Configurazione tramite "Suite"**: Ogni esecuzione è definita da un file JSON (`EvaluationSuite`) che specifica il dataset, il sistema RAG da interrogare e le metriche da calcolare.
- **Astrazione tramite "Adapter"**: L'interazione con il sistema RAG è gestita da `Adapter`, che permettono di collegare facilmente diversi tipi di sistemi (es. un servizio HTTP, un mock locale).
- **Metriche Modulari**: Il calcolo delle metriche è incapsulato in classi dedicate, facilitando l'estensione con nuove misurazioni.

## 2. Flussi di Valutazione

Il cuore del runner è il comando `run`, che orchestra i due principali flussi di valutazione.

### 2.1. Flusso Single-Turn (Query Singola)

Questo flusso è progettato per valutare la performance del sistema RAG su singole domande indipendenti.

**Orchestrazione (`eval/run.py -> _process_query`):**

1.  Il `DataLoader` legge un file `.jsonl` dove ogni riga è un oggetto `Query` (definito in `eval/schemas/data_models.py`).
2.  Per ogni `Query`, l' `RAGHttpAdapter` chiama il metodo `get_response`. Per mantenere il codice pulito (DRY), questo metodo internamente tratta la query come una conversazione di un solo turno e chiama `get_conversation_response`.
3.  Il `MetricManager` viene invocato per calcolare le metriche definite nella suite. Queste includono:
    - **Metriche di Retrieval** (`eval/metrics/retrieval.py`): `recall@k`, `mrr`, `ndcg@k`.
    - **Metriche di Qualità RAG** (`eval/metrics/rag.py`): `faithfulness`, `answer_relevancy`, `contextual_precision`, `contextual_recall`. Queste sfruttano `deepeval` per una valutazione basata su LLM.
    - **Metriche Operative** (`eval/metrics/operational.py`): `cost`, `latency`.
4.  I risultati (output del RAG, punteggi delle metriche) vengono salvati in un file JSON individuale nella cartella `runs/{run_id}/raw/`.

### 2.2. Flusso Multi-Turn (Conversazione)

Questo flusso valuta la capacità del sistema RAG di gestire un dialogo, mantenendo il contesto e fornendo risposte coerenti e accurate turno dopo turno.

**Orchestrazione (`eval/run.py -> _process_conversation`):**

1.  Il `DataLoader` legge un file `.jsonl` dove ogni riga è un oggetto `Conversation`, che contiene una lista di `Turn`.
2.  Il runner itera sui turni della conversazione, simulando il dialogo:
    - Per ogni turno dell'utente, chiama il metodo `get_conversation_response` dell'`RAGHttpAdapter`, passando l'intera cronologia della conversazione fino a quel momento.
    - **Bug Critico Corretto**: La risposta _reale_ dell'assistente viene reinserita nella cronologia per il turno successivo, garantendo un contesto corretto.
3.  Al termine della conversazione, il `MetricManager` viene invocato per calcolare le metriche conversazionali.
4.  **Metriche Turn-by-Turn** (`eval/metrics/conversation.py`): Invece di una singola metrica olistica, il sistema applica un set di metriche a _ciascun turno_ della conversazione e ne calcola la media. Le metriche implementate, basate su `deepeval`, sono:
    - `turn_faithfulness`: Verifica che la risposta sia supportata dai contesti recuperati in quel turno.
    - `turn_answer_relevancy`: Valuta la pertinenza della risposta alla domanda di quel turno.
    - `turn_answer_correctness`: Confronta la risposta con la `gold_answer` (se fornita nel dataset per quel turno) usando la metrica `GEval` di `deepeval`.
5.  I risultati aggregati per la conversazione vengono salvati in un file JSON nella cartella `raw/`.

## 3. Funzionalità Aggiuntive

Oltre al comando `run`, la CLI offre strumenti di analisi post-esecuzione.

### 3.1. Confronto tra Esecuzioni (`compare`)

- **Scopo**: Confrontare i risultati aggregati di due diverse esecuzioni (es. un "baseline" vs un "candidate").
- **Funzionamento**: Lo script `eval/comparison.py` legge i file `metrics.csv` da due directory di `runs` e stampa una tabella comparativa con le differenze percentuali.

### 3.2. Workflow Human-in-the-Loop (HITL)

- **Scopo**: Facilitare la revisione umana dei risultati e confrontarla con i giudizi automatici dell'LLM.
- **Funzionamento (`eval/human_in_the_loop.py`):**
  - **`export-for-review`**: Questo comando prende una directory di `runs`, campiona i risultati delle query singole e crea un file `review_sample.csv` formattato per la revisione manuale.
  - **`analyze-review`**: Dopo che un umano ha compilato il CSV, questo comando lo analizza, calcola metriche di allineamento (es. correlazione di Pearson) tra i punteggi umani e quelli dell'LLM, e mette in evidenza i casi di maggiore disaccordo.

## 4. Reporting

- **Scopo**: Presentare i risultati in un formato accessibile.
- **Funzionamento (`eval/report/builder.py`):**
  - Al termine di ogni esecuzione, viene generato un `report.html`.
  - Il report è dinamico: se rileva risultati di conversazioni, li mostra in un formato "chat" dettagliato. Se rileva risultati di query singole, li mostra nel formato standard.
  - Include una scorecard con le metriche aggregate e un'analisi dettagliata per ogni item valutato.
