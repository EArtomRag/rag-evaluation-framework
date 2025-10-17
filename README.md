# RAG Evaluation Runner

Questo progetto fornisce una CLI (Command Line Interface) in Python per la valutazione strutturata e riproducibile di sistemi RAG (Retrieval-Augmented Generation). Permette di misurare la qualità sia su singole interazioni domanda-risposta sia su conversazioni complesse, utilizzando metriche classiche e valutazioni avanzate basate su LLM tramite `deepeval`.

## Funzionalità Principali

- **Doppia Modalità di Valutazione**: Supporto nativo per test **single-turn** (query singole) e **multi-turn** (conversazioni).
- **Metriche Avanzate con `deepeval`**:
  - Per il **single-turn**: calcola `faithfulness`, `answer_relevancy`, `contextual_precision`, e `contextual_recall`.
  - Per il **multi-turn**: valuta ogni turno per `turn_faithfulness`, `turn_answer_relevancy`, e `turn_answer_correctness` (rispetto a una "gold answer").
- **Metriche di Retrieval**: Include metriche standard come `recall@k`, `mrr`, e `ndcg@k`.
- **Workflow Human-in-the-Loop (HITL)**: Strumenti integrati per esportare campioni per la revisione umana e analizzare l'allineamento con i giudizi dell'LLM.
- **Confronto tra Esecuzioni**: Un comando `compare` per valutare l'impatto delle modifiche tra due run.
- **Reportistica Completa**: Genera un `report.html` dettagliato, `metrics.csv` e `metrics.json` per ogni esecuzione.

## 1. Installazione e Configurazione

I comandi vanno eseguiti dalla root del progetto in un terminale PowerShell.

### a. Installazione delle Dipendenze

Questo progetto usa Poetry. Per installare tutte le librerie necessarie, esegui:

```powershell
py -m poetry install
```

Se il comando `py` non è disponibile, puoi usare il percorso completo a Python: `python -m poetry install`.

### b. Configurazione della Chiave API di OpenAI

Le metriche basate su `deepeval` richiedono una chiave API di OpenAI per funzionare. Impostala come variabile d'ambiente.

```powershell
$env:OPENAI_API_KEY="sk-..."
```

**Nota:** Questa variabile deve essere impostata in ogni nuova sessione del terminale.

## 2. Guida all'Uso

Tutti i comandi vengono eseguiti tramite l'interprete Python dell'ambiente virtuale (`.\.venv\Scripts\python.exe`).

### a. Eseguire una Valutazione

Il comando principale è `run`. Il tipo di valutazione (single o multi-turn) dipende dal contenuto del dataset specificato nella suite.

**Eseguire una Valutazione Single-Turn:**

```powershell
.\.venv\Scripts\python.exe -m eval.run run --suite eval/suites/biblio_app_suite.json
```

**Eseguire una Valutazione Multi-Turn:**

```powershell
.\.venv\Scripts\python.exe -m eval.run run --suite eval/suites/multi_turn_suite.json
```

Al termine, una nuova directory verrà creata in `runs/` con tutti i risultati. Apri il file `report.html` per un'analisi visuale.

### b. Confrontare due Esecuzioni

Dopo aver eseguito almeno due valutazioni, puoi confrontarle con il comando `compare`.

```powershell
.\.venv\Scripts\python.exe -m eval.run compare runs/NOME_DELLA_RUN_BASELINE runs/NOME_DELLA_RUN_NUOVA
```

### c. Usare il Workflow Human-in-the-Loop (HITL)

Questo workflow si applica ai risultati delle esecuzioni **single-turn**.

**Passo 1: Esporta un Campione per la Revisione**
Crea un file CSV da un'esecuzione esistente, campionando (es. il 20%) dei risultati.

```powershell
.\.venv\Scripts\python.exe -m eval.run export-for-review runs/NOME_DELLA_TUA_RUN --sample 0.2
```

Questo creerà un file `review_sample.csv` nella directory della run.

**Passo 2: Compila il CSV**
Apri il file e compila le colonne `..._umano` con i tuoi punteggi e aggiungi note qualitative.

**Passo 3: Analizza i Risultati**
Lancia l'analisi sul file CSV compilato per confrontare i tuoi giudizi con quelli dell'LLM.

```powershell
.\.venv\Scripts\python.exe -m eval.run analyze-review runs/NOME_DELLA_TUA_RUN/review_sample.csv
```

---

_Per una descrizione tecnica dettagliata dell'architettura e dei moduli, fare riferimento al file `project-description.md`._
