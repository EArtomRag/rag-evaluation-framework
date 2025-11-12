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

### a. Prerequisiti

- Python 3.11+
- [uv](https://github.com/astral-sh/uv): Un installer Python estremamente veloce. Per installarlo (una tantum):
  ```powershell
  pip install uv
  ```

### b. Creazione dell'Ambiente e Installazione delle Dipendenze

1.  **Crea l'ambiente virtuale:**
    ```powershell
    uv venv
    ```
2.  **Installa le dipendenze del progetto:**
    ```powershell
    uv pip install -e .
    ```

### c. Configurazione della Chiave API di OpenAI

Le metriche basate su `deepeval` richiedono una chiave API di OpenAI per funzionare. Il modo più semplice è creare un file `.env` nella root del progetto.

1.  Crea un file chiamato `.env`.
2.  Aggiungi la seguente riga al file, sostituendo `sk-...` con la tua chiave:
    ```
    OPENAI_API_KEY="sk-..."
    ```
Lo script caricherà automaticamente questa variabile all'avvio.

## 2. Guida all'Uso

Prima di eseguire i comandi, **attiva l'ambiente virtuale** in una nuova sessione del terminale:

```powershell
.\.venv\Scripts\Activate.ps1
```
Vedrai `(.venv)` apparire all'inizio del prompt, a conferma che l'ambiente è attivo.

### a. Eseguire una Valutazione

Il comando principale è `run`. Il tipo di valutazione (single o multi-turn) dipende dal contenuto del dataset specificato nella suite.

**Eseguire una Valutazione Single-Turn:**
```powershell
python -m eval.run run --suite eval/suites/biblio_app_suite.json
```

**Eseguire una Valutazione Multi-Turn:**
```powershell
python -m eval.run run --suite eval/suites/multi_turn_suite.json
```

Al termine, una nuova directory verrà creata in `runs/` con tutti i risultati. Apri il file `report.html` per un'analisi visuale.

### b. Confrontare due Esecuzioni

Dopo aver eseguito almeno due valutazioni, puoi confrontarle con il comando `compare`.

```powershell
python -m eval.run compare runs/NOME_DELLA_RUN_BASELINE runs/NOME_DELLA_RUN_NUOVA
```

### c. Usare il Workflow Human-in-the-Loop (HITL)

Questo workflow si applica ai risultati delle esecuzioni **single-turn**.

**Passo 1: Esporta un Campione per la Revisione**
Crea un file CSV da un'esecuzione esistente, campionando (es. il 20%) dei risultati.
```powershell
python -m eval.run export-for-review runs/NOME_DELLA_TUA_RUN --sample 0.2
```
Questo creerà un file `review_sample.csv` nella directory della run.

**Passo 2: Compila il CSV**
Apri il file e compila le colonne `..._umano` con i tuoi punteggi e aggiungi note qualitative.

**Passo 3: Analizza i Risultati**
Lancia l'analisi sul file CSV compilato per confrontare i tuoi giudizi con quelli dell'LLM.
```powershell
python -m eval.run analyze-review runs/NOME_DELLA_TUA_RUN/review_sample.csv
```

---

_Per una descrizione tecnica dettagliata dell'architettura e dei moduli, fare riferimento al file `project-description.md`._
