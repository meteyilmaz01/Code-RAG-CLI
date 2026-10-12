# code-rag-cli

[Türkçe](README.tr.md)

A small, local-first **Code-RAG tool for the terminal**. Point it at a codebase, ask a question in plain
language, and get an answer that cites the files and line numbers it relied on.

It is meant as a compact, readable reference for how a code question-answering pipeline fits together
(scan, chunk, embed, hybrid search, answer), not as a replacement for mature tools like Aider or Continue.

```
$ coderag ask "Which folders are always skipped when indexing, and what is the maximum file size?"
✓ Index updated (1 new, 1 changed, 0 removed files) (25.1 s)
✓ Rewriting the question as an English search query (30.6 s)
✓ Searching code (semantic + keyword): test/test_scanner.py: _paths, _write, ... (0.9 s)
✓ Merging results with RRF
✓ Writing the answer (5 code snippets as context) (50.6 s)

 • ALWAYS_IGNORE_DIRS (coderag/config.py:17):
     ALWAYS_IGNORE_DIRS = {".coderag", ".git", "node_modules", "bin", "obj", "__pycache__", ".venv", ...}
   These folders are always skipped when indexing.
 • MAX_FILE_BYTES (coderag/config.py:20):
     MAX_FILE_BYTES = 1_000_000

                  Sources
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ File                    ┃ Lines    ┃ Symbol ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ coderag/core/scanner.py │ 1-10     │        │
│ coderag/cli.py          │ 117-176  │ ask    │
│ coderag/config.py       │ 1-49     │        │
│ coderag/cli.py          │ 99-114   │ index  │
│ coderag/core/scanner.py │ 21-47    │ scan   │
└─────────────────────────┴──────────┴────────┘
```

This is a real run on this repository, translated from Turkish (the CLI messages are currently in
Turkish; questions can be in any language the LLM understands). The cited lines are correct. The search
step shows a weakness listed under Limitations: the rewritten query drifted toward test files, and the
right files were found anyway.

## Features

- **Structure-aware chunking.** Python is split on function and class boundaries with tree-sitter. Other
  languages (C#, JS/TS, Java, Go, Markdown) use LangChain's language-aware text splitter.
- **Hybrid retrieval.** Dense embeddings (`jina-embeddings-v2-base-code`) plus BM25 keyword search that
  splits `getUserById` / `get_user_by_id` into words, merged with Reciprocal Rank Fusion.
- **Cross-language questions.** The question is rewritten as an English search query, using the real
  symbol names from the index, so a Turkish question can find English-named code.
- **Cited answers.** Code reaches the LLM with real line numbers, and the answer cites `file:line`.
- **Incremental indexing.** File hashes are stored, so only changed files are re-embedded. `ask` refreshes
  the index automatically before answering.
- **Streaming answers and visible progress,** with timing for every step.
- **Optional reranker** (`--rerank`) and a **built-in evaluation harness** (`eval/`).
- **Local by default.** Embeddings run on CPU through ONNX, the LLM runs through Ollama, and the vector
  store is a local ChromaDB folder with telemetry disabled.

## Install

Requires Python 3.10+ and, for the default setup, [Ollama](https://ollama.com).

```bash
git clone https://github.com/meteyilmaz01/code-rag-cli.git
cd code-rag-cli
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[ollama]"       # or ".[openai]" / ".[gemini]"
ollama pull qwen2.5-coder:7b     # about 4.7 GB
```

On first use the embedding model (`jinaai/jina-embeddings-v2-base-code`) is downloaded from Hugging Face to
`~/.coderag/models`. The optional reranker model (about 1.1 GB) is downloaded the first time you pass `--rerank`.

## Usage

```bash
coderag index path/to/project                       # first-time index (also: coderag index .)
coderag ask "How does incremental indexing work?" --path path/to/project
```

| Option | Meaning |
|---|---|
| `--path`, `-p` | Indexed project directory (default `.`) |
| `--show-code`, `-c` | Also print the retrieved code with line numbers |
| `--top-k`, `-k` | Number of code snippets given to the LLM (default 5) |
| `--rerank` | Re-rank candidates with a cross-encoder (better ordering, about 20 s slower per question) |
| `--no-update` | Skip the automatic index refresh before answering |

The index lives in `.coderag/` inside the indexed project. `python -m coderag` works too.

### Choosing the LLM

| Variable | Meaning | Default |
|---|---|---|
| `CODERAG_PROVIDER` | `ollama`, `openai` or `gemini` | `ollama` |
| `CODERAG_MODEL` | Model name | provider default, see `coderag/config.py` |

`OPENAI_API_KEY` / `GOOGLE_API_KEY` must be set for the hosted providers. **Only the Ollama path has been
tested end to end**; the OpenAI and Gemini paths are implemented through LangChain but untested.

## How it works

```
files ─▶ scanner ─▶ chunker ─▶ embeddings ─▶ ChromaDB ─┐
(.gitignore aware)  (tree-sitter)  (fastembed)   + BM25   │
                                                          ▼
question ─▶ rewrite to English ─▶ dense + keyword search ─▶ RRF ─▶ (reranker) ─▶ LLM ─▶ cited answer
```

| Module | Responsibility |
|---|---|
| `coderag/core/scanner.py` | Walks the tree, honours `.gitignore`, skips binaries and large files |
| `coderag/core/chunker.py` | Splits files into function/class-sized chunks with line numbers |
| `coderag/core/vector_store.py` | Embeddings, ChromaDB, BM25, file-hash bookkeeping |
| `coderag/core/indexer.py` | Scan, chunk, store, incrementally |
| `coderag/core/reranker.py` | Optional cross-encoder scoring |
| `coderag/core/rag_engine.py` | Query rewriting, hybrid retrieval, prompt, LLM call |
| `coderag/cli.py` | Typer + Rich interface |

## Measurements

`eval/run_eval.py` asks 16 questions about **this repository** and checks whether the functions that
contain the answer appear in the top results. Current result, without the reranker:

| Recall@3 | Recall@5 | Recall@8 | MRR |
|---|---|---|---|
| 0.78 | 0.81 | 0.81 | 0.66 |

What moved the numbers along the way (MRR, measured on the same set as the project grew):

| Change | MRR |
|---|---|
| Hybrid search baseline | 0.44 |
| Query rewriting that sees the real symbol names | 0.68 |
| Down-weighting test and doc chunks | 0.71 |

The reranker experiment was mostly negative: a small English reranker lowered recall (0.78 to 0.66),
and a multilingual one improved ordering (MRR 0.44 to 0.54) without finding anything new, at a cost of
about 20 s per question. That is why it is off by default.

Read these numbers with care: the question set is small, it was written by the author, and it covers the
project's own code. They show direction, not general accuracy. Numbers from different stages were taken
at different index sizes.

Typical speed on a CPU-only laptop (15.7 GB RAM, no GPU) with `qwen2.5-coder:7b`: **about one minute per
question**, almost all of it LLM time.

## Limitations

- **Slow without a GPU.** The local 7B model dominates the response time. Hosted providers should be faster
  but are untested.
- **Only Python gets syntax-aware chunking.** Other languages use a text splitter; C# support in particular
  is not yet first-class. Very long functions are not split further.
- **Query rewriting costs an extra LLM call** and can be wrong. For example, it sometimes copies test
  function names from the symbol list into the search query.
- **Small, self-written evaluation.** No comparison against public benchmarks yet.
- **Answers can be shallow.** The model sometimes returns the right values without explaining the mechanism.
- **Tested on Windows 11 with Python 3.14 only.** Python 3.10+ is declared, but older versions and
  Linux/macOS have not been run. A clean-environment install has not been verified in CI.
- **Heavy dependencies** (ChromaDB, ONNX Runtime, tree-sitter language pack) and large first-run model downloads.

## Development

```bash
pip install -e ".[dev,ollama]"
pytest                          # 57 tests, no model downloads needed
python eval/run_eval.py --only 0   # retrieval quality; --only picks configurations
```

## License

[MIT](LICENSE)
