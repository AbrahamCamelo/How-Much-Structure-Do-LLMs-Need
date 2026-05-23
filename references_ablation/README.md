# LLM Bibliometric

This project evaluates six bibliographic-analysis pipelines and a shared evaluation suite over Scopus corpora.

## Structure

- `scopus_documents/`: source Scopus files such as `001paper.csv`, `002paper.csv`
- `prompts.csv`: editable prompt catalog used by the six pipelines
- `queries.xlsx`: query catalog keyed by paper number
- `pipeline_1.py` to `pipeline_6.py`: root-level pipeline modules
- `bib_coupling.py`: Louvain bibliographic coupling over Scopus files
- `cluster_description.py`: shared entrypoint for labeled-cluster description generation
- `description/`: pipeline outputs saved as `description/pipeline_X/<experiment>/<model>/<file>.csv`
- `description/human_descriptions/`: canonical folder for human-authored descriptions, evaluated alongside the generated outputs
- `clusters/`: Louvain cluster outputs
- `evaluations/coverage_corpus/`: experiment-specific sentence-level embedding caches and average-cosine results
- `evaluations/quality_of_the_induced_clustering/`: experiment-specific abstract/description embedding caches and induced-clustering outputs
- `evaluations/shared_embeddings/`: shared abstract-level embedding cache reused across quality and silhouette
- `evaluations/silhouette_score/`, `evaluations/clustering_similarity/`, `evaluations/modularity/`: remaining experiment-specific evaluation outputs
- `llm_bibliometric/`: shared package used by every script

## Stable `paper_id`

The Scopus loader generates `paper_id` when the source file does not provide it. The generated values are:

- assigned after `reset_index(drop=True)`
- 1-based integers
- created in memory during loading, not required in the original Scopus CSV
- reused by `bib_coupling.py`, `cluster_description.py`, every Scopus-based pipeline, and every evaluation

Whenever a pipeline cites Scopus papers, references are stored as JSON arrays whose elements are `[#]` tokens tied to these `paper_id` values.

## Prompts And Queries

The pipeline scripts can now resolve prompts and queries from project files:

- `prompts.csv`: one row per pipeline, with `prompt1` and optional `prompt2`
- `queries.xlsx`: maps each paper number to its query

The prompt templates use placeholders such as:

- `{{query}}`
- `{{target_cluster_count}}`
- `{{scopus_context}}`
- `{{cluster_contexts}}`
- `{{cluster_id}}`
- `{{cluster_context}}`
- `{{relevant_cluster_context}}`
- `{{significant_papers_per_cluster}}`

You can edit `prompts.csv` directly to change pipeline behavior without modifying Python code.

## Environment Variables

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`

Optional model overrides:

- `CHATGPT_MODEL`
- `GEMINI_MODEL`
- `CLAUDE_MODEL`
- `OPENAI_EMBEDDING_MODEL`

Note:

- `run_descriptions.py` uses the provider you choose for text generation
- `run_evaluations.py` always needs `OPENAI_API_KEY`, because embeddings are computed with the OpenAI embeddings API
- a matching human description file in `description/human_descriptions/` is required for each paper

## General Commands

You can now run the project in two general steps for one paper:

```powershell
python run_descriptions.py --scopus-file 001paper.csv --provider chatgpt
python run_evaluations.py --scopus-file 001paper.csv --provider chatgpt
python pipeline_average_evaluations.py --provider chatgpt
```

What they do:

- `run_descriptions.py`: runs `bib_coupling.py` logic plus pipelines 1 to 6
- `run_evaluations.py`: evaluates all generated descriptions for that paper and provider, and also evaluates the matching human description
- `pipeline_average_evaluations.py`: aggregates the per-paper evaluation summaries under `evaluations/full_run/evaluations/<experiment>/<provider>/` and writes `pipeline_average_evaluations.csv`

Both commands infer paths automatically:

- `001paper.csv` resolves to `scopus_documents/001paper.csv` as the raw input
- the clustered file is saved as `clusters/001paper_louvain.csv`
- pipeline outputs are saved under `description/pipeline_X/<experiment>/<provider>/001paper.csv`
- summaries are saved under `evaluations/full_run/`

The `<experiment>` folder is derived from `DEFAULT_SIGNIFICANT_PAPERS_PER_CLUSTER` or the value passed through `--significant-papers-per-cluster`. This lets you keep separate runs side by side when you change that setting.

To resume description generation without repeating finished outputs:

```powershell
python run_descriptions.py --provider chatgpt --only-missing
python run_descriptions.py --scopus-file 001paper.csv --provider chatgpt --only-missing
```

With `--only-missing` (alias: `--skip-existing`):

- an existing clustered file in `clusters/` is reused if it still matches the human target cluster count
- an existing pipeline output is reused if it exists and its cluster ids still match the expected output
- if any required artifact is missing or stale, only that artifact is regenerated

## Typical Flow

1. Run `run_descriptions.py` on a file in `scopus_documents/` to produce the Louvain clustering and the outputs of pipelines 1 to 6.
   A matching file in `description/human_descriptions/` is required. The human file determines the required number of clusters, and `bib_coupling.py` adaptively tunes Louvain `resolution` from a starting value of `1.0` until it matches that exact number.
2. Store human-authored descriptions in `description/human_descriptions/`. If you have a legacy folder, use `import_original_descriptions.py --source-dir <folder>`.
3. Run `run_evaluations.py` for the same paper and provider to compute coverage, induced clustering, silhouette, similarity, and modularity outputs.
4. If you want to run a single pipeline manually, you can still use `pipeline_1.py` to `pipeline_6.py`. These scripts also require the matching human description file, and pipelines 1 to 3 use its cluster count inside the prompt and validation logic. If you omit `--query` and `--prompt`, the script will try to load the query from `queries.xlsx` and the prompt from `prompts.csv`.
   The query is resolved by `paper` number, inferred from names such as `001paper.csv`, unless you pass `--query-id`.
5. You can also run the evaluation scripts individually on either generated descriptions or imported human descriptions.

Example:

```powershell
python bib_coupling.py --scopus-file 001paper.csv
python pipeline_6.py --provider chatgpt --scopus-file 001paper.csv --output-file 001paper.csv
python evaluation_coverage_corpus.py --description-file 001paper_clusters_original_description.csv --scopus-file 001paper.csv
python evaluation_coverage_corpus.py --description-file description/pipeline_6/12/chatgpt/001paper.csv --scopus-file 001paper.csv
```

To point at a specific human-description file:

```powershell
python bib_coupling.py --scopus-file 001paper.csv --human-description-file 001paper_clusters_original_description.csv
```

## File Resolution

Most root scripts accept a bare filename and resolve the correct folder automatically, so you usually do not need to type full project paths.

Examples:

- `001paper.csv` for raw Scopus inputs resolves to `scopus_documents/001paper.csv`
- `001paper.csv` for labeled-cluster inputs resolves to `clusters/001paper_louvain.csv`
- `001paper_clusters_original_description.csv` resolves to `description/human_descriptions/001paper_clusters_original_description.csv`
- an output like `001paper.csv` in `pipeline_6.py` resolves to `description/pipeline_6/<experiment>/<provider>/001paper.csv`

Generated description files may share the same basename across different pipelines, experiments, and providers. For those files, use either the explicit path such as `description/pipeline_6/12/chatgpt/001paper.csv` or a basename that is unique within `description/`.

If you pass a full or explicit relative path, that path is used as-is.

This applies to the main scripts:

- `bib_coupling.py --scopus-file 001paper.csv`
- `pipeline_2.py --scopus-file 001paper.csv --output-file 001paper.csv`
- `pipeline_4.py --scopus-file 001paper.csv --output-file 001paper.csv`
- `pipeline_6.py --scopus-file 001paper.csv --output-file 001paper.csv`
- `evaluation_coverage_corpus.py --description-file 001paper_clusters_original_description.csv --scopus-file 001paper.csv`
- `evaluation_quality_of_the_induced_clustering.py --description-file 001paper_clusters_original_description.csv --scopus-file 001paper.csv`
- `evaluation_silhouette_score.py --scopus-file 001paper.csv`
- `evaluation_modularity.py --scopus-file 001paper.csv`
