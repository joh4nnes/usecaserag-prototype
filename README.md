# UseCaseRAG Streamlit Prototype

Prototype **Generative AI for Test Scenario Generation and Test Plan Recommendation Based on Use Case**.

## Run locally

Python 3.10-3.12 recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `.streamlit/config.toml`, and optional sample dataset.
3. Commit and push.
4. In Streamlit Community Cloud, create a new app.
5. Select the repository, branch, and main file path `app.py`.
6. Deploy.

## Performance design

- Heavy models use lazy loading.
- Default Fast Template Generator avoids large LLM downloads.
- TF-IDF dense retrieval is CPU-friendly.
- BM25/vectorless retrieval requires no vector database.
- Hybrid retrieval uses Reciprocal Rank Fusion.
- ModernBERT loads only after the synthetic-data button is pressed.
- ModernBERT is an encoder/masked-language model; the app uses masked-token augmentation, not causal generation.
- If ModernBERT cannot load, deterministic synthetic augmentation is used as fallback.

## Recommended columns

`use_case_id`, `use_case_name`, `actor`, `precondition`, `main_flow`, `alternative_flow`, `exception_flow`, `postcondition`, `expected_test_scenario`, `expected_test_plan`
