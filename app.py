# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
import time
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG-LLM Test Scenario & Test Plan", page_icon="🧠", layout="wide")

APP_TITLE = "Prototype Model Generative AI for Test Scenario and Recommender Test Plan Based on Use Case"
BACKGROUND_DESCRIPTION = (
    'LLM (Large Language Model) telah digunakan secara efektif di tahap pertengahan hingga akhir '
    'dari siklus hidup pengujian perangkat lunak menurut Paper Survey. Saat ini belom ada Penelitian '
    'RAG-LLM Domain-Specific di Test Plan dan “Test Design and Review”. (Wang et al., 2024). '
    'Software Testing With Large Language Models: Survey, Landscape, and Vision.'
)
RESEARCH_QUESTION_1 = (
    "RQ1. Bagaimana Optimalisasi Kinerja Large Language Model Test Scenario Generation "
    "dan rekomendasi Test Plan?"
)

RESEARCH_QUESTION_2 = (
    "RQ2. Bagaimana Evaluasi hubungan Response terhadap Relevansi Dokumen dari hasil "
    "Generative Retrieval Augmented Large Language Model Test Scenario Generation "
    "dan rekomendasi Test Plan?"
)
RAG_ARCHITECTURES = [
    "Pure LLM",
    "Standard RAG", "Hybrid RAG", "Cross-Encoder RAG", "Iterative RAG", "Graph RAG",
    "Self-Retrieval RAG", "Planning RAG", "Memory RAG", "CodeRAG", "Multimodal RAG"
]
LLM_OPTIONS = [
    "Fast Template Generator", "google/flan-t5-small", "google/flan-t5-base",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-0.5B-Instruct"
]
CHUNKING_OPTIONS = ["fixed-size", "recursive", "semantic", "use case structure-aware"]
RETRIEVAL_OPTIONS = ["dense retrieval", "BM25/vectorless", "hybrid", "graph", "reranker"]
REQUIRED_COLUMNS = [
    "use_case_id", "use_case_name", "actor", "precondition", "main_flow",
    "alternative_flow", "exception_flow", "postcondition",
    "expected_test_scenario", "expected_test_plan"
]
STOPWORDS = set("""
a an and are as at be by for from has have in into is it its of on or that the this to was were will with
yang dan di ke dari untuk pada adalah dengan ini itu dalam sebagai atau bisa dapat akan tidak
llm rag user system use case test scenario plan
""".split())


def sample_dataset() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "use_case_id": "UC-001", "use_case_name": "User Login", "actor": "Registered User",
            "precondition": "User has an active account and the login page is available.",
            "main_flow": "User enters valid username and password. System validates credentials. System redirects user to dashboard.",
            "alternative_flow": "User enters invalid password. System displays invalid credential message and allows retry.",
            "exception_flow": "Authentication service is unavailable. System displays service unavailable message and logs the failure.",
            "postcondition": "User is authenticated and a session is created.",
            "expected_test_scenario": "Scenario: Login with valid credentials. Precondition: active account. Steps: enter valid username and password and submit login. Expected result: dashboard is displayed. Negative scenario: invalid password displays an error. Exception scenario: authentication service unavailable is handled.",
            "expected_test_plan": "Objective: verify login. Scope: valid login, invalid login, and authentication service failure. Test type: functional, negative, integration. Environment: web browser and test server. Risk: authentication and session failure. Deliverables: scenarios, cases, and report."
        },
        {
            "use_case_id": "UC-002", "use_case_name": "Cancel Order", "actor": "Customer",
            "precondition": "Customer has an order with pending or processing status.",
            "main_flow": "Customer opens order detail, selects cancel order, and confirms. System validates status, cancels the order, and sends notification.",
            "alternative_flow": "Order has already been shipped. System rejects cancellation and displays reason.",
            "exception_flow": "Refund service fails. System marks cancellation as pending refund and records retry.",
            "postcondition": "Order is cancelled or cancellation is rejected.",
            "expected_test_scenario": "Scenario: Cancel pending order. Precondition: order pending. Steps: open detail, cancel, confirm. Expected result: order cancelled and notification sent. Negative: shipped order cannot be cancelled. Exception: refund failure is queued.",
            "expected_test_plan": "Objective: verify order cancellation. Scope: pending cancellation, shipped order rejection, refund failure. Test type: functional, negative, integration. Environment: order and refund services. Risk: status inconsistency and refund failure."
        },
        {
            "use_case_id": "UC-003", "use_case_name": "Generate Sales Report", "actor": "Admin",
            "precondition": "Admin is logged in and sales data exists for selected period.",
            "main_flow": "Admin opens report menu, selects date range, and requests report. System retrieves sales data and displays summary.",
            "alternative_flow": "No sales data exists. System displays an empty report message.",
            "exception_flow": "Database query times out. System displays timeout message and allows retry.",
            "postcondition": "Report is displayed or failure is reported.",
            "expected_test_scenario": "Scenario: Generate report for valid period. Precondition: admin logged in. Steps: select date range and generate. Expected result: report summary displayed. Negative: no data shows empty report. Exception: query timeout allows retry.",
            "expected_test_plan": "Objective: verify sales reporting. Scope: valid report, no data, timeout. Test type: functional, boundary, reliability. Environment: reporting module and database. Risk: inaccurate report and timeout."
        }
    ])


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(normalize_text)
    return out


def load_uploaded_dataset(uploaded_file: Any) -> pd.DataFrame:
    if uploaded_file is None:
        return sample_dataset()
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Format file tidak didukung")


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return not missing, missing


@st.cache_resource(show_spinner=False)
def load_modernbert_fill_mask():
    from transformers import pipeline
    return pipeline("fill-mask", model="answerdotai/ModernBERT-base", device=-1)


def mask_candidate_sentence(text: str) -> str | None:
    words = normalize_text(text).split()
    candidates = [i for i, w in enumerate(words) if len(re.sub(r"[^A-Za-z]", "", w)) >= 5]
    if not candidates:
        return None
    words[candidates[len(candidates) // 2]] = "[MASK]"
    return " ".join(words)


def modernbert_augment_text(text: str) -> str:
    masked = mask_candidate_sentence(text)
    if not masked:
        return normalize_text(text)
    model = load_modernbert_fill_mask()
    out = model(masked, top_k=3)
    return normalize_text(out[0].get("sequence", text)) if out else normalize_text(text)


def fallback_synthetic_dataset(df: pd.DataFrame, variants_per_case: int) -> pd.DataFrame:
    variants = [
        ("Boundary", "boundary input is provided and limits must be validated"),
        ("Negative", "invalid or incomplete data is provided and must be rejected"),
        ("Exception", "an external dependency fails and consistency must be preserved")
    ]
    rows = []
    for _, row in df.iterrows():
        for i in range(variants_per_case):
            label, condition = variants[i % len(variants)]
            new = row.copy()
            ucid = normalize_text(row.get("use_case_id", "UC"))
            name = normalize_text(row.get("use_case_name", "Use Case"))
            new["use_case_id"] = f"SYN-{ucid}-{i+1}"
            new["use_case_name"] = f"{name} - Synthetic {label}"
            new["alternative_flow"] = f"{normalize_text(row.get('alternative_flow',''))} Synthetic {label} condition: {condition}."
            new["exception_flow"] = f"{normalize_text(row.get('exception_flow',''))} Synthetic exception check for {name}."
            new["expected_test_scenario"] = f"Synthetic {label} scenario for {name}. Validate that {condition}. Expected result: system responds according to requirement."
            new["expected_test_plan"] = f"Objective: verify {label.lower()} behavior for {name}. Scope: main, alternative, exception flows. Test type: functional, negative, boundary, reliability. Risk: incomplete coverage."
            rows.append(new)
    return pd.DataFrame(rows)


def generate_synthetic_dataset_modernbert(df: pd.DataFrame, variants_per_case: int) -> Tuple[pd.DataFrame, str]:
    try:
        rows = []
        for _, row in df.iterrows():
            base_id = normalize_text(row.get("use_case_id", "UC"))
            base_name = normalize_text(row.get("use_case_name", "Use Case"))
            for i in range(variants_per_case):
                new = row.copy()
                new["use_case_id"] = f"SYN-{base_id}-{i+1}"
                new["use_case_name"] = f"{base_name} - ModernBERT Variant {i+1}"
                for col in ["main_flow", "alternative_flow", "exception_flow"]:
                    if normalize_text(row.get(col, "")):
                        new[col] = modernbert_augment_text(row[col])
                new["expected_test_scenario"] = f"ModernBERT-assisted synthetic scenario for {base_name}. Main: {new.get('main_flow','')}. Alternative: {new.get('alternative_flow','')}. Exception: {new.get('exception_flow','')}."
                new["expected_test_plan"] = f"Objective: verify ModernBERT-assisted variant of {base_name}. Scope: main, alternative, exception flows. Risk: synthetic drift and incomplete grounding."
                rows.append(new)
        return pd.DataFrame(rows), "ModernBERT masked-language augmentation"
    except Exception as exc:
        return fallback_synthetic_dataset(df, variants_per_case), f"Fallback deterministic augmentation ({type(exc).__name__})"


def row_to_document(row: pd.Series) -> str:
    fields = [
        ("Use Case ID", row.get("use_case_id", "")), ("Use Case Name", row.get("use_case_name", "")),
        ("Actor", row.get("actor", "")), ("Precondition", row.get("precondition", "")),
        ("Main Flow", row.get("main_flow", "")), ("Alternative Flow", row.get("alternative_flow", "")),
        ("Exception Flow", row.get("exception_flow", "")), ("Postcondition", row.get("postcondition", ""))
    ]
    return "\n".join(f"{k}: {normalize_text(v)}" for k, v in fields if normalize_text(v))


def chunk_record(chunk_id: str, row: pd.Series, section: str, scenario_type: str, content: str, strategy: str) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id, "use_case_id": normalize_text(row.get("use_case_id", "UC")),
        "use_case_name": normalize_text(row.get("use_case_name", "Use Case")), "section": section,
        "scenario_type": scenario_type, "content": normalize_text(content), "chunking_strategy": strategy
    }


def fixed_size_chunking(df: pd.DataFrame, chunk_size: int, overlap: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        doc = row_to_document(row); ucid = normalize_text(row.get("use_case_id", "UC")); step = max(1, chunk_size-overlap)
        for idx, start in enumerate(range(0, len(doc), step), start=1):
            piece = doc[start:start+chunk_size]
            if normalize_text(piece):
                rows.append(chunk_record(f"{ucid}-FIX-{idx}", row, "fixed_size", "general", piece, "fixed-size"))
    return pd.DataFrame(rows)


def recursive_chunking(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        ucid = normalize_text(row.get("use_case_id", "UC")); parts = [p.strip() for p in row_to_document(row).split("\n") if p.strip()]
        buf = ""; idx = 1
        for part in parts:
            candidate = (buf + "\n" + part).strip()
            if len(candidate) <= chunk_size: buf = candidate
            else:
                if buf:
                    rows.append(chunk_record(f"{ucid}-REC-{idx}", row, "recursive", "general", buf, "recursive")); idx += 1
                buf = part
        if buf: rows.append(chunk_record(f"{ucid}-REC-{idx}", row, "recursive", "general", buf, "recursive"))
    return pd.DataFrame(rows)


def semantic_chunking(df: pd.DataFrame) -> pd.DataFrame:
    rows = []; groups = [
        ("identity_context", ["use_case_name", "actor", "precondition"], "general"),
        ("normal_behavior", ["main_flow", "postcondition"], "positive"),
        ("negative_exception_behavior", ["alternative_flow", "exception_flow"], "negative_exception")
    ]
    for _, row in df.iterrows():
        ucid = normalize_text(row.get("use_case_id", "UC"))
        for section, cols, scenario_type in groups:
            content = "\n".join(f"{c.replace('_',' ').title()}: {normalize_text(row.get(c,''))}" for c in cols if normalize_text(row.get(c,"")))
            if content: rows.append(chunk_record(f"{ucid}-SEM-{section}", row, section, scenario_type, content, "semantic"))
    return pd.DataFrame(rows)


def structure_aware_chunking(df: pd.DataFrame) -> pd.DataFrame:
    rows = []; mapping = [
        ("use_case_name", "use_case_identity", "general"), ("actor", "actor", "general"),
        ("precondition", "precondition", "general"), ("main_flow", "main_flow", "positive"),
        ("alternative_flow", "alternative_flow", "negative_alternative"),
        ("exception_flow", "exception_flow", "exception"), ("postcondition", "postcondition", "general")
    ]
    for _, row in df.iterrows():
        ucid = normalize_text(row.get("use_case_id", "UC"))
        for col, section, scenario_type in mapping:
            value = normalize_text(row.get(col, ""))
            if value: rows.append(chunk_record(f"{ucid}-STRUCT-{section}", row, section, scenario_type, f"{section.replace('_',' ').title()}: {value}", "use case structure-aware"))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_chunks_cached(data_json: str, strategy: str, chunk_size: int, overlap: int) -> pd.DataFrame:
    df = pd.read_json(data_json)
    if strategy == "fixed-size": return fixed_size_chunking(df, chunk_size, overlap)
    if strategy == "recursive": return recursive_chunking(df, chunk_size)
    if strategy == "semantic": return semantic_chunking(df)
    return structure_aware_chunking(df)


def tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^A-Za-z0-9_ ]", " ", normalize_text(text).lower())
    return [t for t in cleaned.split() if t and t not in STOPWORDS]


class SimpleBM25:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = [list(d) for d in corpus_tokens]; self.k1 = k1; self.b = b; self.n_docs = len(self.corpus)
        self.avgdl = float(np.mean([len(d) for d in self.corpus])) if self.corpus else 0.0
        self.term_freqs = [Counter(d) for d in self.corpus]; self.doc_freq = Counter()
        for d in self.corpus: self.doc_freq.update(set(d))
    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0); return math.log((self.n_docs-df+0.5)/(df+0.5)+1.0)
    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        out = []
        for idx, doc in enumerate(self.corpus):
            dl = len(doc); tf = self.term_freqs[idx]; score = 0.0
            for term in query_tokens:
                freq = tf.get(term, 0)
                if not freq: continue
                denom = freq + self.k1*(1-self.b+self.b*dl/(self.avgdl+1e-9))
                score += self.idf(term)*(freq*(self.k1+1))/(denom+1e-9)
            out.append(score)
        return np.asarray(out)


@st.cache_resource(show_spinner=False)
def build_index(chunks_json: str):
    chunks_df = pd.read_json(chunks_json); texts = chunks_df["content"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None); matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, SimpleBM25([tokenize(t) for t in texts])


def dense_retrieve(query: str, chunks_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    vectorizer, matrix, _ = build_index(chunks_df.to_json()); q = vectorizer.transform([query]); scores = cosine_similarity(q, matrix).ravel()
    out = chunks_df.copy(); out["score"] = scores; out["retrieval_method"] = "dense retrieval"
    return out.sort_values("score", ascending=False).head(top_k)


def bm25_retrieve(query: str, chunks_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    _, _, bm25 = build_index(chunks_df.to_json()); scores = bm25.scores(tokenize(query))
    out = chunks_df.copy(); out["score"] = scores; out["retrieval_method"] = "BM25/vectorless"
    return out.sort_values("score", ascending=False).head(top_k)


def rrf_fusion(dense_df: pd.DataFrame, bm25_df: pd.DataFrame, chunks_df: pd.DataFrame, top_k: int, k: int = 60) -> pd.DataFrame:
    score_map = {}; method_map = {}
    for rank, row in enumerate(dense_df.itertuples(), start=1):
        score_map[row.chunk_id] = score_map.get(row.chunk_id, 0.0)+1/(k+rank); method_map[row.chunk_id] = method_map.get(row.chunk_id, "")+"|dense"
    for rank, row in enumerate(bm25_df.itertuples(), start=1):
        score_map[row.chunk_id] = score_map.get(row.chunk_id, 0.0)+1/(k+rank); method_map[row.chunk_id] = method_map.get(row.chunk_id, "")+"|bm25"
    rows = []
    for cid, score in score_map.items():
        rec = chunks_df[chunks_df["chunk_id"] == cid].iloc[0].to_dict(); rec["score"] = score; rec["retrieval_method"] = "hybrid"+method_map[cid]; rows.append(rec)
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)


def architecture_boost(result: pd.DataFrame, query: str, architecture: str) -> pd.DataFrame:
    if result.empty: return result
    q = query.lower(); arch = architecture.lower(); out = result.copy(); boosts = []
    for _, row in out.iterrows():
        section = str(row.get("section", "")).lower(); scenario = str(row.get("scenario_type", "")).lower(); b = 0.0
        if any(w in q for w in ["negative", "invalid", "error", "exception"]):
            if any(w in section for w in ["alternative", "exception"]) or "negative" in scenario: b += 0.18
        if "graph" in arch and section in {"actor","precondition","main_flow","alternative_flow","exception_flow","postcondition"}: b += 0.08
        if "planning" in arch and section in {"precondition","main_flow","alternative_flow","exception_flow","postcondition"}: b += 0.10
        if "self-retrieval" in arch and float(row.get("score", 0)) > 0: b += 0.05
        boosts.append(b)
    out["score"] = out["score"].astype(float)+np.asarray(boosts)
    return out.sort_values("score", ascending=False)



def boost_exact_use_case_id(
    result: pd.DataFrame,
    chunks_df: pd.DataFrame,
    query: str,
) -> pd.DataFrame:
    """
    Metadata-aware boost for explicit IDs such as UC-001 in the query.

    Structure-aware chunks may not repeat use_case_id inside chunk content,
    so lexical/dense retrieval alone can miss an exact ID query. This helper
    injects matching chunks from metadata and prioritizes testing-relevant
    sections.
    """
    if chunks_df is None or chunks_df.empty or "use_case_id" not in chunks_df.columns:
        return result

    query_upper = normalize_text(query).upper()
    available_ids = [
        normalize_text(value)
        for value in chunks_df["use_case_id"].dropna().astype(str).unique().tolist()
        if normalize_text(value)
    ]
    target_ids = [
        use_case_id
        for use_case_id in available_ids
        if use_case_id.upper() in query_upper
    ]

    if not target_ids:
        return result

    exact = chunks_df[
        chunks_df["use_case_id"].astype(str).str.upper().isin(
            [target.upper() for target in target_ids]
        )
    ].copy()

    if exact.empty:
        return result

    current_max = 0.0
    if result is not None and not result.empty and "score" in result.columns:
        numeric_scores = pd.to_numeric(result["score"], errors="coerce").fillna(0.0)
        current_max = float(numeric_scores.max())

    section_priority = {
        "main_flow": 0.60,
        "alternative_flow": 0.55,
        "exception_flow": 0.50,
        "precondition": 0.45,
        "postcondition": 0.40,
        "actor": 0.35,
        "use_case_identity": 0.30,
        "fixed_size": 0.25,
        "recursive": 0.25,
        "normal_behavior": 0.55,
        "negative_exception_behavior": 0.50,
        "identity_context": 0.35,
    }

    exact["score"] = [
        current_max + 1.0 + section_priority.get(str(section).lower(), 0.20)
        for section in exact.get("section", pd.Series([""] * len(exact)))
    ]
    exact["retrieval_method"] = "metadata_exact_use_case_id"

    if result is None or result.empty:
        combined = exact
    else:
        combined = pd.concat([exact, result], ignore_index=True, sort=False)

    # Keep the highest-scoring occurrence of each chunk.
    if "chunk_id" in combined.columns:
        combined = (
            combined.sort_values("score", ascending=False)
            .drop_duplicates(subset=["chunk_id"], keep="first")
        )
    else:
        combined = combined.sort_values("score", ascending=False)

    return combined.reset_index(drop=True)


def retrieve_context(query: str, chunks_df: pd.DataFrame, method: str, top_k: int, architecture: str) -> pd.DataFrame:
    # Pure LLM baseline deliberately bypasses retrieval augmentation.
    if architecture == "Pure LLM":
        base_columns = list(chunks_df.columns)
        for extra_col in ["score", "retrieval_method"]:
            if extra_col not in base_columns:
                base_columns.append(extra_col)
        return pd.DataFrame(columns=base_columns)

    if chunks_df is None or chunks_df.empty:
        return pd.DataFrame(
            columns=["chunk_id", "use_case_id", "use_case_name", "section",
                     "scenario_type", "content", "chunking_strategy",
                     "score", "retrieval_method"]
        )

    pool = max(8, top_k*3)

    if method == "dense retrieval":
        result = dense_retrieve(query, chunks_df, pool)
    elif method == "BM25/vectorless":
        result = bm25_retrieve(query, chunks_df, pool)
    elif method == "hybrid":
        result = rrf_fusion(
            dense_retrieve(query, chunks_df, pool),
            bm25_retrieve(query, chunks_df, pool),
            chunks_df,
            pool
        )
    elif method == "graph":
        result = architecture_boost(
            bm25_retrieve(query, chunks_df, pool),
            query,
            "Graph RAG"
        )
    else:
        result = architecture_boost(
            rrf_fusion(
                dense_retrieve(query, chunks_df, pool),
                bm25_retrieve(query, chunks_df, pool),
                chunks_df,
                pool
            ),
            query,
            "Cross-Encoder RAG"
        )

    result = architecture_boost(result, query, architecture)
    result = boost_exact_use_case_id(result, chunks_df, query)

    return result.head(top_k).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_hf_generator(model_name: str):
    from transformers import pipeline
    task = "text2text-generation" if "flan-t5" in model_name else "text-generation"
    return pipeline(task, model=model_name, device=-1)


def run_open_source_llm(prompt: str, model_name: str, max_new_tokens: int, temperature: float) -> str | None:
    if model_name == "Fast Template Generator": return None
    try:
        generator = load_hf_generator(model_name); kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        kwargs.update({"do_sample": temperature > 0});
        if temperature > 0: kwargs["temperature"] = temperature
        out = generator(prompt, **kwargs)
        if not out: return None
        return normalize_text(out[0].get("generated_text", str(out[0])))
    except Exception as exc:
        st.warning(f"Model {model_name} gagal dijalankan ({type(exc).__name__}); fallback ke Fast Template Generator.")
        return None


def context_text(retrieved: pd.DataFrame) -> str:
    return "\n".join(retrieved["content"].fillna("").astype(str).tolist()) if retrieved is not None and not retrieved.empty else ""


def scenario_template(row: pd.Series, architecture: str, llm_name: str) -> str:
    ucid = normalize_text(row.get("use_case_id","UC")); name = normalize_text(row.get("use_case_name","Use Case"))
    lines = [
        f"Test Scenario ID: TS-{ucid}", f"Use Case: {name}", f"Architecture: {architecture}", f"LLM: {llm_name}",
        f"Actor: {normalize_text(row.get('actor','Actor'))}", f"Objective: Validate {name} according to the use case.",
        f"Precondition: {normalize_text(row.get('precondition',''))}", "", "A. Positive / Main Scenario",
        f"Steps: {normalize_text(row.get('main_flow',''))}", f"Expected Result: {normalize_text(row.get('postcondition',''))}", "",
        "B. Negative / Alternative Scenario", f"Steps: {normalize_text(row.get('alternative_flow',''))}",
        "Expected Result: Invalid condition is rejected and a correct message is displayed.", "", "C. Exception Scenario",
        f"Steps: {normalize_text(row.get('exception_flow',''))}", "Expected Result: Failure is handled safely and data consistency is preserved.", "",
        "D. Traceability", f"{ucid} -> Actor -> Precondition -> Main Flow -> Alternative Flow -> Exception Flow -> Postcondition"
    ]
    if architecture in {"Iterative RAG","Self-Retrieval RAG","Planning RAG"}: lines.append("Refinement: Positive, negative, and exception coverage are checked.")
    if architecture == "Graph RAG": lines.append("Graph Trace: Actor and flow relations are preserved.")
    if architecture == "Memory RAG": lines.append("Memory: Reusable testing template is applied.")
    return "\n".join(lines)


def generate_test_scenario(row: pd.Series, architecture: str, retrieved: pd.DataFrame, llm_name: str, max_new_tokens: int, temperature: float) -> str:
    if architecture == "Pure LLM":
        prompt = (
            "You are a software testing expert. Generate structured positive, negative, alternative, "
            "and exception test scenarios directly from the raw use case input. "
            "Do not use retrieval augmentation.\n"
            f"Architecture: {architecture}\n"
            f"Raw Use Case Input:\n{row_to_document(row)}"
        )
    else:
        prompt = (
            "You are a software testing expert. Generate structured positive, negative, alternative, "
            "and exception test scenarios using only retrieved context.\n"
            f"Architecture: {architecture}\n"
            f"Context:\n{context_text(retrieved)}"
        )
    return run_open_source_llm(prompt, llm_name, max_new_tokens, temperature) or scenario_template(row, architecture, llm_name)


def test_plan_template(row: pd.Series, architecture: str, llm_name: str) -> str:
    ucid = normalize_text(row.get("use_case_id","UC")); name = normalize_text(row.get("use_case_name","Use Case"))
    lines = [
        f"Test Plan ID: TP-{ucid}", f"Title: Test Plan Recommendation for {name}", f"Architecture: {architecture}", f"LLM: {llm_name}", "",
        "1. Test Objective", f"Verify functional correctness, negative behavior, and exception handling of {name}.", "",
        "2. Test Scope", f"Actor {normalize_text(row.get('actor',''))}; precondition; main flow; alternative flow; exception flow.", "",
        "3. Test Items", f"- Use Case: {name}", f"- Precondition: {normalize_text(row.get('precondition',''))}", "",
        "4. Features to be Tested", f"- Main Flow: {normalize_text(row.get('main_flow',''))}", f"- Alternative Flow: {normalize_text(row.get('alternative_flow',''))}", f"- Exception Flow: {normalize_text(row.get('exception_flow',''))}", "",
        "5. Test Approach", "Requirement-based testing using RAG-LLM generated scenarios and retrieved use case context.", "",
        "6. Test Type", "Functional, negative, boundary, integration, and exception testing.", "",
        "7. Test Environment", "Streamlit prototype, local/Streamlit Cloud runtime, dataset, retrieval index, selected LLM.", "",
        "8. Entry Criteria", "Dataset is available, preprocessed, chunked, indexed, and selected architecture is configured.", "",
        "9. Exit Criteria", "Main, alternative, and exception flows are covered and metrics reach an acceptable threshold.", "",
        "10. Risk", "Incomplete context, hallucination, missing exception coverage, inconsistent recommendation.", "",
        "11. Deliverables", "Generated scenarios, recommended test plan, retrieved context, metrics, charts, comparison report."
    ]
    if architecture == "Planning RAG": lines.append("Planning Note: Objective, scope, risk, entry, exit, and deliverables are explicitly organized.")
    return "\n".join(lines)


def generate_test_plan(row: pd.Series, architecture: str, retrieved: pd.DataFrame, scenario_output: str, llm_name: str, max_new_tokens: int, temperature: float) -> str:
    if architecture == "Pure LLM":
        prompt = (
            "You are a software test manager. Generate a structured test plan recommendation directly "
            "from the raw use case and generated scenario. Do not use retrieval augmentation.\n"
            f"Architecture: {architecture}\n"
            f"Raw Use Case Input:\n{row_to_document(row)}\n"
            f"Scenario:\n{scenario_output}"
        )
    else:
        prompt = (
            "You are a software test manager. Generate a structured test plan using only retrieved "
            "context and scenario.\n"
            f"Architecture: {architecture}\n"
            f"Context:\n{context_text(retrieved)}\n"
            f"Scenario:\n{scenario_output}"
        )
    return run_open_source_llm(prompt, llm_name, max_new_tokens, temperature) or test_plan_template(row, architecture, llm_name)


def recall_at_3(retrieved: pd.DataFrame, target: str) -> float:
    return float(retrieved is not None and not retrieved.empty and any(retrieved.head(3)["use_case_id"].astype(str) == str(target)))


def mrr(retrieved: pd.DataFrame, target: str) -> float:
    if retrieved is None or retrieved.empty: return 0.0
    for rank, row in enumerate(retrieved.itertuples(), start=1):
        if str(row.use_case_id) == str(target): return 1.0/rank
    return 0.0


def semantic_similarity(pred: str, ref: str) -> float:
    if not normalize_text(pred) or not normalize_text(ref): return 0.0
    v = TfidfVectorizer(tokenizer=tokenize, token_pattern=None); m = v.fit_transform([pred, ref]); return float(cosine_similarity(m[0], m[1])[0,0])


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    dp = [0]*(len(b)+1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            temp = dp[j]; dp[j] = prev+1 if x == y else max(dp[j], dp[j-1]); prev = temp
    return dp[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r: return 0.0
    l = lcs_length(p,r); precision = l/len(p); recall = l/len(r)
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.0


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str,...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def bleu(pred: str, ref: str, max_n: int = 2) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r: return 0.0
    precisions = []
    for n in range(1,max_n+1):
        pg, rg = ngrams(p,n), ngrams(r,n)
        if not pg: precisions.append(0.0); continue
        counts = Counter(rg); matches = 0
        for g in pg:
            if counts[g] > 0: matches += 1; counts[g] -= 1
        precisions.append((matches+1)/(len(pg)+1))
    geo = math.exp(np.mean([math.log(max(x,1e-12)) for x in precisions])); bp = 1.0 if len(p) >= len(r) else math.exp(1-len(r)/max(len(p),1))
    return float(bp*geo)


def coverage(output: str, row: pd.Series, task: str) -> float:
    output_l = normalize_text(output).lower()
    items = {
        "actor": row.get("actor",""), "precondition": row.get("precondition",""), "main": row.get("main_flow",""),
        "alternative": row.get("alternative_flow",""), "exception": row.get("exception_flow",""), "post": row.get("postcondition","")
    } if task == "scenario" else {
        "objective":"objective", "scope":"scope", "test_type":"functional negative integration exception", "environment":"environment",
        "entry":"entry criteria", "exit":"exit criteria", "risk":"risk", "deliverables":"deliverables"
    }
    total = 0; covered = 0
    for val in items.values():
        terms = tokenize(str(val))
        if not terms: continue
        total += 1; hits = sum(1 for t in terms[:14] if t in output_l)
        if hits >= max(1,min(2,len(terms))): covered += 1
    return covered/total if total else 0.0


def faithfulness(output: str, context: str) -> float:
    o = {t for t in tokenize(output) if len(t)>2}; c = set(tokenize(context))
    return len(o & c)/len(o) if o else 0.0


def calculate_metrics(output: str, reference: str, retrieved: pd.DataFrame, target: str, row: pd.Series, task: str) -> Dict[str,float]:
    return {
        "Recall@3": recall_at_3(retrieved,target), "MRR": mrr(retrieved,target),
        "Semantic Similarity": semantic_similarity(output,reference), "ROUGE-L": rouge_l(output,reference),
        "BLEU": bleu(output,reference), "Coverage": coverage(output,row,task), "Faithfulness": faithfulness(output,context_text(retrieved))
    }


def overall_score(metrics: Dict[str,float]) -> float:
    weights = {"Recall@3":0.15,"MRR":0.15,"Semantic Similarity":0.20,"ROUGE-L":0.10,"BLEU":0.05,"Coverage":0.20,"Faithfulness":0.15}
    return sum(metrics[k]*w for k,w in weights.items())


# Session state
for key, value in {
    "original_dataset": sample_dataset(), "synthetic_dataset": pd.DataFrame(), "selected_architectures": RAG_ARCHITECTURES.copy(),
    "last_scenario": "", "last_plan": "", "last_retrieved_scenario": pd.DataFrame(), "last_retrieved_plan": pd.DataFrame()
}.items():
    if key not in st.session_state: st.session_state[key] = value

st.sidebar.title("📌 Menu Prototype")
menu = st.sidebar.radio("Pilih Menu", [
    "Background",
    "Problem Statement",
    "Use Case Dataset",
    "Preprocessing",
    "Chunking Strategy",
    "Retrieval Configuration",
    "RAG Architecture Selection",
    "LLM Configuration",
    "Test Scenario Generation",
    "Test Plan Recommendation",
    "Evaluation Metric",
    "Comparison",
    "About"
])
st.sidebar.divider(); st.sidebar.subheader("⚙️ Global Configuration")
uploaded = st.sidebar.file_uploader("Upload Original Dataset", type=["csv","xlsx","xls"])
if uploaded is not None:
    try: st.session_state.original_dataset = clean_dataframe(load_uploaded_dataset(uploaded))
    except Exception as exc: st.sidebar.error(f"Gagal membaca dataset: {exc}")

dataset_mode = st.sidebar.selectbox("Dataset Aktif", ["Original Dataset","Synthetic Dataset","Original + Synthetic"], key="sidebar_dataset_mode")
if dataset_mode == "Synthetic Dataset" and not st.session_state.synthetic_dataset.empty: active_df = st.session_state.synthetic_dataset.copy()
elif dataset_mode == "Original + Synthetic" and not st.session_state.synthetic_dataset.empty: active_df = pd.concat([st.session_state.original_dataset, st.session_state.synthetic_dataset], ignore_index=True)
else: active_df = st.session_state.original_dataset.copy()
active_df = clean_dataframe(active_df)
chunking_strategy = st.sidebar.selectbox("Chunking Strategy", CHUNKING_OPTIONS, index=3, key="sidebar_chunking_strategy")
retrieval_method = st.sidebar.selectbox("Retrieval Method", RETRIEVAL_OPTIONS, index=2, key="sidebar_retrieval_method")
generation_architecture = st.sidebar.selectbox(
    "Architecture untuk Generation",
    RAG_ARCHITECTURES,
    index=RAG_ARCHITECTURES.index("Standard RAG"),
    key="sidebar_generation_architecture"
)
llm_name = st.sidebar.selectbox("Open Source LLM", LLM_OPTIONS, key="sidebar_open_source_llm")
top_k = st.sidebar.slider("Top-K Retrieval",1,10,3, key="sidebar_top_k")
chunk_size = st.sidebar.slider("Chunk Size",300,1600,800,100, key="sidebar_chunk_size")
chunk_overlap = st.sidebar.slider("Chunk Overlap",0,300,100,50, key="sidebar_chunk_overlap")
llm_temperature = st.sidebar.slider("LLM Temperature",0.0,1.0,0.2,0.1, key="sidebar_llm_temperature")
llm_max_new_tokens = st.sidebar.slider("LLM Max New Tokens",128,1024,384,64, key="sidebar_llm_max_tokens")
use_case_ids = active_df["use_case_id"].astype(str).tolist() if "use_case_id" in active_df.columns and not active_df.empty else ["UC-001"]
selected_use_case_id = st.sidebar.selectbox("Selected Use Case", use_case_ids, key="sidebar_selected_use_case")
chunks_df = build_chunks_cached(active_df.to_json(), chunking_strategy, chunk_size, chunk_overlap)

st.markdown(f"<div style='background:#d9e2f3;padding:18px;border:1px solid #222;text-align:center'><h1>{APP_TITLE}</h1><h3 style='color:#365f91'>RAG-LLM Test Scenario Generation and Test Plan Recommendation</h3></div>", unsafe_allow_html=True)
st.write("")

if menu == "Background":
    st.header("Background")
    st.subheader("Description")
    st.info(BACKGROUND_DESCRIPTION)

    background_image_path = Path(__file__).parent /  "llm_testing_task_distribution.png"
    if background_image_path.exists():
        image_caption = (
            " "
        )

        # Cross-version Streamlit compatibility:
        # - Newer versions support use_container_width for st.image.
        # - Older versions use use_column_width.
        try:
            st.image(
                str(background_image_path),
                caption=image_caption,
                use_container_width=True,
            )
        except TypeError:
            st.image(
                str(background_image_path),
                caption=image_caption,
                use_column_width=True,
            )

    st.subheader("RAG-LLM Pipeline")
    st.code(
        "Use Case Dataset -> Preprocessing -> Chunking -> Retrieval -> RAG Architecture -> "
        "Open Source LLM -> Test Scenario Generation -> Test Plan Recommendation -> "
        "Evaluation Metric -> Comparison",
        language="text"
    )
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Use Cases",len(active_df))
    c2.metric("Chunks",len(chunks_df))
    c3.metric("Chunking",chunking_strategy)
    c4.metric("Retrieval",retrieval_method)

elif menu == "About":
    st.header("About")
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f3f6fa, #e8eef7);
            padding: 28px;
            border-radius: 14px;
            border: 1px solid #c7d2e3;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        ">
            <h3 style="margin-top:0; color:#1f3b5b;">
                Berikut Prototype Hasil Experiment dari Penelitian Disertasi Doktoral Ilmu Komputer
            </h3>
            <p style="font-size:1.15rem; line-height:1.7; margin-bottom:0;">
                <strong>Model Generative AI untuk Test Scenario dan Rekomendasi Test Plan berdasarkan Use Case</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.info(
        "Prototype ini digunakan untuk mendemonstrasikan pipeline eksperimen RAG-LLM, "
        "Test Scenario Generation, Test Plan Recommendation, evaluasi metric, serta "
        "perbandingan Pure LLM baseline dengan berbagai arsitektur RAG."
    )

elif menu == "Problem Statement":
    st.header("Problem Statement")
    st.subheader("Research Questions")

    st.markdown(
        f"""
        <div style="
            background:#f8fafc;
            padding:20px;
            border-radius:12px;
            border-left:5px solid #365F91;
            margin-bottom:14px;
        ">
            <div style="font-size:1.05rem; line-height:1.7;">
                <strong>RQ1.</strong> Bagaimana Optimalisasi Kinerja Large Language Model
                Test Scenario Generation dan rekomendasi Test Plan?
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            background:#f8fafc;
            padding:20px;
            border-radius:12px;
            border-left:5px solid #6B8E23;
            margin-bottom:14px;
        ">
            <div style="font-size:1.05rem; line-height:1.7;">
                <strong>RQ2.</strong> Bagaimana Evaluasi hubungan Response terhadap Relevansi Dokumen
                dari hasil Generative Retrieval Augmented Large Language Model
                Test Scenario Generation dan rekomendasi Test Plan?
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Research Question Summary")
    rq_df = pd.DataFrame(
        [
            {
                "Research Question": "RQ1",
                "Focus": "Optimalisasi Kinerja",
                "Description": RESEARCH_QUESTION_1.replace("RQ1. ", "")
            },
            {
                "Research Question": "RQ2",
                "Focus": "Evaluasi Response terhadap Relevansi Dokumen",
                "Description": RESEARCH_QUESTION_2.replace("RQ2. ", "")
            },
        ]
    )
    st.dataframe(rq_df, use_container_width=True, hide_index=True)

elif menu == "Use Case Dataset":
    st.header("Use Case Dataset"); original_tab, synthetic_tab = st.tabs(["Original Dataset","Synthetic Dataset"])
    with original_tab:
        st.subheader("Upload Data"); st.write("Gunakan uploader pada sidebar untuk CSV/XLSX.")
        valid, missing = validate_dataset(st.session_state.original_dataset)
        if valid:
            st.success("Struktur kolom dataset lengkap.")
        else:
            st.warning("Kolom yang belum tersedia: " + ", ".join(missing))
        st.subheader("Data Description"); d1,d2 = st.columns(2); d1.metric("Rows",len(st.session_state.original_dataset)); d2.metric("Columns",len(st.session_state.original_dataset.columns)); st.write("Columns:",list(st.session_state.original_dataset.columns))
        st.subheader("Tabular Data"); st.dataframe(st.session_state.original_dataset,use_container_width=True)
    with synthetic_tab:
        st.subheader("Generate Synthetic Dataset menggunakan ModernBERT")
        st.caption("ModernBERT adalah encoder/masked-language model. Prototype memakai masked-token augmentation, bukan causal text generation.")
        variants = st.slider("Synthetic variants per use case",1,4,2, key="synthetic_variants_slider")
        if st.button("Generate Synthetic Dataset menggunakan ModernBERT",type="primary", key="btn_generate_synthetic_modernbert"):
            with st.status("Synthetic dataset generation...",expanded=True) as status:
                st.write("Step 1: Membaca original dataset."); st.write("Step 2: Lazy-load ModernBERT masked-language model."); st.write("Step 3: Membuat masked-token variants.")
                synthetic, mode = generate_synthetic_dataset_modernbert(st.session_state.original_dataset, variants); st.session_state.synthetic_dataset = synthetic; st.write(f"Step 4: Mode hasil: {mode}"); status.update(label="Synthetic dataset selesai.",state="complete")
        st.subheader("Data Description")
        if st.session_state.synthetic_dataset.empty: st.info("Belum ada synthetic dataset.")
        else:
            s1,s2 = st.columns(2); s1.metric("Rows",len(st.session_state.synthetic_dataset)); s2.metric("Columns",len(st.session_state.synthetic_dataset.columns))
        st.subheader("Tabular Synthetic Dataset"); st.dataframe(st.session_state.synthetic_dataset,use_container_width=True)

elif menu == "Preprocessing":
    st.header("Preprocessing"); st.write("Proses: **Cleaning, parsing, chunking, dan metadata extraction**.")
    if st.button("Run Preprocessing",type="primary", key="btn_run_preprocessing"):
        with st.status("Preprocessing...",expanded=True) as status:
            st.write("1. Cleaning: whitespace normalization dan null handling."); cleaned = clean_dataframe(active_df)
            st.write("2. Parsing: actor, precondition, main flow, alternative flow, exception flow, postcondition.")
            st.write(f"3. Chunking: {chunking_strategy}."); processed_chunks = build_chunks_cached(cleaned.to_json(),chunking_strategy,chunk_size,chunk_overlap)
            st.write("4. Metadata extraction: use_case_id, section, scenario_type, chunking_strategy."); status.update(label="Preprocessing selesai.",state="complete")
        st.subheader("Cleaned Dataset"); st.dataframe(cleaned,use_container_width=True); st.subheader("Chunk & Metadata Preview"); st.dataframe(processed_chunks,use_container_width=True)

elif menu == "Chunking Strategy":
    st.header("Chunking Strategy"); selected = st.selectbox("Pilih Chunking Strategy",CHUNKING_OPTIONS,index=CHUNKING_OPTIONS.index(chunking_strategy), key="page_chunking_strategy_select")
    explanations = {
        "fixed-size":"Memotong teks berdasarkan panjang karakter/token tetap. Cepat sebagai baseline, tetapi dapat memisahkan flow dari konteks.",
        "recursive":"Memotong bertahap berdasarkan separator paragraf/newline. Lebih natural daripada fixed-size.",
        "semantic":"Mengelompokkan bagian berdasarkan kedekatan makna. Prototype memakai semantic-light grouping agar deployment ringan.",
        "use case structure-aware":"Memotong berdasarkan actor, precondition, main flow, alternative flow, exception flow, postcondition sehingga traceability terjaga."
    }
    st.info(explanations[selected]); preview = build_chunks_cached(active_df.to_json(),selected,chunk_size,chunk_overlap); st.subheader("Chunk Preview"); st.dataframe(preview,use_container_width=True)

elif menu == "Retrieval Configuration":
    st.header("Retrieval Configuration")

    selected = st.selectbox(
        "Pilih Retrieval Method",
        RETRIEVAL_OPTIONS,
        index=RETRIEVAL_OPTIONS.index(retrieval_method),
        key="page_retrieval_method_select"
    )

    explanations = {
        "dense retrieval": (
            "TF-IDF cosine retrieval untuk deployment cepat; dapat diganti "
            "SentenceTransformer/FAISS pada eksperimen berat."
        ),
        "BM25/vectorless": (
            "Keyword ranking tanpa vector database; cocok untuk istilah eksplisit."
        ),
        "hybrid": (
            "Menggabungkan dense retrieval dan BM25/vectorless menggunakan "
            "Reciprocal Rank Fusion."
        ),
        "graph": (
            "Retrieval dengan boost berbasis struktur/relasi use case."
        ),
        "reranker": (
            "Candidate retrieval lalu lightweight reranking berbasis query dan section."
        ),
    }
    st.info(explanations[selected])

    st.caption(
        "Retrieval Test dijalankan sebagai retrieval pipeline independen. "
        "Pure LLM tidak digunakan pada halaman ini karena Pure LLM memang tidak memiliki retrieval stage."
    )

    query = st.text_input(
        "Retrieval Query",
        f"Generate test scenario and test plan for {selected_use_case_id}",
        key="retrieval_query_input"
    )

    if st.button(
        "Run Retrieval Test",
        type="primary",
        key="btn_run_retrieval_test"
    ):
        # Important: retrieval configuration must not inherit Pure LLM bypass.
        retrieval_test_architecture = "Standard RAG"

        with st.status("Running retrieval test...", expanded=True) as status:
            st.write(f"Query: {query}")
            st.write(f"Method: {selected}")
            st.write(f"Top-K: {top_k}")
            st.write(
                "Architecture context for retrieval test: Standard RAG "
                "(prevents Pure LLM no-retrieval bypass)."
            )

            result = retrieve_context(
                query,
                chunks_df,
                selected,
                top_k,
                retrieval_test_architecture
            )

            status.update(label="Retrieval test selesai.", state="complete")

        if result.empty:
            st.error(
                "Retrieval masih kosong. Periksa dataset aktif, hasil chunking, "
                "dan pastikan use_case_id tersedia."
            )
            st.subheader("Diagnostic")
            st.write("Active dataset rows:", len(active_df))
            st.write("Chunk rows:", len(chunks_df))
            st.write(
                "Available Use Case IDs:",
                chunks_df["use_case_id"].dropna().astype(str).unique().tolist()
                if "use_case_id" in chunks_df.columns else []
            )
        else:
            st.success(
                f"Ditemukan {len(result)} chunk untuk query retrieval."
            )
            display_cols = [
                col for col in [
                    "chunk_id",
                    "use_case_id",
                    "use_case_name",
                    "section",
                    "scenario_type",
                    "score",
                    "retrieval_method",
                    "content",
                ]
                if col in result.columns
            ]
            st.dataframe(
                result[display_cols],
                use_container_width=True
            )

            st.subheader("Retrieval Summary")
            s1, s2, s3 = st.columns(3)
            s1.metric("Retrieved Chunks", len(result))
            s2.metric(
                "Top Use Case",
                str(result.iloc[0].get("use_case_id", "N/A"))
            )
            s3.metric(
                "Top Score",
                f"{float(result.iloc[0].get('score', 0.0)):.4f}"
            )

elif menu == "RAG Architecture Selection":
    st.header("RAG Architecture Selection")
    st.write("Checklist 10 arsitektur RAG. **Pure LLM selalu aktif sebagai baseline eksperimen.**")

    selected_architectures = ["Pure LLM"]
    cols = st.columns(2)

    with cols[0]:
        st.checkbox(
            "Pure LLM (Baseline - Always Enabled)",
            value=True,
            disabled=True,
            key="arch_pure_llm_baseline"
        )

    rag_only_architectures = [arch for arch in RAG_ARCHITECTURES if arch != "Pure LLM"]
    for idx, arch in enumerate(rag_only_architectures):
        with cols[idx % 2]:
            checked = st.checkbox(
                arch,
                value=arch in st.session_state.selected_architectures,
                key=f"arch_{arch}"
            )
            if checked:
                selected_architectures.append(arch)

    st.session_state.selected_architectures = selected_architectures
    st.subheader("Selected Architectures")
    st.success(", ".join(selected_architectures))
    st.caption(
        "Semua hasil evaluasi dan Comparison akan otomatis dihitung relatif terhadap Pure LLM baseline."
    )

elif menu == "LLM Configuration":
    st.header("LLM Configuration"); selected_llm = st.selectbox("Open Source LLM",LLM_OPTIONS,index=LLM_OPTIONS.index(llm_name), key="page_llm_configuration_select")
    st.write("Model Hugging Face dimuat hanya saat digunakan agar startup tetap cepat. Fast Template Generator direkomendasikan untuk resource terbatas.")
    st.json({"selected_llm":selected_llm,"temperature":llm_temperature,"max_new_tokens":llm_max_new_tokens,"runtime":"CPU-compatible; lazy loading"})

elif menu == "Test Scenario Generation":
    st.header("Test Scenario Generation"); row = active_df[active_df["use_case_id"].astype(str)==str(selected_use_case_id)].iloc[0]
    query = st.text_input("Generation Query",f"Generate complete test scenario for {row.get('use_case_name',selected_use_case_id)} including positive, negative, alternative, and exception scenario.", key="scenario_generation_query")
    if st.button("Generate Test Scenario dari Use Case",type="primary", key="btn_generate_test_scenario"):
        logs = []
        with st.status("Generate Test Scenario...",expanded=True) as status:
            logs.append(f"Step 1: Load use case {selected_use_case_id}."); st.write(logs[-1])
            logs.append(f"Step 2: Use chunking strategy = {chunking_strategy}."); st.write(logs[-1])
            logs.append("Step 3: Skip retrieval augmentation for Pure LLM baseline." if generation_architecture == "Pure LLM" else f"Step 3: Retrieve top-{top_k} context using {retrieval_method}."); st.write(logs[-1]); retrieved = retrieve_context(query,chunks_df,retrieval_method,top_k,generation_architecture)
            logs.append(f"Step 4: Apply RAG architecture = {generation_architecture}."); st.write(logs[-1])
            logs.append(f"Step 5: Generate with open-source LLM = {llm_name}."); st.write(logs[-1]); scenario = generate_test_scenario(row,generation_architecture,retrieved,llm_name,llm_max_new_tokens,llm_temperature)
            logs.append("Step 6: Save generated scenario and retrieved evidence."); st.write(logs[-1]); st.session_state.last_scenario = scenario; st.session_state.last_retrieved_scenario = retrieved; status.update(label="Test Scenario selesai dibuat.",state="complete")
        st.subheader("Verbose Process"); st.code("\n".join(logs),language="text"); st.subheader("Retrieved Context"); st.dataframe(retrieved,use_container_width=True); st.subheader("Generated Test Scenario"); st.text_area("Output",scenario,height=430, key="scenario_output_textarea")

elif menu == "Test Plan Recommendation":
    st.header("Test Plan Recommendation"); row = active_df[active_df["use_case_id"].astype(str)==str(selected_use_case_id)].iloc[0]
    query = st.text_input("Test Plan Query",f"Generate recommended test plan for {row.get('use_case_name',selected_use_case_id)}.", key="test_plan_query_input")
    if st.button("Generate Rekomendasi Test Plan",type="primary", key="btn_generate_test_plan"):
        logs = []
        with st.status("Generate Test Plan Recommendation...",expanded=True) as status:
            logs.append(f"Step 1: Load use case {selected_use_case_id}."); st.write(logs[-1])
            logs.append("Step 2: Skip retrieval augmentation for Pure LLM baseline." if generation_architecture == "Pure LLM" else f"Step 2: Retrieve top-{top_k} planning context using {retrieval_method}."); st.write(logs[-1]); retrieved = retrieve_context(query,chunks_df,retrieval_method,top_k,generation_architecture)
            logs.append("Step 3: Read generated test scenario; create temporary scenario if absent."); st.write(logs[-1]); scenario = st.session_state.last_scenario or generate_test_scenario(row,generation_architecture,retrieved,llm_name,llm_max_new_tokens,llm_temperature)
            logs.append(f"Step 4: Apply RAG architecture = {generation_architecture}."); st.write(logs[-1])
            logs.append(f"Step 5: Generate recommendation with LLM = {llm_name}."); st.write(logs[-1]); plan = generate_test_plan(row,generation_architecture,retrieved,scenario,llm_name,llm_max_new_tokens,llm_temperature)
            logs.append("Step 6: Save recommended test plan and evidence."); st.write(logs[-1]); st.session_state.last_plan = plan; st.session_state.last_retrieved_plan = retrieved; status.update(label="Test Plan Recommendation selesai.",state="complete")
        st.subheader("Verbose Process"); st.code("\n".join(logs),language="text"); st.subheader("Retrieved Context"); st.dataframe(retrieved,use_container_width=True); st.subheader("Recommended Test Plan"); st.text_area("Output",plan,height=500, key="test_plan_output_textarea")

elif menu == "Evaluation Metric":
    st.header("Evaluation Metric")
    st.write(
        "Kalkulasi Recall@3, MRR, Semantic Similarity, ROUGE-L, BLEU, Coverage, "
        "Faithfulness. **Setiap evaluasi selalu dibandingkan dengan Pure LLM baseline.**"
    )

    row = active_df[active_df["use_case_id"].astype(str) == str(selected_use_case_id)].iloc[0]
    scenario_tab, plan_tab = st.tabs(["Scenario Metrics", "Test Plan Metrics"])

    with scenario_tab:
        scenario_output = st.text_area(
            "Generated Test Scenario",
            st.session_state.last_scenario,
            height=250,
            key="eval_generated_scenario"
        )
        scenario_ref = st.text_area(
            "Ground Truth Test Scenario",
            normalize_text(row.get("expected_test_scenario", "")),
            height=160,
            key="eval_ground_truth_scenario"
        )

        if st.button(
            "Calculate Scenario Metrics vs Pure LLM",
            type="primary",
            key="btn_calculate_scenario_metrics"
        ):
            selected_retrieved = st.session_state.last_retrieved_scenario
            if generation_architecture == "Pure LLM":
                selected_retrieved = retrieve_context(
                    f"test scenario {row.get('use_case_name','')}",
                    chunks_df,
                    retrieval_method,
                    top_k,
                    "Pure LLM"
                )
            elif selected_retrieved.empty:
                selected_retrieved = retrieve_context(
                    f"test scenario {row.get('use_case_name','')}",
                    chunks_df,
                    retrieval_method,
                    top_k,
                    generation_architecture
                )

            selected_metrics = calculate_metrics(
                scenario_output,
                scenario_ref,
                selected_retrieved,
                selected_use_case_id,
                row,
                "scenario"
            )

            pure_retrieved = retrieve_context(
                f"test scenario {row.get('use_case_name','')}",
                chunks_df,
                retrieval_method,
                top_k,
                "Pure LLM"
            )
            pure_output = generate_test_scenario(
                row,
                "Pure LLM",
                pure_retrieved,
                llm_name,
                llm_max_new_tokens,
                llm_temperature
            )
            pure_metrics = calculate_metrics(
                pure_output,
                scenario_ref,
                pure_retrieved,
                selected_use_case_id,
                row,
                "scenario"
            )

            comparison_df = pd.DataFrame([
                {"Model": generation_architecture, **selected_metrics},
                {"Model": "Pure LLM (Baseline)", **pure_metrics},
            ])
            delta_df = pd.DataFrame([{
                "Comparison": f"{generation_architecture} - Pure LLM",
                **{
                    f"Delta {metric}": selected_metrics[metric] - pure_metrics[metric]
                    for metric in selected_metrics
                }
            }])

            st.subheader("Tabel Kalkulasi vs Pure LLM")
            st.dataframe(comparison_df, use_container_width=True)

            st.subheader("Delta terhadap Pure LLM")
            st.dataframe(delta_df, use_container_width=True)

            st.subheader("Grafik Perbandingan")
            st.bar_chart(comparison_df.set_index("Model"))

            with st.expander("Lihat Pure LLM Baseline Output"):
                st.text_area(
                    "Pure LLM Scenario Baseline",
                    pure_output,
                    height=260,
                    key="eval_pure_llm_scenario_output"
                )

    with plan_tab:
        plan_output = st.text_area(
            "Generated Test Plan",
            st.session_state.last_plan,
            height=250,
            key="eval_generated_test_plan"
        )
        plan_ref = st.text_area(
            "Ground Truth Test Plan",
            normalize_text(row.get("expected_test_plan", "")),
            height=160,
            key="eval_ground_truth_test_plan"
        )

        if st.button(
            "Calculate Test Plan Metrics vs Pure LLM",
            type="primary",
            key="btn_calculate_test_plan_metrics"
        ):
            selected_retrieved = st.session_state.last_retrieved_plan
            if generation_architecture == "Pure LLM":
                selected_retrieved = retrieve_context(
                    f"test plan {row.get('use_case_name','')}",
                    chunks_df,
                    retrieval_method,
                    top_k,
                    "Pure LLM"
                )
            elif selected_retrieved.empty:
                selected_retrieved = retrieve_context(
                    f"test plan {row.get('use_case_name','')}",
                    chunks_df,
                    retrieval_method,
                    top_k,
                    generation_architecture
                )

            selected_metrics = calculate_metrics(
                plan_output,
                plan_ref,
                selected_retrieved,
                selected_use_case_id,
                row,
                "plan"
            )

            pure_retrieved = retrieve_context(
                f"test plan {row.get('use_case_name','')}",
                chunks_df,
                retrieval_method,
                top_k,
                "Pure LLM"
            )
            pure_scenario = generate_test_scenario(
                row,
                "Pure LLM",
                pure_retrieved,
                llm_name,
                llm_max_new_tokens,
                llm_temperature
            )
            pure_plan = generate_test_plan(
                row,
                "Pure LLM",
                pure_retrieved,
                pure_scenario,
                llm_name,
                llm_max_new_tokens,
                llm_temperature
            )
            pure_metrics = calculate_metrics(
                pure_plan,
                plan_ref,
                pure_retrieved,
                selected_use_case_id,
                row,
                "plan"
            )

            comparison_df = pd.DataFrame([
                {"Model": generation_architecture, **selected_metrics},
                {"Model": "Pure LLM (Baseline)", **pure_metrics},
            ])
            delta_df = pd.DataFrame([{
                "Comparison": f"{generation_architecture} - Pure LLM",
                **{
                    f"Delta {metric}": selected_metrics[metric] - pure_metrics[metric]
                    for metric in selected_metrics
                }
            }])

            st.subheader("Tabel Kalkulasi vs Pure LLM")
            st.dataframe(comparison_df, use_container_width=True)

            st.subheader("Delta terhadap Pure LLM")
            st.dataframe(delta_df, use_container_width=True)

            st.subheader("Grafik Perbandingan")
            st.bar_chart(comparison_df.set_index("Model"))

            with st.expander("Lihat Pure LLM Baseline Output"):
                st.text_area(
                    "Pure LLM Test Plan Baseline",
                    pure_plan,
                    height=300,
                    key="eval_pure_llm_plan_output"
                )

elif menu == "Comparison":
    st.header("Comparison")

    selected_architectures = list(
        dict.fromkeys(["Pure LLM"] + list(st.session_state.selected_architectures))
    )
    st.info(
        "Pure LLM otomatis digunakan sebagai baseline. Selected architectures: "
        + ", ".join(selected_architectures)
    )

    if st.button(
        "Run Comparison Antar Arsitektur vs Pure LLM",
        type="primary",
        key="btn_run_architecture_comparison"
    ):
        records = []
        total = max(1, len(active_df) * len(selected_architectures))
        progress = st.progress(0)
        counter = 0

        with st.status("Architecture comparison vs Pure LLM...", expanded=True) as status:
            for _, row in active_df.iterrows():
                ucid = normalize_text(row.get("use_case_id", "UC"))
                name = normalize_text(row.get("use_case_name", ucid))

                for arch in selected_architectures:
                    st.write(f"{arch} -> {ucid} {name}")

                    scenario_retrieved = retrieve_context(
                        f"Generate test scenario for {name}",
                        chunks_df,
                        retrieval_method,
                        top_k,
                        arch
                    )
                    scenario = generate_test_scenario(
                        row,
                        arch,
                        scenario_retrieved,
                        llm_name,
                        llm_max_new_tokens,
                        llm_temperature
                    )
                    scenario_metrics = calculate_metrics(
                        scenario,
                        normalize_text(row.get("expected_test_scenario", "")),
                        scenario_retrieved,
                        ucid,
                        row,
                        "scenario"
                    )

                    plan_retrieved = retrieve_context(
                        f"Generate recommended test plan for {name}",
                        chunks_df,
                        retrieval_method,
                        top_k,
                        arch
                    )
                    plan = generate_test_plan(
                        row,
                        arch,
                        plan_retrieved,
                        scenario,
                        llm_name,
                        llm_max_new_tokens,
                        llm_temperature
                    )
                    plan_metrics = calculate_metrics(
                        plan,
                        normalize_text(row.get("expected_test_plan", "")),
                        plan_retrieved,
                        ucid,
                        row,
                        "plan"
                    )

                    combined = {
                        metric: (scenario_metrics[metric] + plan_metrics[metric]) / 2
                        for metric in scenario_metrics
                    }
                    combined["Overall Score"] = overall_score(combined)
                    combined.update({
                        "Use Case ID": ucid,
                        "Use Case Name": name,
                        "RAG Architecture": arch,
                        "Baseline Type": "Baseline" if arch == "Pure LLM" else "RAG Candidate",
                        "Chunking": "N/A - No Retrieval" if arch == "Pure LLM" else chunking_strategy,
                        "Retrieval": "None" if arch == "Pure LLM" else retrieval_method,
                        "LLM": llm_name,
                    })
                    records.append(combined)

                    counter += 1
                    progress.progress(counter / total)

            status.update(label="Comparison vs Pure LLM selesai.", state="complete")

        detail_df = pd.DataFrame(records)

        metric_columns = [
            "Recall@3",
            "MRR",
            "Semantic Similarity",
            "ROUGE-L",
            "BLEU",
            "Coverage",
            "Faithfulness",
            "Overall Score",
        ]

        # Join each architecture result with its Pure LLM result for the same use case.
        baseline_df = detail_df[
            detail_df["RAG Architecture"] == "Pure LLM"
        ][["Use Case ID"] + metric_columns].copy()

        baseline_df = baseline_df.rename(
            columns={metric: f"Pure LLM {metric}" for metric in metric_columns}
        )

        detail_df = detail_df.merge(
            baseline_df,
            on="Use Case ID",
            how="left"
        )

        for metric in metric_columns:
            detail_df[f"Delta {metric} vs Pure LLM"] = (
                detail_df[metric] - detail_df[f"Pure LLM {metric}"]
            )

        delta_columns = [
            f"Delta {metric} vs Pure LLM" for metric in metric_columns
        ]

        summary_df = (
            detail_df.groupby("RAG Architecture")[
                metric_columns + delta_columns
            ]
            .mean()
            .reset_index()
            .sort_values("Overall Score", ascending=False)
        )

        summary_df["Baseline Comparison"] = np.where(
            summary_df["RAG Architecture"] == "Pure LLM",
            "Baseline",
            np.where(
                summary_df["Delta Overall Score vs Pure LLM"] > 0,
                "Better than Pure LLM",
                np.where(
                    summary_df["Delta Overall Score vs Pure LLM"] < 0,
                    "Worse than Pure LLM",
                    "Equal to Pure LLM"
                )
            )
        )

        st.session_state.comparison_detail = detail_df
        st.session_state.comparison_summary = summary_df

        st.subheader("Detail Comparison vs Pure LLM")
        st.dataframe(detail_df, use_container_width=True)

        st.subheader("Architecture Ranking")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Overall Score")
        st.bar_chart(
            summary_df.set_index("RAG Architecture")["Overall Score"]
        )

        st.subheader("Delta Overall Score vs Pure LLM")
        st.bar_chart(
            summary_df.set_index("RAG Architecture")[
                "Delta Overall Score vs Pure LLM"
            ]
        )

        st.subheader("Metric Comparison")
        st.bar_chart(
            summary_df.set_index("RAG Architecture")[
                [
                    "Recall@3",
                    "MRR",
                    "Semantic Similarity",
                    "Coverage",
                    "Faithfulness",
                ]
            ]
        )

        st.subheader("Metric Delta vs Pure LLM")
        st.bar_chart(
            summary_df.set_index("RAG Architecture")[
                [
                    "Delta Recall@3 vs Pure LLM",
                    "Delta MRR vs Pure LLM",
                    "Delta Semantic Similarity vs Pure LLM",
                    "Delta Coverage vs Pure LLM",
                    "Delta Faithfulness vs Pure LLM",
                ]
            ]
        )

        st.download_button(
            "Download Comparison Detail CSV",
            detail_df.to_csv(index=False).encode("utf-8"),
            "rag_comparison_vs_pure_llm_detail.csv",
            "text/csv",
            key="download_comparison_vs_pure_llm_detail"
        )
        st.download_button(
            "Download Comparison Summary CSV",
            summary_df.to_csv(index=False).encode("utf-8"),
            "rag_comparison_vs_pure_llm_summary.csv",
            "text/csv",
            key="download_comparison_vs_pure_llm_summary"
        )
