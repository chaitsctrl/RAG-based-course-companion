from __future__ import annotations

import json
import os
from typing import Dict, List

import joblib
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity


EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
GENERATE_URL = os.environ.get("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
GENERATION_MODEL = os.environ.get("OLLAMA_GENERATE_MODEL", "llama3.2")
TOP_K_DEFAULT = int(os.environ.get("RAG_TOP_K", 5))
EMBEDDINGS_PATH = os.environ.get("RAG_EMBEDDINGS_PATH", "embeddings.joblib")


app = Flask(__name__)


try:
    _df = joblib.load(EMBEDDINGS_PATH)
except FileNotFoundError as exc:
    raise SystemExit(
        f"Could not find embeddings at {EMBEDDINGS_PATH}. "
        "Generate embeddings before starting the server."
    ) from exc


if "embedding" not in _df.columns:
    raise SystemExit("Loaded dataframe is missing the 'embedding' column.")


_embedding_matrix = np.vstack(_df["embedding"].values)


def _create_embedding(text_list: List[str]) -> List[List[float]]:
    response = requests.post(
        EMBED_URL,
        json={"model": EMBEDDING_MODEL, "input": text_list},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["embeddings"]


def _inference(prompt: str) -> str:
    response = requests.post(
        GENERATE_URL,
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "").strip()


def _get_context(question_embedding: np.ndarray, top_k: int) -> List[Dict]:
    similarities = cosine_similarity(_embedding_matrix, [question_embedding]).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    contexts: List[Dict] = []
    for idx in top_indices:
        row = _df.iloc[idx]
        contexts.append(
            {
                "title": row.get("title"),
                "start": row.get("start"),
                "end": row.get("end"),
                "text": row.get("text"),
                "score": float(similarities[idx]),
            }
        )
    return contexts


def _build_prompt(contexts: List[Dict], question: str) -> str:
    context_json = json.dumps(contexts, ensure_ascii=False)
    return (
        "This course teaches basic English and conversation. "
        "Here are subtitle chunks containing the video title, "
        "start time (seconds), end time (seconds), and transcript text:\n\n"
        f"{context_json}\n"
        "---------------------------------\n"
        f'"{question}"\n'
        "Respond in a friendly tone that references the relevant video(s) "
        "and timestamps. If the question is unrelated to the course, politely "
        "state that you can only answer course-related questions."
    )


def run_rag(question: str, top_k: int = TOP_K_DEFAULT) -> Dict:
    if not question:
        raise ValueError("Question must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    question_embedding = np.array(_create_embedding([question])[0])
    contexts = _get_context(question_embedding, min(top_k, len(_df)))
    prompt = _build_prompt(contexts, question)
    answer = _inference(prompt)
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "prompt": prompt,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    payload = request.get_json(force=True, silent=True) or {}
    question = (payload.get("question") or "").strip()
    top_k = payload.get("top_k", TOP_K_DEFAULT)

    try:
        top_k_int = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer."}), 400

    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        result = run_rag(question, top_k_int)
    except requests.HTTPError as exc:
        return (
            jsonify(
                {"error": "Model endpoint returned an error.", "details": str(exc)}
            ),
            502,
        )
    except requests.RequestException as exc:
        return (
            jsonify(
                {"error": "Unable to reach the model endpoint.", "details": str(exc)}
            ),
            504,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "question": result["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

