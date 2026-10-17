"""Offline semantic retain prototypes for UniBal dual feedback."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def retain_texts(dataset, tokenizer):
    """Use the same question/answer or text chunk represented by each retain item."""
    if hasattr(dataset, "question_key") and hasattr(dataset, "answer_key"):
        for row in dataset.data:
            answer = row[dataset.answer_key]
            if isinstance(answer, list):
                answer = " ".join(map(str, answer))
            yield f"{row[dataset.question_key]}\n{answer}"
    elif hasattr(dataset, "chunks"):
        yield from dataset.chunks
    else:
        for item in dataset:
            yield tokenizer.decode(item["input_ids"], skip_special_tokens=True)


def encode_texts(texts, model_name, batch_size=32, max_length=256, device="cpu"):
    if not texts or batch_size <= 0 or max_length <= 0:
        raise ValueError(
            "Encoder input and batch settings must be nonempty and positive"
        )
    encoder_tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = encoder_tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(
                torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy()
            )
    return np.concatenate(vectors, axis=0)


def k_medoids_prototypes(
    vectors, n_prototypes, seed=0, sample_size=5000, iterations=10
):
    """Alternating k-medoids on a bounded subset, then full-set assignment.

    The sampling limit bounds the quadratic distance cost on large corpora.
    Every selected prototype is an actual retain example.
    """
    n = len(vectors)
    if n == 0:
        raise ValueError("The retain dataset is empty")
    k = min(n_prototypes, n)
    if k <= 0:
        raise ValueError("n_prototypes must be positive")
    if sample_size <= 0 or iterations <= 0:
        raise ValueError("k-medoids sample size and iterations must be positive")
    rng = np.random.default_rng(seed)
    subset = rng.choice(n, size=min(n, max(sample_size, k)), replace=False)
    x = vectors[subset]
    distance = np.maximum(0, 1 - x @ x.T)
    medoids = [int(rng.integers(len(subset)))]
    nearest = distance[:, medoids[0]].copy()
    while len(medoids) < k:
        scores = nearest.copy()
        scores[medoids] = -1
        next_medoid = int(np.argmax(scores))
        medoids.append(next_medoid)
        nearest = np.minimum(nearest, distance[:, next_medoid])
    medoids = np.asarray(medoids, dtype=np.int64)
    for _ in range(iterations):
        labels = distance[:, medoids].argmin(axis=1)
        updated = medoids.copy()
        for cluster in range(k):
            members = np.flatnonzero(labels == cluster)
            if len(members):
                updated[cluster] = members[
                    np.argmin(distance[np.ix_(members, members)].sum(axis=1))
                ]
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    full_labels = np.empty(n, dtype=np.int64)
    for start in range(0, n, 4096):
        full_labels[start : start + 4096] = (
            1 - vectors[start : start + 4096] @ vectors[subset[medoids]].T
        ).argmin(axis=1)
    counts = np.bincount(full_labels, minlength=k)
    active = counts > 0
    return subset[medoids[active]].tolist(), (counts[active] / n).tolist()


def load_or_build_prototypes(
    dataset,
    tokenizer,
    encoder_name,
    n_prototypes=40,
    cache_path=None,
    batch_size=32,
    max_length=256,
    sample_size=5000,
    seed=0,
):
    texts = list(retain_texts(dataset, tokenizer))
    valid_indices = [index for index, value in enumerate(texts) if value.strip()]
    if not valid_indices:
        raise ValueError("The retain dataset contains no nonempty examples")
    digest = hashlib.sha256()
    for item in texts:
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    signature = {
        "cache_version": 2,
        "dataset_sha256": digest.hexdigest(),
        "encoder": encoder_name,
        "n_prototypes": n_prototypes,
        "max_length": max_length,
        "sample_size": sample_size,
        "seed": seed,
    }
    if cache_path and Path(cache_path).exists():
        payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if all(payload.get(key) == value for key, value in signature.items()):
            return payload["indices"], payload["weights"]
    vectors = encode_texts(
        [texts[index] for index in valid_indices], encoder_name, batch_size, max_length
    )
    local_indices, weights = k_medoids_prototypes(
        vectors, n_prototypes, seed, sample_size
    )
    indices = [valid_indices[index] for index in local_indices]
    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**signature, "indices": indices, "weights": weights}),
            encoding="utf-8",
        )
    return indices, weights
