import json
import re
import numpy as np
import faiss
import pickle
import os
from fastembed import TextEmbedding

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
FASTEMBED_CACHE = '.fastembed_cache'
CHUNK_SIZE = 600
OVERLAP = 100

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _split_at_boundary(text: str, max_chars: int) -> tuple[str, str]:
    if len(text) <= max_chars:
        return text, ""
    for sep in (' ', '.', ','):
        pos = text.rfind(sep, max_chars // 2, max_chars)
        if pos != -1:
            return text[:pos].rstrip(), text[pos:].lstrip()
    pos = text.rfind(' ', 0, max_chars)
    if pos > 0:
        return text[:pos].rstrip(), text[pos:].lstrip()
    return text[:max_chars], text[max_chars:]


def chunk_text(text: str, url: str) -> list[dict]:
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    current_sents: list[str] = []
    current_len: int = 0

    def flush():
        if current_sents:
            chunks.append({'text': ' '.join(current_sents), 'url': url})

    def overlap_tail() -> tuple[list[str], int]:
        tail, length = [], 0
        for s in reversed(current_sents):
            if length + len(s) + 1 > OVERLAP:
                break
            tail.insert(0, s)
            length += len(s) + 1
        return tail, length

    for para in paragraphs:
        for sent in _sentences(para):
            sent_len = len(sent) + 1
            if current_len + sent_len <= CHUNK_SIZE:
                current_sents.append(sent)
                current_len += sent_len
            else:
                flush()
                tail, tail_len = overlap_tail()
                if len(sent) > CHUNK_SIZE:
                    current_sents = []
                    current_len = 0
                    piece, rest = _split_at_boundary(sent, CHUNK_SIZE)
                    while piece:
                        chunks.append({'text': piece, 'url': url})
                        if not rest:
                            break
                        piece, rest = _split_at_boundary(rest, CHUNK_SIZE)
                else:
                    current_sents = tail + [sent]
                    current_len = tail_len + sent_len

    flush()
    return chunks


def build():
    with open('data/pages.json', 'r', encoding='utf-8') as f:
        pages = json.load(f)

    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page['content'], page['url']))

    print(f"Total chunks: {len(all_chunks)}")

    model = TextEmbedding(MODEL_NAME, cache_dir=FASTEMBED_CACHE)
    texts = [c['text'] for c in all_chunks]
    embeddings = np.array(list(model.embed(texts)), dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs('data', exist_ok=True)
    faiss.write_index(index, 'data/faiss.index')
    with open('data/chunks.pkl', 'wb') as f:
        pickle.dump(all_chunks, f)

    print(f"Knowledge base ready: {len(all_chunks)} chunks, {dimension}d embeddings")
    print("Saved: data/faiss.index, data/chunks.pkl")


if __name__ == '__main__':
    build()
