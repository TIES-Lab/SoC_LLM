#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*|\d+'[bhd][0-9a-fA-F_xzXZ]+|\d+|[&|~^!=<>]+")


class HtmlTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.links = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in {"p", "div", "section", "article", "li", "br", "pre", "code", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "div", "section", "article", "li", "pre", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        data = data.strip()
        if data:
            self.parts.append(data + " ")

    def text(self):
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_source_list(path):
    sources = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            sources.append(line)
    return sources


def fetch_url(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 rag-inference"})
    with urlopen(req, timeout=30) as resp:
        content_type = resp.headers.get("content-type", "")
        body = resp.read().decode("utf-8", errors="replace")
    return body, content_type


def extract_html(html):
    parser = HtmlTextExtractor()
    parser.feed(html)
    return parser.text(), parser.links


def should_follow_url(root_url, candidate_link):
    joined = urljoin(root_url, candidate_link)
    root = urlparse(root_url)
    candidate = urlparse(joined)
    if candidate.netloc != root.netloc:
        return False, joined
    if not candidate.path.endswith((".html", "/")):
        return False, joined
    return True, joined


def load_url_docs(url, max_pages):
    queue = [url]
    seen = set()
    docs = []

    while queue and len(seen) < max_pages:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)

        try:
            body, content_type = fetch_url(current)
        except Exception as exc:
            print(f"Warning: could not fetch {current}: {exc}")
            continue

        if "html" in content_type or current.endswith((".html", "/")):
            text, links = extract_html(body)
            for link in links:
                ok, joined = should_follow_url(url, link)
                if ok and joined not in seen and joined not in queue:
                    queue.append(joined)
        else:
            text = body

        if text:
            docs.append({"source": current, "text": text})

    return docs


def load_json_docs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            text = item.get("text") or item.get("prompt", "") + "\n" + item.get("response", "")
        else:
            text = str(item)
        text = text.strip()
        if text:
            docs.append({"source": f"{path}#{idx}", "text": text})
    return docs


def load_local_doc(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if path.lower().endswith((".html", ".htm")):
        text, _ = extract_html(text)
    return [{"source": path, "text": text}]


def load_docs(sources, max_html_pages_per_url):
    docs = []
    for source in sources:
        if source.startswith(("http://", "https://")):
            docs.extend(load_url_docs(source, max_html_pages_per_url))
        elif source.lower().endswith(".json"):
            docs.extend(load_json_docs(source))
        else:
            docs.extend(load_local_doc(source))
    return docs


def chunk_docs(docs, chunk_chars, chunk_overlap):
    chunks = []
    for doc in docs:
        text = re.sub(r"\n{3,}", "\n\n", doc["text"]).strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"source": doc["source"], "text": chunk})
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks


def search_tokens(text):
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [search_tokens(chunk["text"]) for chunk in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)

        self.term_freqs = []
        doc_freq = defaultdict(int)
        for tokens in self.doc_tokens:
            counts = Counter(tokens)
            self.term_freqs.append(counts)
            for term in counts:
                doc_freq[term] += 1

        n_docs = len(chunks)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(self, query, top_k):
        query_terms = search_tokens(query)
        scores = []
        for idx, freqs in enumerate(self.term_freqs):
            doc_len = self.doc_lens[idx] or 1
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1) / denom)
            scores.append((score, idx))

        scores.sort(reverse=True)
        return [self.chunks[idx] for score, idx in scores[:top_k] if score > 0]


def build_context(retrieved, max_context_chars):
    blocks = []
    used = 0
    for rank, chunk in enumerate(retrieved, start=1):
        block = f"### Retrieved Context {rank}\nSource: {chunk['source']}\n{chunk['text']}\n"
        if used + len(block) > max_context_chars:
            block = block[: max_context_chars - used]
        if block.strip():
            blocks.append(block)
            used += len(block)
        if used >= max_context_chars:
            break
    return "\n\n".join(blocks)


def make_prompt(user_code, context):
    return f"""You are an AI programming assistant using a fine-tuned Deepseek Coder model.
You only answer questions related to computer science.

Use the retrieved context as reference material. If the retrieved context is irrelevant,
ignore it and answer from the user's SystemVerilog code.

### Retrieved Context:
{context}

### Instruction:
Generate a list of asynchronous SystemVerilog assertions from the following code:

{user_code}

### Response:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rag_config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_code = Path(cfg["input_code_file"]).read_text(encoding="utf-8", errors="replace")

    rag_sources = load_source_list(cfg["rag_sources_file"])

    print("Loading RAG sources...")
    docs = load_docs(rag_sources, cfg.get("max_html_pages_per_url", 20))
    chunks = chunk_docs(
        docs,
        cfg.get("chunk_chars", 1800),
        cfg.get("chunk_overlap", 250),
    )
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")

    index = BM25Index(chunks)
    retrieved = index.search(input_code, cfg.get("top_k", 5))
    context = build_context(retrieved, cfg.get("max_context_chars", 7000))

    print(f"Loading model from {cfg['model_dir']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_dir"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    prompt = make_prompt(input_code, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Generating response...\n")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=cfg.get("max_new_tokens", 2048),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.9),
            do_sample=cfg.get("temperature", 0.7) > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("### Response:")[-1].strip()

    output_file = cfg.get("output_file")
    if output_file:
        Path(output_file).write_text(response + "\n", encoding="utf-8")
        print(f"Saved response to {output_file}\n")

    print("=== MODEL OUTPUT ===\n")
    print(response)


if __name__ == "__main__":
    main()

