"""Skeleton-driven extraction, navigation, and grounded answer generation."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

import fitz

from .skeleton_generation import SkeletonLLM
from .skeleton_retrieval import SkeletonHybridRetriever
from .skeleton_pipeline_types import Navigation

logger = logging.getLogger(__name__)


class SkeletonValidationError(ValueError):
    """Raised when a skeleton boundary cannot be resolved unambiguously."""


def _normalise_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _comparison_text(value: str) -> str:
    value = value.replace("\ufffd", "'")
    value = value.replace("’", "'").replace("‘", "'")
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"\.{2,}", ".", value)
    value = re.sub(r"^[^A-Za-z0-9\u0600-\u06ff]+", "", value)
    value = re.sub(r"(?<=\d)\.(?=\s)", "", value)
    return re.sub(r"\s+", " ", value).strip().rstrip(" .:").casefold()


def extract_clean_text(pdf_path: Path) -> str:
    """Extract text and remove only repeated page furniture/watermarks."""
    document = fitz.open(pdf_path)
    pages = [[_normalise_line(line) for line in page.get_text().splitlines()]
             for page in document]
    frequencies: dict[str, int] = {}
    for lines in pages:
        for line in set(lines[:8] + lines[-8:]):
            if line:
                frequencies[line] = frequencies.get(line, 0) + 1
    repeated_edges = {line for line, count in frequencies.items() if count >= 2}

    cleaned_pages: list[str] = []
    for lines in pages:
        kept: list[str] = []
        for line in lines:
            if not line or line in repeated_edges:
                continue
            if re.search(r"docusign|electronically signed|document id", line, re.I):
                continue
            if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", line, re.I):
                continue
            if re.fullmatch(r"\d+", line):
                continue
            kept.append(line)
        cleaned_pages.append("\n".join(kept))
    body_start = 0
    contents_seen = False
    for index, page in enumerate(cleaned_pages):
        lines = page.splitlines()
        if any(line.casefold() == "contents" for line in lines):
            contents_seen = True
            continue
        if contents_seen and any(line.casefold() in {"introduction", "introduction"} for line in lines):
            body_start = index
            break
    return "\n\n".join(page for page in cleaned_pages[body_start:] if page).strip()


def _iter_nodes(tree: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from tree.get("nodes", [])


def _find_boundary(text: str, value: str, start: int) -> int | None:
    expected = _comparison_text(value)
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    for index, line_offset in enumerate(offsets):
        if line_offset < start:
            continue
        for width in range(1, 5):
            if index + width > len(lines):
                break
            candidate = _comparison_text(" ".join(line.strip() for line in lines[index:index + width]))
            if not candidate.startswith(expected):
                continue
            remainder = candidate[len(expected):].strip()
            if not remainder or re.match(r"^[-,:;()\[\]]", remainder):
                return line_offset + len(line) - len(line.lstrip())
    return None


def _validate_boundary(text: str, node: dict[str, Any], field: str, start: int) -> int:
    value = node.get(field)
    if not value:
        raise SkeletonValidationError(f"Node {node.get('id')} has no {field} boundary")
    position = _find_boundary(text, value, start)
    if position is None:
        raise SkeletonValidationError(
            f"Node {node.get('id')} {field} {value!r} was not found after character {start}"
        )
    return position


def _row_sentence(cells: list[str]) -> str:
    values = [value.strip(" -") for value in cells]
    if len(values) < 6:
        return ": ".join(values)
    communication, audience, messages, medium, owner, timing = values[:6]
    return (
        f"{communication}: sent to {audience} via {medium} by {owner}, "
        f"{timing}. Covers: {messages}"
    )


def _extract_table_rows(table_text: str) -> list[str]:
    rows: list[str] = []
    for raw_line in table_text.splitlines():
        line = raw_line.strip()
        if not line or re.search(r"communication.*audience.*key messages", line, re.I):
            continue
        cells = [cell.strip() for cell in re.split(r"\s*\|\s*|\t+", line) if cell.strip()]
        if len(cells) >= 5:
            rows.append(_row_sentence(cells))
    return rows


def enrich_tree(skeleton: dict[str, Any], text: str) -> dict[str, Any]:
    """Fill leaf/form text and expand table nodes into row leaves."""
    tree = json.loads(json.dumps(skeleton))
    nodes = list(_iter_nodes(tree))
    generated: list[dict[str, Any]] = []
    cursor = 0

    for node in nodes:
        node_type = node.get("node_type")
        if node_type not in {"leaf", "form", "table"} or not node.get("heading"):
            continue
        start = _validate_boundary(text, node, "heading", cursor)
        end = len(text)
        next_heading = node.get("next_heading")
        if next_heading:
            end = _validate_boundary(text, node, "next_heading", start + len(node["heading"]))
        own_text = text[start + len(node["heading"]):end].strip()
        if node_type == "table":
            child_ids: list[str] = []
            for index, row in enumerate(_extract_table_rows(own_text), 1):
                row_id = f"{node['id']}.row{index:02d}"
                generated.append({
                    "id": row_id,
                    "path": f"{node['path']} > Row {index}",
                    "level": node.get("level", 1) + 1,
                    "node_type": "leaf",
                    "own_text": row,
                })
                child_ids.append(row_id)
            node["children"] = child_ids
        node["own_text"] = own_text
        cursor = end

    tree["nodes"] = nodes + generated
    return tree


def write_enriched_corpus(
    skeleton_dir: Path = Path("data/files_as_nodes"),
    pdf_dir: Path = Path("data/pdf"),
    output_dir: Path = Path("data/enriched_nodes"),
    outline_path: Path = Path("data/outline.json"),
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outlines: list[dict[str, Any]] = []
    written: list[Path] = []
    for skeleton_path in sorted(skeleton_dir.glob("skeleton_*.json")):
        skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
        matches = list(pdf_dir.glob(f"*{skeleton['doc_id']}*"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one PDF for {skeleton['doc_id']}, found {len(matches)}"
            )
        enriched = enrich_tree(skeleton, extract_clean_text(matches[0]))
        output_path = output_dir / f"{skeleton['doc_id']}.json"
        output_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(output_path)
        for node in enriched["nodes"]:
            outlines.append({
                "doc_id": enriched["doc_id"],
                "id": node["id"],
                "path": node["path"],
                "node_type": node["node_type"],
            })
    outline_path.write_text(json.dumps(outlines, ensure_ascii=False, indent=2), encoding="utf-8")
    return written


class SkeletonCorpus:
    """In-memory lookup over enriched trees and the combined outline."""

    def __init__(self, tree_dir: Path = Path("data/enriched_nodes"), outline_path: Path = Path("data/outline.json")):
        self.outline = json.loads(outline_path.read_text(encoding="utf-8"))
        self.nodes: dict[tuple[str, str], dict[str, Any]] = {}
        for tree_path in tree_dir.glob("*.json"):
            tree = json.loads(tree_path.read_text(encoding="utf-8"))
            for node in tree.get("nodes", []):
                self.nodes[(tree["doc_id"], node["id"])] = node

    def fetch(self, navigation: Navigation) -> list[dict[str, str]]:
        contexts: list[dict[str, str]] = []
        for doc_id, node_id in navigation.node_refs:
            ids = self._descendant_ids(doc_id, node_id) if navigation.broad else [node_id]
            for node_id in ids:
                node = self.nodes[(doc_id, node_id)]
                if node.get("own_text"):
                    contexts.append({"path": node["path"], "text": node["own_text"]})
        return contexts

    def _descendant_ids(self, doc_id: str, root_id: str) -> list[str]:
        result: list[str] = []

        def visit(node_id: str) -> None:
            result.append(node_id)
            for child_id in self.nodes[(doc_id, node_id)].get("children", []):
                visit(child_id)

        visit(root_id)
        return result


class SkeletonQA:
    def __init__(self, corpus: SkeletonCorpus, llm: SkeletonLLM, retriever: SkeletonHybridRetriever | None = None):
        self.corpus = corpus
        self.llm = llm
        self.retriever = retriever

    def answer(self, question: str) -> dict[str, Any]:
        similarity = self.retriever.search(question) if self.retriever else []
        navigation = self.llm.navigate(question, self.corpus.outline, similarity)
        recommended = set(navigation.node_refs)
        for result in similarity[:2]:
            recommended.add((result["doc_id"], result["id"]))
        navigation = Navigation(list(recommended), navigation.broad)
        contexts = self.corpus.fetch(navigation)
        result = self.llm.answer(question, contexts)
        result["navigation"] = navigation
        result["retrieval"] = similarity
        return result


def build_from_environment() -> list[Path]:
    return write_enriched_corpus(
        skeleton_dir=Path(os.getenv("SKELETON_DIR", "data/files_as_nodes")),
        pdf_dir=Path(os.getenv("PDF_DIR", "data/pdf")),
        output_dir=Path(os.getenv("ENRICHED_NODES_DIR", "data/enriched_nodes")),
        outline_path=Path(os.getenv("OUTLINE_PATH", "data/outline.json")),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for path in build_from_environment():
        print(path)