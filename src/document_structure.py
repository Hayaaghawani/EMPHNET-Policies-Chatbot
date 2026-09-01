"""
Map user questions to manual structure (sections, subsections, procedures, policies).

Intent is structural when the question targets a document header/TOC element
and should return ALL points under that exact subheader — not when the user
asks a focused fact (how many days, how to do X, etc.).
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .ingestion import Chunk

SUBSECTION_TYPE_TO_NUM = {
    "purpose": 1,
    "scope": 2,
    "responsibilities": 3,
    "policies": 4,
    "procedures": 5,
    "related documents": 6,
    "references": 7,
}

SUBSECTION_TYPES = list(SUBSECTION_TYPE_TO_NUM.keys())

PROCEDURE_PREFIX_RE = re.compile(r"Procedure\s+\d+\s*:\s*([^|\]]+)", re.IGNORECASE)

# Focused factual questions — always semantic/specific search
SPECIFIC_FACT_RE = re.compile(
    r"\b("
    r"how many|how much|how long|how do i|how to|how can|how should|"
    r"when |who |can i|am i|is there|does |do i|eligible|"
    r"number of days|days of|weeks of|what happens if"
    r")\b",
    re.IGNORECASE,
)

# Responsibility codes mistaken as policy headers during ingestion
ACRONYM_POLICY_BLOCKLIST = {
    "hrm", "hrs", "hro", "emp", "dm", "sys", "ed", "tl", "dir", "f&ad",
    "f&ad", "hr", "its", "to", "m", "tl/m/hrm", "director/ hrm",
}


@dataclass
class QueryAnalysis:
    intent: str  # "structural" or "specific"
    section_num: Optional[int] = None
    section_title: Optional[str] = None
    subsection_num: Optional[str] = None
    subsection_title: Optional[str] = None
    policy_name: Optional[str] = None
    procedure_code: Optional[str] = None
    procedure_name: Optional[str] = None


def _meta_dict(metadata) -> dict:
    if isinstance(metadata, dict):
        return metadata
    return metadata.__dict__ if hasattr(metadata, "__dict__") else {}


def normalize_text(text: str) -> str:
    text = text.lower().replace("&", " and ").replace("-", " ")
    return re.sub(r"[^a-z0-9\s]", " ", text)


def tokenize(text: str) -> Set[str]:
    return {w for w in normalize_text(text).split() if len(w) > 2}


def subsection_num_for(section_num: int, subsection_title: str) -> Optional[str]:
    key = subsection_title.lower()
    sub = SUBSECTION_TYPE_TO_NUM.get(key)
    if sub is None:
        return None
    return f"{section_num}.{sub}"


class DocumentStructureIndex:
    """Built once from ingested chunks — mirrors manual headers / TOC."""

    def __init__(self, chunks: List[Chunk]):
        self.sections: Dict[int, str] = {}
        self.procedures: List[Dict[str, Any]] = []
        self.policies: List[Dict[str, Any]] = []
        self._build(chunks)

    def _extract_procedure_name(self, text: str) -> Optional[str]:
        match = PROCEDURE_PREFIX_RE.search(text)
        return match.group(1).strip() if match else None

    def _build(self, chunks: List[Chunk]) -> None:
        seen_procedures: Set[str] = set()
        seen_policies: Set[Tuple[int, str]] = set()

        for chunk in chunks:
            meta = _meta_dict(chunk.metadata)
            section_num = meta.get("section_num")
            section_title = meta.get("section_title")
            if section_num and section_title:
                self.sections[int(section_num)] = section_title

            code = meta.get("procedure_code")
            if code and code not in seen_procedures:
                name = self._extract_procedure_name(chunk.text) or code
                seen_procedures.add(code)
                self.procedures.append(
                    {
                        "code": code,
                        "name": name,
                        "section_num": section_num,
                        "section_title": section_title,
                        "tokens": tokenize(name),
                        "norm_name": normalize_text(name),
                    }
                )

            policy_name = meta.get("policy_name")
            if (
                policy_name
                and meta.get("chunk_type") == "policy"
                and policy_name.lower() not in ACRONYM_POLICY_BLOCKLIST
                and len(policy_name) > 4
            ):
                key = (section_num, policy_name)
                if key not in seen_policies:
                    seen_policies.add(key)
                    self.policies.append(
                        {
                            "name": policy_name,
                            "section_num": section_num,
                            "section_title": section_title,
                            "subsection_num": meta.get("subsection_num"),
                            "tokens": tokenize(policy_name),
                            "norm_name": normalize_text(policy_name),
                        }
                    )

    def match_section(self, query: str) -> Optional[Tuple[int, str]]:
        q_norm = normalize_text(query)
        q_tokens = tokenize(query)
        best: Optional[Tuple[int, str]] = None
        best_score = 0.0

        for num, title in self.sections.items():
            t_tokens = tokenize(title)
            if not t_tokens:
                continue
            overlap = len(q_tokens & t_tokens)
            score = overlap / len(t_tokens)
            if overlap >= 2 or score >= 0.6:
                if score > best_score:
                    best_score = score
                    best = (num, title)

        # Handle "recruitment and selection" style phrasing
        if best is None:
            for num, title in self.sections.items():
                t_norm = normalize_text(title)
                if t_norm and t_norm in q_norm:
                    return num, title
                # All significant section words appear in query
                words = [w for w in t_norm.split() if len(w) > 3]
                if words and all(w in q_norm for w in words):
                    return num, title

        return best

    def match_subsection_type(self, query: str) -> Optional[str]:
        for sub_type in SUBSECTION_TYPES:
            if re.search(rf"\b{re.escape(sub_type)}\b", query, re.IGNORECASE):
                if sub_type == "purpose":
                    return "Purpose"
                return sub_type.title()
        return None

    def match_procedure(
        self, query: str, section_num: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        q_norm = normalize_text(query)
        q_tokens = tokenize(query)
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for proc in self.procedures:
            if section_num and proc["section_num"] != section_num:
                continue

            score = float(len(q_tokens & proc["tokens"]))
            norm_name = proc["norm_name"]

            # Phrase boosts for common paraphrases
            if "full time" in q_norm and "full time" in norm_name:
                score += 4
            if ("hire" in q_norm or "hiring" in q_norm) and "hire" in norm_name:
                score += 3
            if "vacancy" in q_norm and "vacancy" in norm_name:
                score += 2
            if "orientation" in q_norm and "orient" in norm_name:
                score += 3

            if score > best_score:
                best_score = score
                best = proc

        return best if best_score >= 3 else None

    def match_policy(
        self, query: str, section_num: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        q_norm = normalize_text(query)
        q_tokens = tokenize(query)
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for policy in self.policies:
            if section_num and policy["section_num"] != section_num:
                continue
            score = float(len(q_tokens & policy["tokens"]))
            if policy["norm_name"] in q_norm:
                score += 3
            if score > best_score:
                best_score = score
                best = policy

        return best if best_score >= 2 else None

    def analyze(self, query: str) -> QueryAnalysis:
        if SPECIFIC_FACT_RE.search(query):
            return QueryAnalysis(intent="specific")

        section = self.match_section(query)
        section_num, section_title = section if section else (None, None)

        subsection_title = self.match_subsection_type(query)
        subsection_num = (
            subsection_num_for(section_num, subsection_title)
            if section_num and subsection_title
            else None
        )

        procedure = self.match_procedure(query, section_num)
        policy = self.match_policy(query, section_num)

        # Prefer procedure when query mentions a specific procedure topic
        if procedure and re.search(r"\b(procedure|procedures|steps|process)\b", query, re.I):
            return QueryAnalysis(
                intent="structural",
                section_num=procedure["section_num"],
                section_title=procedure.get("section_title"),
                subsection_num=subsection_num_for(procedure["section_num"], "Procedures")
                if procedure.get("section_num")
                else None,
                subsection_title="Procedures",
                procedure_code=procedure["code"],
                procedure_name=procedure["name"],
            )

        # Subsection under a matched section (e.g. 1.3 Responsibilities)
        if section_num and subsection_title:
            return QueryAnalysis(
                intent="structural",
                section_num=section_num,
                section_title=section_title,
                subsection_num=subsection_num,
                subsection_title=subsection_title,
            )

        # Named policy overview
        if policy and re.search(
            r"\b(policy|policies|points|rules|requirements|what are the)\b", query, re.I
        ):
            return QueryAnalysis(
                intent="structural",
                section_num=policy["section_num"],
                section_title=policy.get("section_title"),
                subsection_num=policy.get("subsection_num"),
                subsection_title="Policies",
                policy_name=policy["name"],
            )

        return QueryAnalysis(intent="specific")


def analyze_query(query: str, structure_index: Optional[DocumentStructureIndex] = None) -> QueryAnalysis:
    if structure_index is None:
        return QueryAnalysis(intent="specific")
    return structure_index.analyze(query)
