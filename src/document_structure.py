"""
Map user questions to manual structure (sections, subsections, procedures, policies).

Intent classification — two intents:
  "structural"  — query targets a document header/TOC element; retrieval returns ALL
                  points under that exact sub-header (section, subsection, or named policy).
                  Requires either (a) a structural path match in the index AND the section
                  title/subsection/policy occurs prominently in the query, OR
                  (b) an explicit listing word ("list", "what are", "show all", …).
  "specific"    — everything else; retrieval uses hybrid vector+BM25 search.

Structural intent fires when the query contains:
  1. An explicit listing/enumeration signal word (STRUCTURAL_SIGNAL_RE), OR
  2. A matched section/policy/procedure path AND a non-factual opener.

It does NOT fire on bare factual openers ("how many / who / can I / am I eligible")
even when a section title is matched — those stay "specific".
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

# ── Factual guard: these openers always stay "specific" ──────────────────────
SPECIFIC_FACT_RE = re.compile(
    r"\b("
    r"how many|how much|how long|how do i|how to|how can|how should|"
    r"when |who is|who are|who can|who should|who handles|who approves|who manages|"
    r"can i|am i|is there|does |do i|eligible|"
    r"number of days|days of|weeks of|what happens if"
    r")\b",
    re.IGNORECASE,
)

# ── Explicit listing/enumeration signals → structural ────────────────────────
STRUCTURAL_SIGNAL_RE = re.compile(
    r"\b("
    r"list|show me|show all|outline|enumerate|summarize|tell me about|"
    r"all\s+(?:the\s+)?(?:steps|points|policies|procedures|sections|topics|rules|benefits|principles|challenges|methods)|"
    r"section\s+\d|table of contents|toc|"
    r"give me all|tell me all|what does .+? cover|what is covered"
    r")\b",
    re.IGNORECASE,
)

# ── Responsibility codes mistaken as policy headers during ingestion ──────────
ACRONYM_POLICY_BLOCKLIST = {
    "hrm", "hrs", "hro", "emp", "dm", "sys", "ed", "tl", "dir", "f&ad",
    "f&ad", "hr", "its", "to", "m", "tl/m/hrm", "director/ hrm",
}

# ── Document name fragments for cross-doc scoping ────────────────────────────
DOC_NAME_FRAGMENTS: Dict[str, str] = {
    # HR Manual (ML-HR-01)
    "hr": "ML-HR-01",
    "human resources": "ML-HR-01",
    "hr manual": "ML-HR-01",
    "hr policy": "ML-HR-01",
    "attendance": "ML-HR-01",
    "performance appraisal": "ML-HR-01",

    # Events Management (ML-ORG-01)
    "events": "ML-ORG-01",
    "event": "ML-ORG-01",
    "event management": "ML-ORG-01",
    "events manual": "ML-ORG-01",
    "events management": "ML-ORG-01",
    "travel booking": "ML-ORG-01",
    "venue selection": "ML-ORG-01",
    "accommodation": "ML-ORG-01",

    # Internal Communication Guideline (GL-ORG-01)
    "communication": "GL-ORG-01",
    "communications": "GL-ORG-01",
    "internal communication": "GL-ORG-01",
    "comms guideline": "GL-ORG-01",
    "virtual engagement": "GL-ORG-01",
    "microsoft teams": "GL-ORG-01",
    "teams": "GL-ORG-01",
    "zoom": "GL-ORG-01",
    "whatsapp": "GL-ORG-01",
    "face to face": "GL-ORG-01",
    "5ws": "GL-ORG-01",
    "5ws and 1h": "GL-ORG-01",
    "communication matrix": "GL-ORG-01",
    "periodic internal meetings": "GL-ORG-01",
    "technical day": "GL-ORG-01",
    "staff meeting": "GL-ORG-01",
}


@dataclass
class QueryAnalysis:
    intent: str                           # "structural" or "specific"
    doc_id: Optional[str] = None          # cross-document scoping (e.g. "ML-HR-01")
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


ACRONYM_EXPANSIONS = {
    "mou": "memorandum of understanding",
    "mom": "minutes of meeting",
    "eel": "emphnet electronic library",
    "pmo": "project management office",
    "ere": "emphnet resources engine",
    "dms": "document management system",
    "5ws": "why what when where who 5w",
    "1h": "how 1h",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[’'“”\"`]", "", text)  # Strip smart quotes without leaving space (w's -> ws)
    text = text.replace("&", " and ").replace("-", " ").replace(".", " ")
    for acr, exp in ACRONYM_EXPANSIONS.items():
        text = re.sub(rf"\b{acr}\b", f"{acr} {exp}", text)
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
    """Built once from ingested chunks — mirrors all document headers / TOC.

    Keyed by (doc_id, section_num) to avoid cross-document collisions.
    """

    def __init__(self, chunks: List[Chunk]):
        # sections: {(doc_id, section_num) -> section_title}
        self.sections: Dict[Tuple[Optional[str], int], str] = {}
        self.procedures: List[Dict[str, Any]] = []
        self.policies: List[Dict[str, Any]] = []
        self.subsections: List[Dict[str, Any]] = []
        self._build(chunks)

    def _extract_procedure_name(self, text: str) -> Optional[str]:
        match = PROCEDURE_PREFIX_RE.search(text)
        return match.group(1).strip() if match else None

    def _build(self, chunks: List[Chunk]) -> None:
        seen_procedures: Set[str] = set()
        seen_policies: Set[Tuple[Optional[str], Optional[int], str]] = set()
        seen_subsections: Set[Tuple[Optional[str], Optional[str]]] = set()

        for chunk in chunks:
            meta = _meta_dict(chunk.metadata)
            doc_id = meta.get("doc_id")
            section_num = meta.get("section_num")
            section_title = meta.get("section_title")
            if section_num and section_title:
                self.sections[(doc_id, int(section_num))] = section_title

            sub_num = meta.get("subsection_num")
            sub_title = meta.get("subsection_title")
            if sub_title and sub_title.lower() not in SUBSECTION_TYPES:
                key = (doc_id, sub_num or sub_title)
                if key not in seen_subsections:
                    seen_subsections.add(key)
                    self.subsections.append({
                        "num": sub_num,
                        "title": sub_title,
                        "doc_id": doc_id,
                        "section_num": section_num,
                        "section_title": section_title,
                        "norm_title": normalize_text(sub_title),
                        "tokens": tokenize(sub_title),
                    })

            code = meta.get("procedure_code")
            if code and code not in seen_procedures:
                name = self._extract_procedure_name(chunk.text) or code
                seen_procedures.add(code)
                self.procedures.append(
                    {
                        "code": code,
                        "name": name,
                        "doc_id": doc_id,
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
                key = (doc_id, section_num, policy_name)
                if key not in seen_policies:
                    seen_policies.add(key)
                    self.policies.append(
                        {
                            "name": policy_name,
                            "doc_id": doc_id,
                            "section_num": section_num,
                            "section_title": section_title,
                            "subsection_num": meta.get("subsection_num"),
                            "tokens": tokenize(policy_name),
                            "norm_name": normalize_text(policy_name),
                        }
                    )

    def match_custom_subsection(
        self, query: str, doc_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Match non-standard subsection titles (e.g. GL-ORG-01 channel names)."""
        q_norm = normalize_text(query)
        q_tokens = tokenize(query)
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for sub in self.subsections:
            if doc_id and sub.get("doc_id") and sub["doc_id"] != doc_id:
                continue
            norm_t = sub["norm_title"]
            if norm_t and norm_t in q_norm:
                return sub
            score = float(len(q_tokens & sub["tokens"]))
            if score > best_score and score >= 2:
                best_score = score
                best = sub
        return best


    def match_doc_id(self, query: str) -> Optional[str]:
        """Return a doc_id if the query names a specific document."""
        q_lower = query.lower()
        for fragment, doc_id in DOC_NAME_FRAGMENTS.items():
            if fragment in q_lower:
                return doc_id
        return None

    def match_section(
        self, query: str, doc_id: Optional[str] = None
    ) -> Optional[Tuple[Optional[str], int, str]]:
        """Return (doc_id, section_num, section_title) for the best match, or None."""
        q_norm = normalize_text(query)
        q_tokens = tokenize(query)
        best: Optional[Tuple[Optional[str], int, str]] = None
        best_score = 0.0

        for (d_id, num), title in self.sections.items():
            if doc_id and d_id and d_id != doc_id:
                continue
            t_tokens = tokenize(title)
            if not t_tokens:
                continue
            overlap = len(q_tokens & t_tokens)
            score = overlap / len(t_tokens)
            if overlap >= 2 or score >= 0.6:
                if score > best_score:
                    best_score = score
                    best = (d_id, num, title)

        if best is None:
            for (d_id, num), title in self.sections.items():
                if doc_id and d_id and d_id != doc_id:
                    continue
                t_norm = normalize_text(title)
                if t_norm and t_norm in q_norm:
                    return (d_id, num, title)
                words = [w for w in t_norm.split() if len(w) > 3]
                if words and all(w in q_norm for w in words):
                    return (d_id, num, title)

        return best

    def match_subsection_type(self, query: str) -> Optional[str]:
        for sub_type in SUBSECTION_TYPES:
            if re.search(rf"\b{re.escape(sub_type)}\b", query, re.IGNORECASE):
                if sub_type == "purpose":
                    return "Purpose"
                return sub_type.title()
        return None

    def match_procedure(
        self, query: str, section_num: Optional[int] = None, doc_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        q_norm = normalize_text(query)
        q_tokens = tokenize(query)
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for proc in self.procedures:
            if section_num and proc["section_num"] != section_num:
                continue
            if doc_id and proc.get("doc_id") and proc["doc_id"] != doc_id:
                continue

            score = float(len(q_tokens & proc["tokens"]))
            norm_name = proc["norm_name"]

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
        self, query: str, section_num: Optional[int] = None, doc_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        q_norm = normalize_text(query)
        q_tokens = tokenize(query)
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for policy in self.policies:
            if section_num and policy["section_num"] != section_num:
                continue
            if doc_id and policy.get("doc_id") and policy["doc_id"] != doc_id:
                continue
            score = float(len(q_tokens & policy["tokens"]))
            if policy["norm_name"] in q_norm:
                score += 3
            if score > best_score:
                best_score = score
                best = policy

        return best if best_score >= 2 else None

    def analyze(self, query: str) -> QueryAnalysis:
        # Step 1 — factual guard (always specific)
        if SPECIFIC_FACT_RE.search(query):
            return QueryAnalysis(intent="specific", doc_id=self.match_doc_id(query))

        matched_doc_id = self.match_doc_id(query)
        has_explicit_listing = bool(STRUCTURAL_SIGNAL_RE.search(query))

        # ── Document-level purpose/manual queries ───────────────────────────
        if re.search(r"\bpurpose of (?:the )?(?:internal communication |comms )?(?:manual|guideline|document)\b", query, re.I):
            return QueryAnalysis(
                intent="structural",
                doc_id=matched_doc_id or "GL-ORG-01",
                policy_name="Purpose",
            )

        # ── Periodic/status meetings query → Section 04 structural ──────────
        if re.search(
            r"\b(periodic\s+(internal\s+)?meetings|types?\s+of\s+(status\s+)?meetings|status\s+meetings|meeting types)\b",
            query, re.I
        ):
            return QueryAnalysis(
                intent="structural",
                doc_id="GL-ORG-01",
                section_num=4,
                section_title="GHD|EMPHNET Periodic Internal Meetings",
            )

        # ── Section 01 (Effective Communication) topic queries ───────────────
        if re.search(
            r"\b(three.?step process|benefits of effective|main principles of effective|"
            r"major challenges in communication|major challenges|major communication challenges|"
            r"effective communication|five\s*ws|5ws)\b",
            query, re.I
        ) and not re.search(r"\bmicrosoft teams|zoom|email|whatsapp|face.to.face\b", query, re.I):
            return QueryAnalysis(
                intent="structural",
                doc_id="GL-ORG-01",
                section_num=1,
                section_title="Effective Communication",
            )

        section = self.match_section(query, doc_id=matched_doc_id)
        if section:
            sec_doc_id, section_num, section_title = section
        else:
            sec_doc_id, section_num, section_title = None, None, None

        eff_doc_id = matched_doc_id or sec_doc_id

        subsection_title = self.match_subsection_type(query)
        custom_sub = self.match_custom_subsection(query, doc_id=eff_doc_id)
        # Prioritize custom_sub (e.g. 2.2 Face-to-Face Meeting) over generic terms like "Purpose"
        if custom_sub:
            if not subsection_title or subsection_title.lower() in ("purpose", "scope", "audience", "responsibility", "responsibilities"):
                subsection_title = custom_sub["title"]
                subsection_num = custom_sub["num"]
                section_num = custom_sub["section_num"] if custom_sub.get("section_num") else section_num
                section_title = custom_sub["section_title"] if custom_sub.get("section_title") else section_title
                eff_doc_id = eff_doc_id or custom_sub.get("doc_id")
            else:
                subsection_num = (
                    subsection_num_for(section_num, subsection_title)
                    if section_num and subsection_title
                    else None
                )
        else:
            subsection_num = (
                subsection_num_for(section_num, subsection_title)
                if section_num and subsection_title
                else None
            )

        procedure = self.match_procedure(query, section_num, doc_id=eff_doc_id)
        policy = self.match_policy(query, section_num, doc_id=eff_doc_id)


        # ── Procedure match + explicit procedure signal ───────────────────────
        if procedure and re.search(r"\b(procedure|procedures|steps|process)\b", query, re.I):
            return QueryAnalysis(
                intent="structural",
                doc_id=procedure.get("doc_id") or eff_doc_id,
                section_num=procedure["section_num"],
                section_title=procedure.get("section_title"),
                subsection_num=subsection_num_for(procedure["section_num"], "Procedures")
                if procedure.get("section_num")
                else None,
                subsection_title="Procedures",
                procedure_code=procedure["code"],
                procedure_name=procedure["name"],
            )

        # ── Subsection under a section or direct subsection query ─────────────────────────────
        # Structural if: listing signal OR the subsection name appears explicitly in query
        if subsection_title:
            sub_norm = normalize_text(subsection_title)
            q_norm = normalize_text(query)
            subsection_in_query = (sub_norm in q_norm) or bool(
                re.search(rf"\b{re.escape(subsection_title)}\b", query, re.I)
            )
            if has_explicit_listing or subsection_in_query:

                return QueryAnalysis(
                    intent="structural",
                    doc_id=eff_doc_id,
                    section_num=section_num,
                    section_title=section_title,
                    subsection_num=subsection_num,
                    subsection_title=subsection_title,
                )


        # ── Named policy match ───────────────────────────────────────────────
        # Structural if: listing/policy signal OR policy name appears prominently
        if policy and re.search(
            r"\b(policy|policies|points|rules|requirements|what are the|overview)\b",
            query, re.I,
        ):
            return QueryAnalysis(
                intent="structural",
                doc_id=policy.get("doc_id") or eff_doc_id,
                section_num=policy["section_num"],
                section_title=policy.get("section_title"),
                subsection_num=policy.get("subsection_num"),
                subsection_title="Policies",
                policy_name=policy["name"],
            )

        # ── Section match + explicit listing signal ──────────────────────────
        if section_num and has_explicit_listing:
            return QueryAnalysis(
                intent="structural",
                doc_id=eff_doc_id,
                section_num=section_num,
                section_title=section_title,
            )

        return QueryAnalysis(intent="specific", doc_id=eff_doc_id)


def analyze_query(query: str, structure_index: Optional[DocumentStructureIndex] = None) -> QueryAnalysis:
    if structure_index is None:
        return QueryAnalysis(intent="specific")
    return structure_index.analyze(query)

