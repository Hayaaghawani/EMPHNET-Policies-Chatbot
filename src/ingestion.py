"""
PDF Ingestion Pipeline for EMPHNET Policy Documents

Extracts text page-by-page from any GHD|EMPHNET policy manual or guideline,
parses the document hierarchy using per-document configuration, and produces
self-contained chunks suitable for multi-document RAG retrieval.

Supported documents:
  - ML-HR-01  Human Resources Policies & Procedures Manual
  - ML-ORG-01 Events Management Policies and Procedures Manual
  - GL-ORG-01 Internal Communication Guideline
"""

import hashlib
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Standard subsection vocabulary shared across all manuals
SUBSECTION_TYPES = (
    "Purpose", "Scope", "Responsibilities", "Policies",
    "Procedures", "Related Documents", "References"
)


@dataclass
class DocumentConfig:
    """Per-document parser configuration — keeps all doc-specific knowledge here."""

    # Identity
    doc_id: str                   # e.g. "ML-HR-01"
    doc_type: str                 # "policy_manual" | "procedure_manual" | "guideline"
    document_title: str           # Human-readable title for source display
    manual_code: str              # e.g. "ML-HR-01"
    version: str                  # e.g. "V.01"
    effective_date: str           # e.g. "August 7, 2022"
    language: str = "en"

    # Regex strings (compiled at parse time) — defaults work for standard manuals
    section_re_str: str = r'^Section\s*0?\s*(\d{1,2})\s*:\s*(.+)$'

    subsection_re_str: Optional[str] = None   # None → auto-build from SUBSECTION_TYPES
    procedure_re_str: str = r'^Procedure\s+(\d+)\s*:\s*(.+)$'
    procedure_code_template: Optional[str] = None  # e.g. "{code}.P{num:02d}"
    annex_re_str: str = r'^(?:ANNEX|Annex)\s+(\d{1,2})\b'
    attachment_re_str: str = r'^(?:ATTACHMENT|Attachment)\s+(\d{1,2})\b'
    appendix_re_str: Optional[str] = None   # e.g. for GL-ORG-01

    # Header/footer line patterns to strip (list of raw regex strings)
    header_patterns: List[str] = field(default_factory=list)

    # Behaviour flags
    enable_policy_headers: bool = True   # False for guidelines (no named policies)
    enable_procedures: bool = True       # False for GL-ORG-01
    enable_acronym_table: bool = False   # True when doc has Acronym/Description table
    enable_channel_cards: bool = False   # True for GL-ORG-01 channel cards

    # Known policy topic names for _is_policy_header (boosts precision)
    known_policy_topics: frozenset = field(default_factory=frozenset)


KNOWN_POLICY_TOPICS = {
    "employee attendance", "personal leave and annual vacations", "business leave",
    "working from anywhere (wfa)", "unpaid vacation", "sick leave", "official holidays",
    "hajj vacation", "emergency vacation", "paternity leave", "maternity leave",
    "general", "disciplinary actions", "grievance", "performance management",
    "recruitment & selection", "training & development",
}


@dataclass
class DocumentMetadata:
    """Document-level metadata populated from DocumentConfig. Kept for backward compat."""
    doc_id: str = "ML-HR-01"
    doc_type: str = "policy_manual"
    document_title: str = "HR Policies & Procedures Manual"
    manual_code: str = "ML-HR-01"
    version: str = "V.01"
    effective_date: str = "August 7, 2022"
    language: str = "en"


@dataclass
class ChunkMetadata(DocumentMetadata):
    # Structural location
    section_num: Optional[int] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None          # Full breadcrumb: "S01 > 1.3 Responsibilities > Sick Leave"
    subsection_num: Optional[str] = None
    subsection_title: Optional[str] = None
    policy_name: Optional[str] = None
    procedure_code: Optional[str] = None
    parent_header: Optional[str] = None         # Immediate parent section/subsection title

    # Source
    page_number: int = 0

    # Content classification
    chunk_type: str = "text"                    # text|section_intro|subsection|policy|procedure|annex|attachment|definition|table
    content_type: str = "narrative"             # narrative|definition|table_row|list
    has_list: bool = False                      # True if body contains numbered/bulleted items

    # Split siblings (for large chunks split by RecursiveCharacterTextSplitter)
    split_group_id: Optional[str] = None
    split_part: Optional[int] = None
    split_total: Optional[int] = None



@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata

    def to_dict(self):
        return {"text": self.text, "metadata": asdict(self.metadata)}


class HeaderFooterRemover:
    """Strips recurring header/footer noise from extracted PDF text.

    The default patterns cover ML-HR-01. Pass ``extra_patterns`` from
    ``DocumentConfig.header_patterns`` to extend for other documents.
    """

    # Patterns common to all GHD|EMPHNET documents
    _BASE_PATTERNS = [
        r"^Global Health Development[\s|].*EMPHNET\s*$",
        r"^GHD[\s|]+EMPHNET\s*$",
        r"^Eastern Mediterranean Public Health Network \(EMPHNET\)\s*$",
        r"^Page \d+ of \d+\s*$",
        r"^DocuSign Envelope ID:\s*[A-Z0-9-]+\s*$",
    ]

    def __init__(self, extra_patterns: Optional[List[str]] = None):
        self._patterns = list(self._BASE_PATTERNS) + (extra_patterns or [])

    def clean(self, text: str) -> str:
        lines = []
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) <= 120 and any(
                re.search(p, stripped, re.IGNORECASE) for p in self._patterns
            ):
                continue
            lines.append(stripped)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Acronym/definition two-column table parser
# ---------------------------------------------------------------------------

ACRONYM_ROW_RE = re.compile(r'^([A-Za-z0-9/&]{2,12})\s+(.{4,})$')
CHANNEL_CARD_LABELS = frozenset({
    "Audience", "Purpose", "Best For", "Responsibility", "Key Considerations"
})
CHANNEL_CARD_RE = re.compile(
    r'^(Audience|Purpose|Best For|Responsibility|Key Considerations)\s+(.+)$'
)
TABLE_HEADER_RE = re.compile(r'^Table \d+\.\d+:', re.IGNORECASE)


class PolicyManualParser:
    """Parse a GHD|EMPHNET policy manual or guideline using the supplied DocumentConfig.

    This replaces the legacy ``HierarchicalPDFParser`` which was hardcoded to ML-HR-01.
    All document-specific knowledge lives in ``DocumentConfig``; this class contains
    only structural parsing logic.
    """

    NUMBERED_ITEM_RE = re.compile(r'^(\d+)\.\s+')
    TOC_LINE_RE = re.compile(r'\.{4,}')
    PROCESS_MAP_RE = re.compile(r'^Process Map', re.IGNORECASE)
    STEP_HEADER_RE = re.compile(r'^Step\s+Action\s+Responsibility\s*$', re.IGNORECASE)

    def __init__(self, pdf_path: str, doc_config: DocumentConfig):
        self.pdf_path = Path(pdf_path)
        self.doc_config = doc_config
        self.reader = PdfReader(str(self.pdf_path))
        self._remover = HeaderFooterRemover(extra_patterns=doc_config.header_patterns)

        # Compile regexes from config
        self.SECTION_RE = re.compile(doc_config.section_re_str, re.IGNORECASE)

        if doc_config.subsection_re_str:
            self.SUBSECTION_RE = re.compile(doc_config.subsection_re_str, re.IGNORECASE)
        else:
            self.SUBSECTION_RE = re.compile(
                r'^(\d{1,2})\.(\d+)\s+(' + '|'.join(map(re.escape, SUBSECTION_TYPES)) + r')',
                re.IGNORECASE,
            )

        if doc_config.appendix_re_str:
            self.APPENDIX_RE = re.compile(doc_config.appendix_re_str, re.IGNORECASE)
        else:
            self.APPENDIX_RE = None

        self.ANNEX_RE = re.compile(r'^(?:Annex|ANNEX)\s+[A-Z\d]+', re.IGNORECASE)
        self.ATTACHMENT_RE = re.compile(r'^(?:Attachment|ATTACHMENT)\s+[A-Z\d]+', re.IGNORECASE)
        self.PROCEDURE_START_RE = re.compile(
            r'^(?:Procedure|PROCEDURE)\s+(\d{1,2})\s*:\s*(.+)$', re.IGNORECASE
        )

        # ML-HR-01 style inline procedure code lines (e.g. "ML-HR-01.P03 ...")
        manual_code_escaped = re.escape(doc_config.manual_code)
        self.PROCEDURE_CODE_RE = re.compile(
            rf'^({manual_code_escaped}\.[A-Z]\d+)\s+(.+)$', re.IGNORECASE
        )

        logger.info(
            f"Loaded PDF: {self.pdf_path.name} ({len(self.reader.pages)} pages) "
            f"[{doc_config.doc_id}]"
        )


    def extract_text_by_page(self) -> List[Tuple[int, str]]:
        pages = []
        for page_num, page in enumerate(self.reader.pages, 1):
            text = self._remover.clean(page.extract_text() or "")
            pages.append((page_num, text))
        logger.info(f"Extracted text from {len(pages)} pages [{self.doc_config.doc_id}]")
        return pages


    @staticmethod
    def _is_toc_page(text: str) -> bool:
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        if not lines:
            return True
        toc_lines = sum(1 for ln in lines if PolicyManualParser.TOC_LINE_RE.search(ln))
        # Contents pages are mostly dot-leader lines
        if toc_lines >= 3 or (len(lines) <= 6 and toc_lines >= 1):
            return True
        # Early manual pages that are only section index lists
        if any(re.match(r'^\d-\s+\w', ln) for ln in lines) and len(lines) < 15:
            return True
        return False

    def _is_policy_header(self, line: str, in_policies: bool) -> bool:
        """Return True if *line* is a named policy header or topic header."""
        if not self.doc_config.enable_policy_headers:
            return False
        known = self.doc_config.known_policy_topics or KNOWN_POLICY_TOPICS
        if line.lower() in known:
            return True
        if not in_policies:
            return False
        if len(line) < 3 or len(line) > 70:
            return False
        if self.NUMBERED_ITEM_RE.match(line):
            return False
        if self.SUBSECTION_RE.match(line):
            return False
        if self.SECTION_RE.match(line):
            return False
        if self.PROCEDURE_START_RE.match(line):
            return False
        if line.upper() == line and len(line) <= 5:
            return False
        # Title-case standalone header (e.g. "Hajj Vacation", "Sick Leave")
        if re.match(r'^[A-Z][A-Za-z\s&()/-]+$', line) and not line.endswith('.'):
            words = line.split()
            if 1 <= len(words) <= 7:
                return True
        return False



    @staticmethod
    def _build_prefix(ctx: dict) -> str:
        parts = []
        sec = ctx.get("section_num")
        title = ctx.get("section_title")
        if sec and title:
            parts.append(f"Section {sec:02d}: {title}")
        sub_num = ctx.get("subsection_num")
        sub_title = ctx.get("subsection_title")
        if sub_num and sub_title:
            parts.append(f"{sub_num} {sub_title}")
        policy = ctx.get("policy_name")
        if policy:
            parts.append(policy)
        proc = ctx.get("procedure_name")
        if proc:
            parts.append(proc)
        code = ctx.get("procedure_code")
        if code:
            parts.append(code)
        return " | ".join(parts)

    @staticmethod
    def _detect_content_type(body: str, chunk_type: str) -> str:
        """Infer content_type from body text and structural chunk_type."""
        if chunk_type in ("definition",):
            return "definition"
        if chunk_type == "table":
            return "table_row"
        # Detect list: body contains 2+ numbered items
        if len(re.findall(r'^\d+\.', body, re.MULTILINE)) >= 2:
            return "list"
        return "narrative"

    @staticmethod
    def _has_list(body: str) -> bool:
        return bool(re.search(r'(^\d+\.\s|\n\d+\.\s|^-\s|\n-\s)', body))

    @staticmethod
    def _build_section_path(ctx: dict, policy_name: Optional[str] = None) -> Optional[str]:
        """Build a breadcrumb string for section_path metadata."""
        parts = []
        sec = ctx.get("section_num")
        title = ctx.get("section_title")
        if sec and title:
            parts.append(f"S{sec:02d}: {title}")
        sub_num = ctx.get("subsection_num")
        sub_title = ctx.get("subsection_title")
        if sub_num and sub_title:
            parts.append(f"{sub_num} {sub_title}")
        pname = policy_name or ctx.get("policy_name")
        if pname:
            parts.append(pname)
        proc = ctx.get("procedure_name")
        if proc:
            parts.append(proc)
        return " > ".join(parts) if parts else None

    def _make_chunk(
        self,
        body: str,
        ctx: dict,
        chunk_type: str,
        page_number: int,
        policy_name: Optional[str] = None,
        procedure_code: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Optional[Chunk]:
        body = body.strip()
        if not body or len(body) < 20:
            return None
        if self.PROCESS_MAP_RE.search(body[:80]):
            return None
        if self.TOC_LINE_RE.search(body) and body.count('....') >= 2:
            return None

        eff_policy = policy_name or ctx.get("policy_name")
        prefix = self._build_prefix({**ctx, "policy_name": eff_policy})
        text = f"[{prefix}]\n{body}" if prefix else body

        eff_content_type = content_type or self._detect_content_type(body, chunk_type)
        cfg = self.doc_config

        # Determine parent_header for structural navigation
        parent_header = ctx.get("subsection_title") or ctx.get("section_title")

        metadata = ChunkMetadata(
            # Document identity (from config)
            doc_id=cfg.doc_id,
            doc_type=cfg.doc_type,
            document_title=cfg.document_title,
            manual_code=cfg.manual_code,
            version=cfg.version,
            effective_date=cfg.effective_date,
            language=cfg.language,
            # Structural location
            section_num=ctx.get("section_num"),
            section_title=ctx.get("section_title"),
            section_path=self._build_section_path(ctx, eff_policy),
            subsection_num=ctx.get("subsection_num"),
            subsection_title=ctx.get("subsection_title"),
            policy_name=eff_policy,
            procedure_code=procedure_code or ctx.get("procedure_code"),
            parent_header=parent_header,
            # Source
            page_number=page_number,
            # Content classification
            chunk_type=chunk_type,
            content_type=eff_content_type,
            has_list=self._has_list(body),
        )
        return Chunk(text=text, metadata=metadata)



    def _split_large_chunk(self, chunk: Chunk, max_size: int = 2000, overlap: int = 200) -> List[Chunk]:
        if len(chunk.text) <= max_size:
            return [chunk]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        parts = splitter.split_text(chunk.text)
        if len(parts) <= 1:
            return [chunk]

        meta = chunk.metadata
        group_key = "|".join(
            str(x or "")
            for x in (
                getattr(meta, "doc_id", ""),
                meta.section_num,
                meta.subsection_num,
                meta.policy_name,
                meta.procedure_code,
                meta.chunk_type,
            )
        )
        group_id = hashlib.md5(group_key.encode()).hexdigest()[:12]
        topic_label = meta.policy_name or meta.procedure_code or meta.subsection_title or "this topic"

        result = []
        for i, part in enumerate(parts, 1):
            part_meta = ChunkMetadata(
                # Document identity
                doc_id=getattr(meta, "doc_id", "ML-HR-01"),
                doc_type=getattr(meta, "doc_type", "policy_manual"),
                document_title=meta.document_title,
                manual_code=meta.manual_code,
                version=meta.version,
                effective_date=meta.effective_date,
                language=meta.language,
                # Structural location
                section_num=meta.section_num,
                section_title=meta.section_title,
                section_path=getattr(meta, "section_path", None),
                subsection_num=meta.subsection_num,
                subsection_title=meta.subsection_title,
                policy_name=meta.policy_name,
                procedure_code=meta.procedure_code,
                parent_header=getattr(meta, "parent_header", None),
                # Source
                page_number=meta.page_number,
                # Content classification
                chunk_type=meta.chunk_type,
                content_type=getattr(meta, "content_type", "narrative"),
                has_list=getattr(meta, "has_list", False),
                # Split tracking
                split_group_id=group_id,
                split_part=i,
                split_total=len(parts),
            )
            if i == 1:
                body = part + f"\n[Part 1 of {len(parts)} — more points continue in related chunks]"
            else:
                body = (
                    f"[Continued: {topic_label} — Part {i} of {len(parts)}]\n{part}"
                )
            result.append(Chunk(text=body, metadata=part_meta))
        return result


    def parse_sections(self, pages_text: List[Tuple[int, str]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        cfg = self.doc_config
        ctx = {
            "section_num": None,
            "section_title": None,
            "subsection_num": None,
            "subsection_title": None,
            "policy_name": None,
            "procedure_name": None,
            "procedure_code": None,
        }

        block_lines: List[str] = []
        block_type = "text"
        block_page = 1
        block_policy: Optional[str] = None
        block_procedure_code: Optional[str] = None
        block_content_type: Optional[str] = None
        in_procedure = False
        skip_until_section = False

        # State for acronym/definition table parsing
        in_acronym_table = False
        acronym_pairs: List[str] = []

        # State for GL-ORG-01 channel card parsing
        in_channel_card = False
        channel_card_lines: List[str] = []
        channel_card_label: Optional[str] = None

        def flush():
            nonlocal block_lines, block_type, block_policy, block_procedure_code
            nonlocal in_procedure, block_content_type
            if not block_lines:
                return
            body = '\n'.join(block_lines)
            chunk = self._make_chunk(
                body, ctx, block_type, block_page,
                policy_name=block_policy,
                procedure_code=block_procedure_code,
                content_type=block_content_type,
            )
            if chunk:
                chunks.extend(self._split_large_chunk(chunk))
            block_lines = []
            block_policy = None
            block_procedure_code = None
            block_content_type = None
            in_procedure = False

        def flush_acronym_table():
            nonlocal in_acronym_table, acronym_pairs
            if not acronym_pairs:
                in_acronym_table = False
                return
            body = '\n'.join(acronym_pairs)
            chunk = self._make_chunk(
                body, ctx, "definition", block_page,
                content_type="definition",
            )
            if chunk:
                chunks.append(chunk)
            acronym_pairs = []
            in_acronym_table = False

        def flush_channel_card():
            nonlocal in_channel_card, channel_card_lines
            if not channel_card_lines:
                in_channel_card = False
                return
            body = '\n'.join(channel_card_lines)
            chunk = self._make_chunk(
                body, ctx, "definition", block_page,
                content_type="definition",
            )
            if chunk:
                chunks.append(chunk)
            channel_card_lines = []
            in_channel_card = False

        for page_num, page_text in pages_text:
            if self._is_toc_page(page_text):
                continue

            in_policies = (
                (ctx.get("subsection_title") or "").lower() == "policies"
            )

            for line in page_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # ------------------------------------------------------------------
                # Acronym table detection (ML-ORG-01, GL-ORG-01)
                # ------------------------------------------------------------------
                if cfg.enable_acronym_table:
                    # Table header triggers: "Acronym  Description" or "Definitions and Acronyms"
                    if re.match(r'^(?:Acronym|Definitions and Acronyms|Acronyms)\s', line, re.IGNORECASE) \
                            or (re.match(r'^Acronym\s+Description\s*$', line, re.IGNORECASE)):
                        flush()
                        flush_acronym_table()
                        in_acronym_table = True
                        continue
                    if in_acronym_table:
                        # A section or subsection header always ends the acronym table
                        if self.SECTION_RE.match(line) or self.SUBSECTION_RE.match(line):
                            flush_acronym_table()
                            # fall through to Section/Subsection handling below
                        else:
                            m = ACRONYM_ROW_RE.match(line)
                            if m:
                                term = m.group(1).strip()
                                desc = m.group(2).strip()
                                acronym_pairs.append(f"**{term}**: {desc} ({term} stands for {desc})")
                                continue
                            else:
                                # Non-matching line ends the table
                                flush_acronym_table()

                # ------------------------------------------------------------------
                # Section
                # ------------------------------------------------------------------
                section_m = self.SECTION_RE.match(line)
                if section_m:
                    flush()
                    flush_channel_card()
                    flush_acronym_table()
                    ctx["section_num"] = int(section_m.group(1))
                    ctx["section_title"] = section_m.group(2).strip()
                    ctx["subsection_num"] = None
                    ctx["subsection_title"] = None
                    ctx["policy_name"] = None
                    ctx["procedure_name"] = None
                    ctx["procedure_code"] = None
                    block_type = "section_intro"
                    block_page = page_num
                    logger.info(f"Section {ctx['section_num']:02d}: {ctx['section_title']} [{cfg.doc_id}]")
                    continue

                # ------------------------------------------------------------------
                # Subsection (supports 2-level N.N and 3-level N.N.N)
                # ------------------------------------------------------------------
                sub_m = self.SUBSECTION_RE.match(line)
                if sub_m:
                    flush()
                    flush_channel_card()
                    flush_acronym_table()
                    groups = sub_m.groups()
                    sec = groups[0]
                    sub1 = groups[1]
                    sub_raw = (groups[-1] or "").strip()
                    if len(groups) >= 4 and groups[2]:
                        sub2 = groups[2]
                        sub_num = f"{sec}.{sub1}.{sub2}"
                    else:
                        sub_num = f"{sec}.{sub1}"
                    sub_type = sub_raw.title() if sub_raw.lower() == "purpose" else sub_raw
                    ctx["subsection_num"] = sub_num
                    ctx["subsection_title"] = sub_type
                    ctx["policy_name"] = None
                    ctx["procedure_name"] = None
                    ctx["procedure_code"] = None
                    block_type = "subsection"
                    block_page = page_num
                    logger.info(f"  {ctx['subsection_num']} {sub_type}")
                    continue

                # ------------------------------------------------------------------
                # Sub-Topic Bullet Headings (e.g. ➢ Monthly Technical Day)
                # ------------------------------------------------------------------
                sub_topic_m = re.match(r'^[➢\u27a2]\s*([A-Za-z0-9\s|&\’\'\-\–\—\(\)]+?)(?:[:;.]|\s*$)', line)
                if sub_topic_m and len(sub_topic_m.group(1).strip()) > 3:
                    flush()
                    flush_channel_card()
                    topic_title = sub_topic_m.group(1).strip()
                    ctx["policy_name"] = topic_title
                    block_type = "sub_topic"
                    block_page = page_num
                    logger.info(f"    Bullet Sub-Topic: {topic_title}")
                    # Keep the header in block_lines for context
                    block_lines.append(line)
                    continue

                # ------------------------------------------------------------------
                # GL-ORG-01 channel card detection (Audience/Purpose/Best For...)
                # ------------------------------------------------------------------
                if cfg.enable_channel_cards:
                    m = CHANNEL_CARD_RE.match(line)
                    if m:
                        label = m.group(1)
                        value = m.group(2).strip()
                        if label == "Audience":
                            # New card starts
                            flush()
                            flush_channel_card()
                            in_channel_card = True
                        if in_channel_card:
                            channel_card_lines.append(f"**{label}**: {value}")
                            continue
                    elif in_channel_card:
                        # Continuation line for the last card field
                        channel_card_lines.append(line)
                        continue



                # ------------------------------------------------------------------
                # Procedure
                # ------------------------------------------------------------------
                if cfg.enable_procedures:
                    proc_m = self.PROCEDURE_START_RE.match(line)
                    if proc_m:
                        flush()
                        flush_channel_card()
                        proc_num = proc_m.group(1)
                        proc_name = proc_m.group(2).strip()
                        ctx["procedure_name"] = f"Procedure {proc_num}: {proc_name}"
                        # Use config template or fall back to generic
                        if cfg.procedure_code_template:
                            ctx["procedure_code"] = cfg.procedure_code_template.format(
                                code=cfg.manual_code, num=int(proc_num)
                            )
                        else:
                            ctx["procedure_code"] = f"{cfg.manual_code}.P{int(proc_num):02d}"
                        ctx["policy_name"] = None
                        block_type = "procedure"
                        block_page = page_num
                        in_procedure = True
                        block_procedure_code = ctx["procedure_code"]
                        block_lines = []
                        continue

                    if in_procedure and self.PROCEDURE_CODE_RE.match(line):
                        # Inline procedure code metadata — skip
                        continue

                if self.STEP_HEADER_RE.match(line):
                    continue

                # ------------------------------------------------------------------
                # Annex / Attachment / Appendix
                # ------------------------------------------------------------------
                annex_m = self.ANNEX_RE.match(line)
                attach_m = self.ATTACHMENT_RE.match(line)
                appendix_m = self.APPENDIX_RE.match(line) if self.APPENDIX_RE else None
                if annex_m or attach_m or appendix_m:
                    flush()
                    flush_channel_card()
                    if annex_m:
                        block_type = "annex"
                    elif appendix_m:
                        block_type = "annex"   # treat appendix like annex
                    else:
                        block_type = "attachment"
                    block_page = page_num
                    block_lines = [line]
                    continue

                # ------------------------------------------------------------------
                # Named policy header (inside Policies subsection)
                # ------------------------------------------------------------------
                if self._is_policy_header(line, in_policies):
                    flush()
                    block_policy = line
                    ctx["policy_name"] = line
                    block_type = "policy"
                    block_page = page_num
                    continue

                # ------------------------------------------------------------------
                # Responsibility codes on their own line (EMP, DM, HR, SYS)
                # ------------------------------------------------------------------
                if in_procedure and re.match(r'^[A-Z]{2,5}$', line) and block_lines:
                    block_lines[-1] = f"{block_lines[-1]} (Responsibility: {line})"
                    continue

                if not block_lines:
                    block_page = page_num

                block_lines.append(line)

        flush()
        flush_channel_card()
        flush_acronym_table()
        logger.info(f"Parsed {len(chunks)} chunks from PDF [{cfg.doc_id}]")
        return chunks

    def parse(self) -> List[Chunk]:
        return self.parse_sections(self.extract_text_by_page())


# Backward-compatibility alias so existing code that imports HierarchicalPDFParser still works
HierarchicalPDFParser = PolicyManualParser


def save_chunks_to_file(chunks: List[Chunk], output_path: str) -> None:
    output = [c.to_dict() for c in chunks]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


# ---------------------------------------------------------------------------
# Document configs — one per supported document
# ---------------------------------------------------------------------------

HR_DOC_CONFIG = DocumentConfig(
    doc_id="ML-HR-01",
    doc_type="policy_manual",
    document_title="Human Resources Policies & Procedures Manual",
    manual_code="ML-HR-01",
    version="V.01",
    effective_date="August 7, 2022",
    header_patterns=[
        r"^ML-HR-01\s*,?\s*V\.0?1\s*$",
        r"^V\.0?1\s*$",
    ],
    enable_policy_headers=True,
    enable_procedures=True,
    enable_acronym_table=False,
    enable_channel_cards=False,
    known_policy_topics=frozenset({
        "employee attendance", "personal leave and annual vacations", "business leave",
        "working from anywhere (wfa)", "unpaid vacation", "sick leave", "official holidays",
        "hajj vacation", "emergency vacation", "paternity leave", "maternity leave",
        "general", "disciplinary actions", "grievance", "performance management",
        "recruitment & selection", "training & development",
    }),
)

EVENTS_DOC_CONFIG = DocumentConfig(
    doc_id="ML-ORG-01",
    doc_type="procedure_manual",
    document_title="Events Management Policies and Procedures Manual",
    manual_code="ML-ORG-01",
    version="V.01",
    effective_date="November 16, 2022",
    header_patterns=[
        r"^ML-ORG-01,?\s*V\.0?1\s*Page \d+ of \d+\s*$",
        r"^ML-ORG-01,?\s*V\.0?1\s*$",
    ],
    enable_policy_headers=True,
    enable_procedures=True,
    enable_acronym_table=True,
    enable_channel_cards=False,
    known_policy_topics=frozenset({
        "planning", "venue selection", "travel booking", "accommodation",
        "agenda", "training material", "name tags/ table tags", "photography",
        "social program", "certificates and plaque", "event items", "payments",
        "reporting", "before launching the event", "during the event",
        "delivering payments to participants and facilitators",
    }),
)

COMMS_DOC_CONFIG = DocumentConfig(
    doc_id="GL-ORG-01",
    doc_type="guideline",
    document_title="Internal Communication Guideline",
    manual_code="GL-ORG-01",
    version="V.01",
    effective_date="April 10, 2023",
    # GL-ORG-01 subsections support 2-level N.N and 3-level N.N.N
    subsection_re_str=r'^(\d{1,2})\.(\d+)(?:\.(\d+))?\.?\s+(.+)$',
    appendix_re_str=r'^(?:Appendix|APPENDIX|Template)\s+\d+\b',
    header_patterns=[
        r"^GL-ORG-01,?\s*V\.0?1\s*Page \d+ of \d+\s*$",
        r"^GL-ORG-01,?\s*V\.0?1\s*$",
    ],
    enable_policy_headers=True,   # Enable topic headers for guideline sub-headings
    enable_procedures=False,
    enable_acronym_table=True,
    enable_channel_cards=True,
    known_policy_topics=frozenset({
        "benefits of effective internal communication",
        "main principles of effective internal communication",
        "three-step process for effective communication",
        "identify communication requirements",
        "identify the 5ws (why, what, when, where, who) and 1h (how)",
        "identify the 5ws and 1h",
        "identify and accommodate the enterprise environmental factors and organizational process assets",
        "major challenges in communication",
        "purpose",
        "responsibility",
    }),
)


# Map filename fragments to configs
_FILENAME_CONFIG_MAP: Dict[str, DocumentConfig] = {
    "ML-HR-01": HR_DOC_CONFIG,
    "ML-ORG-01": EVENTS_DOC_CONFIG,
    "GL-ORG-01": COMMS_DOC_CONFIG,
}


def detect_doc_config(pdf_path: Path) -> Optional[DocumentConfig]:
    """Detect the right DocumentConfig from the PDF filename."""
    name = pdf_path.name
    for key, cfg in _FILENAME_CONFIG_MAP.items():
        if key in name:
            return cfg
    return None


def main():
    pdf_dir = Path("data/pdf")
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Legacy merged file kept for backward compatibility
    legacy_path = Path("data/chunks_inspection.json")
    all_chunks: List[Chunk] = []

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return

    for pdf_path in pdfs:
        cfg = detect_doc_config(pdf_path)
        if cfg is None:
            logger.warning(f"No DocumentConfig matched for {pdf_path.name} — skipping.")
            continue

        parser = PolicyManualParser(str(pdf_path), cfg)
        chunks = parser.parse()

        per_doc_path = chunks_dir / f"{cfg.doc_id}_chunks.json"
        save_chunks_to_file(chunks, str(per_doc_path))
        all_chunks.extend(chunks)

        logger.info(f"[{cfg.doc_id}] {len(chunks)} chunks → {per_doc_path}")

    # Write merged file
    save_chunks_to_file(all_chunks, str(legacy_path))

    logger.info(f"\n{'='*60}")
    logger.info(f"Ingestion complete! Total chunks across all docs: {len(all_chunks)}")
    logger.info(f"Per-doc chunks in: {chunks_dir}")
    logger.info(f"Merged file: {legacy_path}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()

