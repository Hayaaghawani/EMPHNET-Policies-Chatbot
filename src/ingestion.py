"""
PDF Ingestion Script for EMPHNET HR Policy Manual

Extracts text page-by-page, parses the real document hierarchy, and produces
self-contained chunks suitable for RAG retrieval.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUBSECTION_TYPES = (
    "Purpose", "Scope", "Responsibilities", "Policies",
    "Procedures", "Related Documents", "References"
)

KNOWN_POLICY_TOPICS = {
    "employee attendance", "personal leave and annual vacations", "business leave",
    "working from anywhere (wfa)", "unpaid vacation", "sick leave", "official holidays",
    "hajj vacation", "emergency vacation", "paternity leave", "maternity leave",
    "general", "disciplinary actions", "grievance", "performance management",
    "recruitment & selection", "training & development",
}


@dataclass
class DocumentMetadata:
    document_title: str = "HR Policies & Procedures Manual"
    manual_code: str = "ML-HR-01"
    version: str = "V.01"
    effective_date: str = "August 7, 2022"
    language: str = "en"


@dataclass
class ChunkMetadata(DocumentMetadata):
    section_num: Optional[int] = None
    section_title: Optional[str] = None
    subsection_num: Optional[str] = None
    subsection_title: Optional[str] = None
    policy_name: Optional[str] = None
    procedure_code: Optional[str] = None
    page_number: int = 0
    chunk_type: str = "text"


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata

    def to_dict(self):
        return {"text": self.text, "metadata": asdict(self.metadata)}


class HeaderFooterRemover:
    HEADER_PATTERNS = [
        r"^Global Health Development[\s|].*EMPHNET\s*$",
        r"^GHD[\s|]+EMPHNET\s*$",
        r"^ML-HR-01\s*,?\s*V\.0?1\s*$",
        r"^V\.0?1\s*$",
        r"^Page \d+ of \d+\s*$",
        r"^DocuSign Envelope ID:\s*[A-Z0-9-]+\s*$",
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        lines = []
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) <= 120 and any(
                re.search(p, stripped, re.IGNORECASE) for p in cls.HEADER_PATTERNS
            ):
                continue
            lines.append(stripped)
        return '\n'.join(lines)


class HierarchicalPDFParser:
    """Parse EMPHNET HR manual with correct section / policy / procedure boundaries."""

    SECTION_RE = re.compile(
        r'^SECTION\s*0?(\d{1,2})\s*:\s*(.+)$', re.IGNORECASE
    )
    SUBSECTION_RE = re.compile(
        r'^(\d{1,2})\.(\d)\s+(' + '|'.join(SUBSECTION_TYPES) + r')\s*$',
        re.IGNORECASE
    )
    PROCEDURE_START_RE = re.compile(
        r'^Procedure\s+(\d+)\s*:\s*(.+)$', re.IGNORECASE
    )
    PROCEDURE_CODE_RE = re.compile(
        r'^(ML-HR-\d{2}\.P\d+)\s+(.+)$', re.IGNORECASE
    )
    ANNEX_RE = re.compile(r'^(?:ANNEX|Annex)\s+(\d{2})\b', re.IGNORECASE)
    ATTACHMENT_RE = re.compile(r'^(?:ATTACHMENT|Attachment)\s+(\d{2})\b', re.IGNORECASE)
    NUMBERED_ITEM_RE = re.compile(r'^(\d+)\.\s+')
    TOC_LINE_RE = re.compile(r'\.{4,}')
    PROCESS_MAP_RE = re.compile(r'^Process Map', re.IGNORECASE)
    STEP_HEADER_RE = re.compile(r'^Step\s+Action\s+Responsibility\s*$', re.IGNORECASE)

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.reader = PdfReader(str(self.pdf_path))
        logger.info(f"Loaded PDF: {self.pdf_path.name} ({len(self.reader.pages)} pages)")

    def extract_text_by_page(self) -> List[Tuple[int, str]]:
        pages = []
        for page_num, page in enumerate(self.reader.pages, 1):
            text = HeaderFooterRemover.clean(page.extract_text() or "")
            pages.append((page_num, text))
        logger.info(f"Extracted text from {len(pages)} pages")
        return pages

    @staticmethod
    def _is_toc_page(text: str) -> bool:
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        if not lines:
            return True
        toc_lines = sum(1 for ln in lines if HierarchicalPDFParser.TOC_LINE_RE.search(ln))
        # Contents pages are mostly dot-leader lines
        if toc_lines >= 3 or (len(lines) <= 6 and toc_lines >= 1):
            return True
        # Early manual pages that are only section index lists
        if any(re.match(r'^\d-\s+\w', ln) for ln in lines) and len(lines) < 15:
            return True
        return False

    @staticmethod
    def _is_policy_header(line: str, in_policies: bool) -> bool:
        if not in_policies:
            return False
        if len(line) < 3 or len(line) > 70:
            return False
        if HierarchicalPDFParser.NUMBERED_ITEM_RE.match(line):
            return False
        if HierarchicalPDFParser.SUBSECTION_RE.match(line):
            return False
        if HierarchicalPDFParser.SECTION_RE.match(line):
            return False
        if HierarchicalPDFParser.PROCEDURE_START_RE.match(line):
            return False
        if line.lower() in KNOWN_POLICY_TOPICS:
            return True
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

    def _make_chunk(
        self,
        body: str,
        ctx: dict,
        chunk_type: str,
        page_number: int,
        policy_name: Optional[str] = None,
        procedure_code: Optional[str] = None,
    ) -> Optional[Chunk]:
        body = body.strip()
        if not body or len(body) < 20:
            return None
        if self.PROCESS_MAP_RE.search(body[:80]):
            return None
        if self.TOC_LINE_RE.search(body) and body.count('....') >= 2:
            return None

        prefix = self._build_prefix({**ctx, "policy_name": policy_name or ctx.get("policy_name")})
        text = f"[{prefix}]\n{body}" if prefix else body

        metadata = ChunkMetadata(
            section_num=ctx.get("section_num"),
            section_title=ctx.get("section_title"),
            subsection_num=ctx.get("subsection_num"),
            subsection_title=ctx.get("subsection_title"),
            policy_name=policy_name or ctx.get("policy_name"),
            procedure_code=procedure_code or ctx.get("procedure_code"),
            page_number=page_number,
            chunk_type=chunk_type,
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
        result = []
        for part in parts:
            result.append(Chunk(text=part, metadata=chunk.metadata))
        return result

    def parse_sections(self, pages_text: List[Tuple[int, str]]) -> List[Chunk]:
        chunks: List[Chunk] = []
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
        in_procedure = False
        skip_until_section = False

        def flush():
            nonlocal block_lines, block_type, block_policy, block_procedure_code, in_procedure
            if not block_lines:
                return
            body = '\n'.join(block_lines)
            chunk = self._make_chunk(
                body, ctx, block_type, block_page,
                policy_name=block_policy,
                procedure_code=block_procedure_code,
            )
            if chunk:
                chunks.extend(self._split_large_chunk(chunk))
            block_lines = []
            block_policy = None
            block_procedure_code = None
            in_procedure = False

        for page_num, page_text in pages_text:
            if self._is_toc_page(page_text):
                continue

            in_policies = (
                (ctx.get("subsection_title") or "").lower() == "policies"
            )
            in_procedures = (
                (ctx.get("subsection_title") or "").lower() == "procedures"
            )

            for line in page_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Skip process-map noise inside procedures
                if self.PROCESS_MAP_RE.match(line):
                    flush()
                    skip_until_section = True
                    continue

                if skip_until_section:
                    if self.SECTION_RE.match(line) or self.PROCEDURE_START_RE.match(line):
                        skip_until_section = False
                    else:
                        continue

                section_m = self.SECTION_RE.match(line)
                if section_m:
                    flush()
                    ctx["section_num"] = int(section_m.group(1))
                    ctx["section_title"] = section_m.group(2).strip()
                    ctx["subsection_num"] = None
                    ctx["subsection_title"] = None
                    ctx["policy_name"] = None
                    ctx["procedure_name"] = None
                    ctx["procedure_code"] = None
                    block_type = "section_intro"
                    block_page = page_num
                    logger.info(f"Section {ctx['section_num']:02d}: {ctx['section_title']}")
                    continue

                sub_m = self.SUBSECTION_RE.match(line)
                if sub_m:
                    flush()
                    sec = int(sub_m.group(1))
                    sub = int(sub_m.group(2))
                    sub_raw = sub_m.group(3)
                    sub_type = sub_raw.title() if sub_raw.lower() == "purpose" else sub_raw
                    ctx["subsection_num"] = f"{sec}.{sub}"
                    ctx["subsection_title"] = sub_type
                    ctx["policy_name"] = None
                    ctx["procedure_name"] = None
                    ctx["procedure_code"] = None
                    block_type = "subsection"
                    block_page = page_num
                    logger.info(f"  {ctx['subsection_num']} {sub_type}")
                    continue

                proc_m = self.PROCEDURE_START_RE.match(line)
                if proc_m:
                    flush()
                    proc_num = proc_m.group(1)
                    proc_name = proc_m.group(2).strip()
                    ctx["procedure_name"] = f"Procedure {proc_num}: {proc_name}"
                    ctx["procedure_code"] = f"ML-HR-01.P{int(proc_num):02d}"
                    ctx["policy_name"] = None
                    block_type = "procedure"
                    block_page = page_num
                    in_procedure = True
                    block_procedure_code = ctx["procedure_code"]
                    block_lines = []
                    continue

                if in_procedure and self.PROCEDURE_CODE_RE.match(line):
                    # Metadata line inside procedure header — skip, don't start new chunk
                    continue

                if self.STEP_HEADER_RE.match(line):
                    continue

                annex_m = self.ANNEX_RE.match(line)
                attach_m = self.ATTACHMENT_RE.match(line)
                if annex_m or attach_m:
                    flush()
                    block_type = "annex" if annex_m else "attachment"
                    block_page = page_num
                    block_lines = [line]
                    continue

                if self._is_policy_header(line, in_policies):
                    flush()
                    block_policy = line
                    ctx["policy_name"] = line
                    block_type = "policy"
                    block_page = page_num
                    continue

                # Responsibility codes on their own line (EMP, DM, HR, SYS)
                if in_procedure and re.match(r'^[A-Z]{2,5}$', line) and block_lines:
                    block_lines[-1] = f"{block_lines[-1]} (Responsibility: {line})"
                    continue

                if not block_lines:
                    block_page = page_num

                block_lines.append(line)

        flush()
        logger.info(f"Parsed {len(chunks)} chunks from PDF")
        return chunks

    def parse(self) -> List[Chunk]:
        return self.parse_sections(self.extract_text_by_page())


def save_chunks_to_file(chunks: List[Chunk], output_path: str) -> None:
    output = [c.to_dict() for c in chunks]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    pdf_path = "data/pdf/Human Resources Policies & Procedures Manual (ML-HR-01, V.01).docx.pdf"
    chunks_output_path = "data/chunks_inspection.json"

    if not Path(pdf_path).exists():
        logger.warning(f"PDF not found at {pdf_path}")
        return

    parser = HierarchicalPDFParser(pdf_path)
    chunks = parser.parse()
    save_chunks_to_file(chunks, chunks_output_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"Ingestion complete! Total chunks: {len(chunks)}")
    logger.info(f"Saved to: {chunks_output_path}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
