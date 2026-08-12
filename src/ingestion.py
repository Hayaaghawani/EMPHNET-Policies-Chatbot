"""
PDF Ingestion Script for EMPHNET HR Policy Manual

This module handles:
1. PDF text extraction with noise removal (headers/footers/watermarks)
2. Hierarchical parsing (Section → Subsection → Policy/Procedure)
3. Metadata extraction and attachment to chunks
4. Chunk inspection and validation before embedding

Document structure:
- 7 top-level Sections (Section 01-07)
- Each Section has subsections: X.1 Purpose, X.2 Scope, X.3 Responsibilities, 
  X.4 Policies, X.5 Procedures, X.6 Related Documents, X.7 References
- X.4 Policies: named policy topics (each becomes its own chunk)
- X.5 Procedures: named procedures with reference codes like ML-HR-01.P18
- After Section 7: Annexes (Annex 0N) and Attachments (Attachment 0N)
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pypdf import PdfReader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata common to all chunks"""
    document_title: str = "HR Policies & Procedures Manual"
    manual_code: str = "ML-HR-01"
    version: str = "V.01"
    effective_date: str = "August 7, 2022"
    language: str = "en"  # "en" or "ar" for bilingual sections


@dataclass
class ChunkMetadata(DocumentMetadata):
    """Metadata for individual chunks"""
    section_num: Optional[int] = None
    section_title: Optional[str] = None
    subsection_num: Optional[str] = None  # e.g., "1.4" for Section 1, Subsection 4
    subsection_title: Optional[str] = None
    policy_name: Optional[str] = None
    procedure_code: Optional[str] = None  # e.g., "ML-HR-01.P18"
    page_number: int = 0
    chunk_type: str = "text"  # "policy", "procedure", "annex", "attachment"


@dataclass
class Chunk:
    """A single chunk of text with metadata"""
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": asdict(self.metadata)
        }


class HeaderFooterRemover:
    """Removes repeating header/footer/watermark noise from PDF text"""
    
    # Common header/footer patterns from the manual
    HEADER_PATTERNS = [
        r"Global Health Development.*?EMPHNET",
        r"GHD.*?EMPHNET",
        r"ML-HR-01",
        r"V\.0?1",
        r"Page \d+ of 158",
        r"DocuSign Envelope ID:.*",
    ]
    
    @staticmethod
    def clean(text: str) -> str:
        """Remove header/footer noise from text"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that match header/footer patterns
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in HeaderFooterRemover.HEADER_PATTERNS):
                continue
            # Skip near-empty lines (just whitespace)
            if line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class HierarchicalPDFParser:
    """Parses PDF hierarchically: Section → Subsection → Policy/Procedure"""
    
    # Section patterns: "Section 01: Recruitment & Selection" or "SECTION 1: ..."
    SECTION_PATTERN = r"^(?:SECTION |Section )?(\d{1,2})[\s:]*(.+?)$"
    
    # Subsection patterns: "1.1 Purpose", "1.2 Scope", etc.
    # Pattern: number.number followed by subsection type
    SUBSECTION_PATTERN = r"^(\d{1,2})\.(\d)\s+(Purpose|Scope|Responsibilities|Policies|Procedures|Related Documents|References)\s*$"
    
    # Policy topic pattern: typically bold or indented, followed by text
    # We'll look for lines that look like headers within Policies (1.4)
    POLICY_PATTERN = r"^(?:\*\*)?([A-Z][a-zA-Z\s&-]+?)(?:\*\*)?:\s*$"
    
    # Procedure pattern with reference code: "ML-HR-01.P18 Leave Request"
    PROCEDURE_PATTERN = r"^(ML-HR-\d{2}\.P\d+)\s+(.+?)$"
    
    # Annex/Attachment patterns
    ANNEX_PATTERN = r"^(?:ANNEX|Annex)\s+(\d{2})(?:\s|:|\(|$)"
    ATTACHMENT_PATTERN = r"^(?:ATTACHMENT|Attachment)\s+(\d{2})(?:\s|:|\(|$)"
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.reader = PdfReader(self.pdf_path)
        self.num_pages = len(self.reader.pages)
        logger.info(f"Loaded PDF: {self.pdf_path.name} ({self.num_pages} pages)")
    
    def extract_text_by_page(self) -> List[Tuple[int, str]]:
        """Extract text from each page, returning (page_num, text) tuples"""
        pages_text = []
        for page_num, page in enumerate(self.reader.pages, 1):
            text = page.extract_text()
            text = HeaderFooterRemover.clean(text)
            pages_text.append((page_num, text))
        
        logger.info(f"Extracted text from {len(pages_text)} pages")
        return pages_text
    
    def parse_sections(self, pages_text: List[Tuple[int, str]]) -> List[Chunk]:
        """
        Parse text hierarchically and return chunks with metadata.
        
        Strategy:
        1. Combine all text with page markers
        2. Find Section boundaries
        3. For each section, find subsections
        4. For Policies (X.4), split by policy name
        5. For Procedures (X.5), identify procedure blocks with their codes
        6. Track current section/subsection context for metadata
        """
        chunks = []
        full_text = '\n'.join([text for _, text in pages_text])
        lines = full_text.split('\n')
        
        current_section = None
        current_subsection = None
        current_context = {"section_num": None, "section_title": None, "page": 1}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check for Section header
            section_match = re.match(self.SECTION_PATTERN, line, re.IGNORECASE)
            if section_match:
                section_num = int(section_match.group(1))
                section_title = section_match.group(2).strip()
                current_section = {"num": section_num, "title": section_title}
                current_context = {
                    "section_num": section_num,
                    "section_title": section_title,
                    "page": 1
                }
                logger.info(f"Found Section {section_num}: {section_title}")
                i += 1
                continue
            
            # Check for Subsection header
            subsection_match = re.match(self.SUBSECTION_PATTERN, line, re.IGNORECASE)
            if subsection_match:
                sec_num = int(subsection_match.group(1))
                sub_num = int(subsection_match.group(2))
                sub_type = subsection_match.group(3)
                current_subsection = {"num": f"{sec_num}.{sub_num}", "type": sub_type}
                current_context["subsection_num"] = f"{sec_num}.{sub_num}"
                current_context["subsection_type"] = sub_type
                logger.info(f"Found Subsection {sec_num}.{sub_num}: {sub_type}")
                i += 1
                continue
            
            # Check for Annex/Attachment
            annex_match = re.match(self.ANNEX_PATTERN, line, re.IGNORECASE)
            attachment_match = re.match(self.ATTACHMENT_PATTERN, line, re.IGNORECASE)
            
            if annex_match or attachment_match:
                # Handle Annex/Attachment chunks
                chunk_type = "annex" if annex_match else "attachment"
                chunk_id = annex_match.group(1) if annex_match else attachment_match.group(1)
                
                # Collect lines until next annex/attachment or end
                chunk_lines = [line]
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if (re.match(self.ANNEX_PATTERN, next_line, re.IGNORECASE) or 
                        re.match(self.ATTACHMENT_PATTERN, next_line, re.IGNORECASE)):
                        break
                    if next_line:
                        chunk_lines.append(next_line)
                    i += 1
                
                chunk_text = '\n'.join(chunk_lines)
                metadata = ChunkMetadata(
                    section_num=current_context.get("section_num"),
                    section_title=current_context.get("section_title"),
                    chunk_type=chunk_type,
                    page_number=current_context.get("page", 1)
                )
                chunks.append(Chunk(text=chunk_text, metadata=metadata))
                continue
            
            # Generic chunk collection for now
            # This will be expanded with policy/procedure-specific logic
            i += 1
        
        logger.info(f"Parsed {len(chunks)} chunks from PDF")
        return chunks
    
    def parse(self) -> List[Chunk]:
        """Main parsing entry point"""
        pages_text = self.extract_text_by_page()
        chunks = self.parse_sections(pages_text)
        return chunks


def save_chunks_to_file(chunks: List[Chunk], output_path: str) -> None:
    """Save chunks to JSON file for inspection"""
    output = []
    for chunk in chunks:
        output.append(chunk.to_dict())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    """Main ingestion workflow"""
    # Configuration
    pdf_path = "data/pdf/Human_Resources_Policies___Procedures_Manual__ML-HR-01__V_01__docx.pdf"
    chunks_output_path = "data/chunks_inspection.json"
    
    # Check if PDF exists
    if not Path(pdf_path).exists():
        logger.warning(f"PDF not found at {pdf_path}")
        logger.info("Please place the PDF file in the data/pdf/ directory")
        return
    
    # Parse PDF
    parser = HierarchicalPDFParser(pdf_path)
    chunks = parser.parse()
    
    # Save chunks for inspection
    Path(chunks_output_path).parent.mkdir(parents=True, exist_ok=True)
    save_chunks_to_file(chunks, chunks_output_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Ingestion complete!")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Chunks saved to: {chunks_output_path}")
    logger.info(f"Review the output JSON to verify chunk structure before proceeding to embeddings")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
