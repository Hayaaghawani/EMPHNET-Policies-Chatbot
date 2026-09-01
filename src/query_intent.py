"""Re-export query analysis — logic lives in document_structure.py."""

from .document_structure import DocumentStructureIndex, QueryAnalysis, analyze_query

__all__ = ["DocumentStructureIndex", "QueryAnalysis", "analyze_query"]
