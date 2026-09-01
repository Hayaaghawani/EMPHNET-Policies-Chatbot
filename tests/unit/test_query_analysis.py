from src.document_structure import DocumentStructureIndex, analyze_query
from src.ingestion import Chunk, ChunkMetadata


def _index():
    return DocumentStructureIndex([
        Chunk("Procedure 03: Hire Full Time Employees", ChunkMetadata(
            section_num=1, section_title="RECRUITMENT & SELECTION", subsection_num="1.5",
            subsection_title="Procedures", procedure_code="ML-HR-01.P03", chunk_type="procedure",
        )),
        Chunk("Sick Leave", ChunkMetadata(
            section_num=2, section_title="LEAVE", subsection_num="2.4", subsection_title="Policies",
            policy_name="Sick Leave", chunk_type="policy",
        )),
    ])


def test_focused_question_stays_specific():
    assert analyze_query("How many sick leave days are available?", _index()).intent == "specific"


def test_named_policy_is_structural():
    analysis = analyze_query("What are the sick leave policy rules?", _index())
    assert analysis.intent == "structural"
    assert analysis.policy_name == "Sick Leave"


def test_named_procedure_is_structural():
    analysis = analyze_query("What are the procedures to hire full time employees?", _index())
    assert analysis.intent == "structural"
    assert analysis.procedure_code == "ML-HR-01.P03"
