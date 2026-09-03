from src.document_structure import DocumentStructureIndex, QueryAnalysis, analyze_query
from src.ingestion import Chunk, ChunkMetadata


def _multi_doc_index():
    return DocumentStructureIndex([
        # Document 1: HR Policy (ML-HR-01)
        Chunk("Procedure 03: Hire Full Time Employees", ChunkMetadata(
            doc_id="ML-HR-01", doc_type="policy_manual",
            section_num=1, section_title="RECRUITMENT AND SELECTION", subsection_num="1.5",
            subsection_title="Procedures", procedure_code="ML-HR-01.P03", chunk_type="procedure",
        )),
        Chunk("Sick Leave Policy", ChunkMetadata(
            doc_id="ML-HR-01", doc_type="policy_manual",
            section_num=5, section_title="ATTENDANCE LEAVES VACATIONS", subsection_num="5.4",
            subsection_title="Policies", policy_name="Sick Leave", chunk_type="policy",
        )),
        # Document 2: Events Management (ML-ORG-01)
        Chunk("Procedure 01: Event Planning and Preparation Phase", ChunkMetadata(
            doc_id="ML-ORG-01", doc_type="procedure_manual",
            section_num=1, section_title="PLANNING AND PREPARATION FOR THE EVENTS", subsection_num="1.5",
            subsection_title="Procedures", procedure_code="ML-ORG-01.P01", chunk_type="procedure",
        )),
        Chunk("Venue Selection Policy", ChunkMetadata(
            doc_id="ML-ORG-01", doc_type="procedure_manual",
            section_num=1, section_title="PLANNING AND PREPARATION FOR THE EVENTS", subsection_num="1.4",
            subsection_title="Policies", policy_name="Venue Selection", chunk_type="policy",
        )),
        # Document 3: Internal Communication Guideline (GL-ORG-01)
        Chunk("Virtual Engagement Guidelines", ChunkMetadata(
            doc_id="GL-ORG-01", doc_type="guideline",
            section_num=2, section_title="GHD EMPHNET Internal Communication Channels", subsection_num="2.1",
            subsection_title="Virtual Engagement", chunk_type="subsection",
        )),
    ])


def test_doc_id_extraction_from_query():
    idx = _multi_doc_index()
    assert idx.match_doc_id("What is the HR policy on sick leave?") == "ML-HR-01"
    assert idx.match_doc_id("How to handle events management travel booking?") == "ML-ORG-01"
    assert idx.match_doc_id("What are the internal communication guidelines for WhatsApp?") == "GL-ORG-01"
    assert idx.match_doc_id("How many leave days do I get?") is None


def test_focused_question_stays_specific_across_docs():
    idx = _multi_doc_index()
    assert analyze_query("How many days of sick leave am I allowed?", idx).intent == "specific"
    assert analyze_query("Who is responsible for venue selection?", idx).intent == "specific"
    assert analyze_query("When should event planning start?", idx).intent == "specific"


def test_structural_intent_requires_listing_or_path_match():
    idx = _multi_doc_index()
    
    q1 = analyze_query("List all procedures for hiring full time employees", idx)
    assert q1.intent == "structural"
    assert q1.doc_id == "ML-HR-01"
    assert q1.procedure_code == "ML-HR-01.P03"

    q2 = analyze_query("What is the venue selection policy?", idx)
    assert q2.intent == "structural"
    assert q2.doc_id == "ML-ORG-01"
    assert q2.policy_name == "Venue Selection"

    q3 = analyze_query("Show virtual engagement communication guidelines", idx)
    assert q3.intent == "structural"
    assert q3.doc_id == "GL-ORG-01"


def test_cross_doc_section_collision_avoidance():
    idx = _multi_doc_index()
    q_hr = analyze_query("List all recruitment and selection steps in the HR manual", idx)
    assert q_hr.intent == "structural"
    assert q_hr.doc_id == "ML-HR-01"
    assert q_hr.section_num == 1

    q_events = analyze_query("List all planning and preparation steps for events", idx)
    assert q_events.intent == "structural"
    assert q_events.doc_id == "ML-ORG-01"
    assert q_events.section_num == 1
