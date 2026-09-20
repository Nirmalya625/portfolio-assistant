from rag import DocumentIndex


def test_add_document_creates_chunks():
    idx = DocumentIndex()
    added = idx.add_document("doc.pdf", "word " * 500)
    assert added > 0
    assert idx.has_documents()
    assert idx.documents == ["doc.pdf"]


def test_search_finds_relevant_chunk():
    idx = DocumentIndex()
    idx.add_document(
        "policy.pdf",
        "Employees get twenty five days of paid annual leave every year. " * 5
        + "Unrelated filler about office plants and coffee machines. " * 5,
    )
    results = idx.search("how many vacation days do employees get")
    assert len(results) >= 1
    score, chunk = results[0]
    assert "leave" in chunk["text"].lower()
    assert score > 0


def test_search_returns_nothing_for_unrelated_query():
    idx = DocumentIndex()
    idx.add_document("policy.pdf", "Employees get paid annual leave every year. " * 10)
    results = idx.search("what is the airspeed velocity of a swallow")
    assert results == []


def test_search_on_empty_index_returns_nothing():
    idx = DocumentIndex()
    assert idx.search("anything") == []
    assert not idx.has_documents()


def test_clear_removes_all_documents():
    idx = DocumentIndex()
    idx.add_document("doc.pdf", "some content here " * 50)
    assert idx.has_documents()
    idx.clear()
    assert not idx.has_documents()
    assert idx.documents == []


def test_multiple_documents_are_both_searchable():
    idx = DocumentIndex()
    idx.add_document("hr.pdf", "Annual leave is twenty five days per year. " * 10)
    idx.add_document("it.pdf", "Password rotation is required every ninety days. " * 10)
    assert idx.documents == ["hr.pdf", "it.pdf"]

    leave_results = idx.search("how much annual leave")
    assert any(c["doc"] == "hr.pdf" for _s, c in leave_results)

    password_results = idx.search("password rotation policy")
    assert any(c["doc"] == "it.pdf" for _s, c in password_results)


def test_add_sections_makes_one_titled_chunk_per_heading():
    idx = DocumentIndex()
    added = idx.add_sections("kb.md", "# Title\nintro text\n\n## Alpha\napples oranges\n\n## Beta\nbananas grapes\n")
    assert added == 3
    assert [c["title"] for c in idx.chunks] == ["Title", "Alpha", "Beta"]
    score, chunk = idx.search("bananas")[0]
    assert chunk["title"] == "Beta"


def test_question_words_do_not_match_by_themselves():
    idx = DocumentIndex()
    idx.add_document("doc.pdf", "what how which when about me tell " * 20 + "kubernetes cluster " * 20)
    assert idx.search("what about me") == []
