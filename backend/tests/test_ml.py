from app.ml import aspects, models


def test_baseline_detector_used_without_onnx():
    assert models.detector_name() == "baseline"
    p = models.fake_probability(["great product love it", "the strap snapped after two weeks"])
    assert p[0] > p[1]


def test_sentences_and_tags():
    s = aspects.sentences("Battery died fast. But the price is great!")
    assert len(s) == 2
    assert aspects.tag(s[0]) == ["Battery & power"]
    assert aspects.tag(s[1]) == ["Price & value"]


def test_aspect_terms_from_question():
    assert "Battery & power" in aspects.aspect_terms("How long does the battery last?")
    assert aspects.aspect_terms("hello there") == []


def test_analyze_sentence_level_and_excludes_fake():
    reviews = [
        {"id": 1, "text": "Battery died fast. The price is great value.", "rating": 3},
        {"id": 2, "text": "Battery died fast and it stopped working.", "rating": 1},
        {"id": 3, "text": "Great battery, great product love it.", "rating": 5},
    ]
    res = aspects.analyze(reviews, [0.1, 0.1, 0.9])  # review 3 flagged fake
    by = {a["aspect"]: a for a in res["aspects"]}
    assert by["Battery & power"]["mentions"] == 2  # fake review not counted
    assert by["Battery & power"]["negative"] == 2
    assert by["Price & value"]["positive"] == 1    # same review, opposite sentiment
    assert res["summary"]["suspected_fake"] == 1
    assert res["summary"]["avg_rating_genuine"] == 2.0
    assert res["summary"]["avg_rating_all"] == 3.0
    assert "Battery & power" in res["reviews"][2]["aspects"]  # still tagged for display
