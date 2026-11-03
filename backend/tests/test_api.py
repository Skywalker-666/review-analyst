from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_and_model():
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["detector"] == "baseline"
    assert "classic" in client.get("/api/model").json()


def test_sample_products_and_analysis():
    products = client.get("/api/sample").json()["products"]
    assert len(products) == 4
    r = client.get(f"/api/sample/{products[0]}").json()
    assert len(r["reviews"]) == 250
    assert r["analysis"]["summary"]["total"] == 250
    assert 0 <= r["analysis"]["summary"]["detector_accuracy_on_sample"] <= 1
    assert client.get("/api/sample/Nope").status_code == 404


def test_upload_parses_and_validates():
    csv = "review,stars\nThe battery died after a week.,1\nshort,5\nLove the color and the fit.,5\n"
    r = client.post("/api/upload", files={"file": ("r.csv", csv, "text/csv")})
    assert r.status_code == 200
    reviews = r.json()["reviews"]
    assert [x["rating"] for x in reviews] == [1, 5]  # "short" skipped
    bad = client.post("/api/upload", files={"file": ("r.csv", "a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 400


def test_analyze_and_ask_round_trip():
    reviews = [{"id": 1, "text": "Battery died fast. Price is great value.", "rating": 3},
               {"id": 2, "text": "The strap snapped after two weeks.", "rating": 1}]
    a = client.post("/api/analyze", json={"reviews": reviews}).json()
    assert a["summary"]["total"] == 2
    scored = [{**r, **{k: s[k] for k in ("fake_prob", "positive_prob")}} for r, s in zip(reviews, a["reviews"])]
    q = client.post("/api/ask", json={"question": "how is the battery?", "reviews": scored,
                                      "exclude_fake": False}).json()
    assert q["citations"][0]["id"] == 1


def test_input_limits():
    assert client.post("/api/analyze", json={"reviews": []}).status_code == 422
    dup = [{"id": 1, "text": "a fine review"}, {"id": 1, "text": "another one"}]
    assert client.post("/api/analyze", json={"reviews": dup}).status_code == 400
    assert client.post("/api/ask", json={"question": "hi", "reviews": dup[:1]}).status_code == 422
