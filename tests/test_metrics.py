import json

from src.character.metrics import MetricsLog


def test_noop_without_path_never_raises():
    metrics = MetricsLog(path=None)
    metrics.event("engaged", face_x_frac=0.1)
    metrics.sample_resources()
    metrics.close()  # must not raise


def test_event_appends_jsonl_with_kind_and_fields(tmp_path):
    path = tmp_path / "metrics.jsonl"
    metrics = MetricsLog(path=path)
    metrics.event("first_audio", latency_s=1.23)
    metrics.close()

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "first_audio"
    assert record["latency_s"] == 1.23
    assert "t_wall" in record and "t_monotonic" in record


def test_sample_resources_writes_cpu_and_rss(tmp_path):
    path = tmp_path / "metrics.jsonl"
    metrics = MetricsLog(path=path)
    metrics.sample_resources()
    metrics.close()

    record = json.loads(path.read_text().strip().splitlines()[0])
    assert record["kind"] == "resource_sample"
    assert "cpu_percent" in record
    assert record["rss_bytes"] > 0


def test_multiple_events_append_across_calls(tmp_path):
    path = tmp_path / "metrics.jsonl"
    metrics = MetricsLog(path=path)
    metrics.event("engaged", face_x_frac=0.0)
    metrics.event("disengaged")
    metrics.close()

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["kind"] == "engaged"
    assert json.loads(lines[1])["kind"] == "disengaged"
