import json
from datetime import UTC, datetime, timedelta
import numpy as np
from vials.monitoring.drift import load_inference_logs, detect_drift, filter_logs_by_time

#----------------------------------------------------------------
# A baseline like the one training writes: confident model, 50/50 classes
BASELINE = {
    "confidence_statistics": {"mean": 0.9, "std": 0.05, "min": 0.7, "max": 1.0},
    "class_distribution": {"class_0_count": 50, "class_1_count": 50, "total": 100},
}

def _log(prediction, confidence, hours_ago=1):
    """One inference log entry, like the wrapper writes for every request."""
    timestamp = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    return {"timestamp": timestamp, "prediction": prediction, "confidence": confidence}

#----------------------------------------------------------------

def test_load_inference_logs_returns_empty_list_when_file_is_missing(tmp_path):
    log_path = tmp_path / "inference_log.json"

    result = load_inference_logs(log_path)

    assert result == []


def test_load_inference_logs_returns_empty_list_when_json_is_broken(tmp_path):
    log_path = tmp_path / "inference_log.json"
    log_path.write_text("{not valid json")

    result = load_inference_logs(log_path)

    assert result == []


def test_load_inference_logs_returns_empty_list_when_json_is_not_a_list(tmp_path):
    log_path = tmp_path / "inference_log.json"
    log_path.write_text(json.dumps({"prediction": 1}))

    result = load_inference_logs(log_path)

    assert result == []


def test_load_inference_logs_returns_logs_from_valid_file(tmp_path):
    log_path = tmp_path / "inference_log.json"
    logs = [{"prediction": 1, "confidence": 0.9}]
    log_path.write_text(json.dumps(logs))

    result = load_inference_logs(log_path)

    assert result == logs


# filter_logs_by_time 
def test_filter_logs_by_time_keeps_recent_and_drops_old_logs():
    recent = _log(prediction=1, confidence=0.9, hours_ago=1)
    old = _log(prediction=1, confidence=0.9, hours_ago=24 * 10)
 
    result = filter_logs_by_time([recent, old], days=7)
 
    assert result == [recent]
 
 
# detect_drift
def test_detect_drift_with_no_logs_reports_no_drift():
    result = detect_drift([], baseline_stats=BASELINE)
 
    assert result["overall_drift_detected"] is False
    assert result["num_samples"] == 0
    assert result["error"] == "No logs found"
 
 
def test_detect_drift_flags_confidence_drift():
    np.random.seed(0)
    # Baseline confidence is ~0.9, now the model is only 0.5 sure; classes stay 50/50
    logs = [_log(prediction=i % 2, confidence=0.5) for i in range(50)]
 
    result = detect_drift(logs, baseline_stats=BASELINE)
 
    assert result["confidence_drift"]["drift_detected"] is True
    assert result["overall_drift_detected"] is True
 
 
def test_detect_drift_flags_class_distribution_drift():
    np.random.seed(0)
    # Baseline is 50/50 good/defective, now every prediction is defective
    logs = [_log(prediction=1, confidence=0.9) for _ in range(100)]
 
    result = detect_drift(logs, baseline_stats=BASELINE)
 
    assert result["class_distribution_drift"]["drift_detected"] is True
    assert result["overall_drift_detected"] is True
 
 
def test_detect_drift_without_baseline_reports_no_drift():
    logs = [_log(prediction=1, confidence=0.5) for _ in range(10)]
 
    result = detect_drift(logs, baseline_stats=None)
 
    assert result["num_samples"] == 10
    assert result["confidence_drift"]["drift_detected"] is False
    assert result["class_distribution_drift"]["drift_detected"] is False
    assert result["overall_drift_detected"] is False