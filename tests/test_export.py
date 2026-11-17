"""Unit tests for vials/modeling/export.py.
"""

import json
from types import SimpleNamespace

from mlflow.exceptions import MlflowException

from vials.modeling import export
from vials.modeling.export import (
    get_production_model,
    select_run_to_promote,
    write_version_info,
)


def _fake_get_run(f1):
    """Pretend mlflow.get_run: returns an object with .data.metrics."""
    return lambda run_id: SimpleNamespace(data=SimpleNamespace(metrics={"test_f1_score": f1}))


class EmptyClient:
    """Fake MLflow client with no Production model at all."""

    def get_model_version_by_alias(self, name, alias):
        raise MlflowException("no alias")

    def search_model_versions(self, filter_string):
        return []


# select_run_to_promote


def test_select_run_to_promote_promotes_when_nothing_in_production():
    best_run = {"run_id": "new-run", "metrics.test_f1_score": 0.7}

    result = select_run_to_promote(best_run, prod_run_id=None)

    assert result == "new-run"


def test_select_run_to_promote_promotes_when_new_model_is_better(monkeypatch):
    monkeypatch.setattr(export.mlflow, "get_run", _fake_get_run(0.8))
    best_run = {"run_id": "new-run", "metrics.test_f1_score": 0.9}

    result = select_run_to_promote(best_run, prod_run_id="prod-run")

    assert result == "new-run"


def test_select_run_to_promote_keeps_production_when_it_is_better(monkeypatch):
    monkeypatch.setattr(export.mlflow, "get_run", _fake_get_run(0.8))
    best_run = {"run_id": "new-run", "metrics.test_f1_score": 0.7}

    result = select_run_to_promote(best_run, prod_run_id="prod-run")

    assert result is None


# get_production_model


def test_get_production_model_returns_nones_when_nothing_in_production():
    result = get_production_model(EmptyClient(), "cnn_model")

    assert result == (None, None, None)


# write_version_info


def test_write_version_info_uses_fallback_when_no_production_alias(tmp_path):
    write_version_info(EmptyClient(), "cnn_model", export_dir=tmp_path, fallback_version=3)

    saved = json.loads((tmp_path / "version.json").read_text())

    assert saved == {"model_version": "3", "trained_at": None}