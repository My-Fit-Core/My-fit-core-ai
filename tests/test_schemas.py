"""API v1 예시 JSON(docs/api-v1-examples.json)이 스키마와 맞는지 확인한다."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    MAX_ITEMS,
    AnalyzeAccepted,
    AnalyzeRequest,
    AnalyzeResult,
    RecommendRequest,
    RecommendResult,
    StyleScore,
)

EXAMPLES = json.loads(
    (Path(__file__).resolve().parents[1] / "docs/api-v1-examples.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "key, model",
    [
        ("analyze_request", AnalyzeRequest),
        ("analyze_accepted", AnalyzeAccepted),
        ("analyze_result_partial", AnalyzeResult),
        ("analyze_result_failed", AnalyzeResult),
        ("recommend_request", RecommendRequest),
        ("recommend_result", RecommendResult),
    ],
)
def test_examples_match_schema(key, model):
    model.model_validate(EXAMPLES[key])


def test_upload_targets_limited_to_max_items():
    req = dict(EXAMPLES["analyze_request"])
    req["items"] = [{"client_item_key": f"k{i}", "upload_url": "u"} for i in range(MAX_ITEMS + 1)]
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(req)
    req["items"] = []
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(req)


def test_style_score_range():
    with pytest.raises(ValidationError):
        StyleScore(style_id=1, score=1.2)
