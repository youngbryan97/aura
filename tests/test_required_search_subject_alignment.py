"""Retrieved evidence must answer the request that authorized the search."""

from __future__ import annotations

from interface.routes.chat import _filter_required_search_result_by_subject


def _result(*entries: tuple[str, str, str]) -> dict:
    return {
        "ok": True,
        "results": [
            {"title": title, "url": url, "snippet": snippet}
            for title, url, snippet in entries
        ],
    }


def test_unrelated_shipping_result_cannot_become_architecture_evidence():
    result = _filter_required_search_result_by_subject(
        "What is one subtle engineering tradeoff in a hybrid recurrent AI architecture?",
        _result(
            (
                "ONE United States",
                "https://us.one-line.com/",
                "Ocean Network Express provides international container shipping and cargo tracking.",
            )
        ),
    )

    assert result["ok"] is False
    assert result["status"] == "required_search_subject_mismatch"
    assert result["irrelevant_results_removed"] == 1
    assert not result.get("results")


def test_mixed_results_keep_only_subject_aligned_evidence():
    result = _filter_required_search_result_by_subject(
        "Who founded Hugging Face?",
        _result(
            (
                "Hugging Face company history",
                "https://example.org/hugging-face-history",
                "Hugging Face was founded by Clement Delangue, Julien Chaumond, and Thomas Wolf.",
            ),
            (
                "ONE shipping",
                "https://us.one-line.com/",
                "Ocean Network Express provides container shipping and cargo tracking.",
            ),
        ),
    )

    assert result["ok"] is True
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Hugging Face company history"
    assert result["irrelevant_results_removed"] == 1


def test_no_search_result_stays_a_search_failure():
    original = {"ok": False, "error": "network unavailable"}
    assert _filter_required_search_result_by_subject("latest Mistral release", original) is original
