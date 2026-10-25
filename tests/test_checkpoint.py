import json

import pytest


def test_resume_rejects_an_earlier_history_branch(tmp_path):
    from hvl_interact.engine.checkpoint import validate_resume_history

    path = tmp_path / "history.jsonl"
    path.write_text("\n".join(json.dumps({"epoch": n}) for n in (1, 2)) + "\n", encoding="utf-8")
    validate_resume_history(tmp_path, 2)
    with pytest.raises(ValueError, match="history|latest"):
        validate_resume_history(tmp_path, 1)
