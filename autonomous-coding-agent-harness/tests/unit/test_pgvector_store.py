from agent.retrieval.store import _schema_dict


def test_schema_dict_accepts_dict() -> None:
    assert _schema_dict({"path": {"type": "string"}}) == {"path": {"type": "string"}}


def test_schema_dict_parses_json_string() -> None:
    assert _schema_dict('{"path": {"type": "string"}}') == {"path": {"type": "string"}}
