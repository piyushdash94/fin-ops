import pytest

from finops.ledger import Ledger
from finops.track import TrackedClient


class FakeUsage(dict):
    pass


def _message(model="claude-opus-5", **usage):
    payload = {"input_tokens": 100, "output_tokens": 50, **usage}
    return type("Msg", (), {"id": "msg_1", "model": model, "usage": payload})()


class FakeStream:
    def __init__(self, message):
        self._message = message
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def get_final_message(self):
        return self._message


class FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _message()

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(_message())

    def count_tokens(self, **kwargs):
        return type("R", (), {"input_tokens": 7})()


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()
        self.api_key = "sk-fake"


@pytest.fixture
def tracked(tmp_path):
    return TrackedClient(FakeClient(), ledger=Ledger(tmp_path / "u.jsonl"), tags={"env": "prod"})


def test_create_records_usage_and_returns_the_sdk_object(tracked):
    message = tracked.messages.create(model="claude-opus-5", max_tokens=16)
    assert message.id == "msg_1"
    (record,) = list(tracked.ledger.read())
    assert record.usage.input_tokens == 100
    assert record.tags == {"env": "prod"}
    assert record.duration_ms is not None


def test_finops_tags_are_not_forwarded_to_the_sdk(tracked):
    tracked.messages.create(model="claude-opus-5", max_tokens=16, finops_tags={"feature": "x"})
    assert "finops_tags" not in tracked.messages._inner.kwargs
    (record,) = list(tracked.ledger.read())
    assert record.tags == {"env": "prod", "feature": "x"}


def test_tagged_derives_a_scoped_client_sharing_one_ledger(tracked):
    child = tracked.tagged(feature="search")
    child.messages.create(model="claude-opus-5", max_tokens=16)
    tracked.messages.create(model="claude-opus-5", max_tokens=16)

    assert child.ledger is tracked.ledger
    by_tag = tracked.report(group_by="tag:feature")
    assert {g.key for g in by_tag.groups} == {"search", "(untagged)"}


def test_streaming_records_on_context_exit(tracked):
    with tracked.messages.stream(model="claude-opus-5", max_tokens=16) as stream:
        assert stream.get_final_message().id == "msg_1"
    assert len(list(tracked.ledger.read())) == 1


def test_stream_failure_records_nothing_and_propagates(tracked):
    with pytest.raises(RuntimeError):
        with tracked.messages.stream(model="claude-opus-5", max_tokens=16):
            raise RuntimeError("boom")
    assert list(tracked.ledger.read()) == []


def test_unknown_attributes_pass_through(tracked):
    assert tracked.api_key == "sk-fake"
    assert tracked.messages.count_tokens(model="claude-opus-5").input_tokens == 7


def test_a_broken_ledger_never_breaks_the_request(tmp_path):
    class ExplodingLedger(Ledger):
        def record(self, *a, **kw):
            raise OSError("disk full")

    client = TrackedClient(FakeClient(), ledger=ExplodingLedger(tmp_path / "u.jsonl"))
    with pytest.warns(UserWarning, match="could not record usage"):
        message = client.messages.create(model="claude-opus-5", max_tokens=16)
    assert message.id == "msg_1"


def test_unknown_model_in_a_response_is_warned_not_raised(tmp_path):
    class OddClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.messages.create = lambda **kw: _message(model="mystery-model-9")

    client = TrackedClient(OddClient(), ledger=Ledger(tmp_path / "u.jsonl"))
    with pytest.warns(UserWarning):
        assert client.messages.create(model="mystery-model-9").id == "msg_1"
