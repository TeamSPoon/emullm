from pathlib import Path

from emullm.chat import Conversation


class RecordingClient:
    def __init__(self) -> None:
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append([message.copy() for message in messages])
        return f"reply {len(self.requests)}"


def test_conversation_sends_context_from_prior_turns() -> None:
    client = RecordingClient()
    conversation = Conversation("Be concise.")

    assert conversation.ask("My name is Ada.", client) == "reply 1"
    assert conversation.ask("What is my name?", client) == "reply 2"

    assert client.requests[1] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "What is my name?"},
    ]


def test_conversation_history_round_trip(tmp_path: Path) -> None:
    history_file = tmp_path / "chat.json"
    original = Conversation()
    original.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    original.save(history_file)

    restored = Conversation()
    restored.load(history_file)

    assert restored.messages == original.messages


def test_clear_keeps_system_prompt() -> None:
    conversation = Conversation("Be concise.")
    conversation.messages.extend(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )

    conversation.clear()

    assert conversation.messages == [{"role": "system", "content": "Be concise."}]


def test_one_shot_requests_do_not_include_prior_turns() -> None:
    client = RecordingClient()
    conversation = Conversation("Be concise.", contextual=False)

    conversation.ask("My name is Ada.", client)
    conversation.ask("What is my name?", client)

    assert client.requests == [
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "My name is Ada."},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is my name?"},
        ],
    ]
    assert conversation.messages == [{"role": "system", "content": "Be concise."}]
