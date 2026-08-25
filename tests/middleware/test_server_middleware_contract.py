import inspect

from agent import server


def test_agent_factory_wires_terminal_gate_after_message_queue() -> None:
    source = inspect.getsource(server.get_agent)

    queue_position = source.index("check_message_queue_before_model")
    terminal_gate_position = source.index("ensure_no_empty_msg")
    timeout_position = source.index("TimeoutWrapupMiddleware()")

    assert queue_position < terminal_gate_position < timeout_position
