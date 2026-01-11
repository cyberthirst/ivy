from ivy.frontend.loader import loads


def test_event_with_indexed_arg_from_constructor():
    src = """
event Deployed:
    creator: indexed(address)
    value: uint256

@deploy
def __init__():
    log Deployed(creator=msg.sender, value=42)
    """

    c = loads(src)
    logs = c.get_logs()

    assert len(logs) == 1
    log = logs[0]
    assert log.event == "Deployed"
    assert log.args_obj.creator is not None
    assert log.args_obj.value == 42


def test_event_with_multiple_indexed_args():
    src = """
event Transfer:
    sender: indexed(address)
    receiver: indexed(address)
    amount: uint256

@external
def do_transfer():
    log Transfer(sender=msg.sender, receiver=msg.sender, amount=100)
    """

    c = loads(src)
    c.do_transfer()
    logs = c.get_logs()

    assert len(logs) == 1
    log = logs[0]
    assert log.event == "Transfer"
    assert log.args_obj.sender is not None
    assert log.args_obj.receiver is not None
    assert log.args_obj.amount == 100


def test_event_without_indexed_args():
    src = """
event Log:
    a: uint256
    b: uint256

@external
def emit_log():
    log Log(a=1, b=2)
    """

    c = loads(src)
    c.emit_log()
    logs = c.get_logs()

    assert len(logs) == 1
    log = logs[0]
    assert log.event == "Log"
    assert log.args_obj.a == 1
    assert log.args_obj.b == 2


def test_event_with_all_indexed_args():
    src = """
event AllIndexed:
    a: indexed(address)
    b: indexed(uint256)

@external
def emit():
    log AllIndexed(a=msg.sender, b=123)
    """

    c = loads(src)
    c.emit()
    logs = c.get_logs()

    assert len(logs) == 1
    log = logs[0]
    assert log.event == "AllIndexed"
    assert log.args_obj.a is not None
    assert log.args_obj.b == 123


def test_event_ordered_args():
    src = """
event Mixed:
    a: indexed(address)
    b: uint256
    c: indexed(uint256)
    d: address

@external
def emit():
    log Mixed(a=msg.sender, b=1, c=2, d=msg.sender)
    """

    c = loads(src)
    c.emit()
    logs = c.get_logs()

    assert len(logs) == 1
    log = logs[0]
    ordered = log.ordered_args()
    keys = [k for k, v in ordered]
    assert keys == ["a", "b", "c", "d"]
    assert ordered[1][1] == 1  # b
    assert ordered[2][1] == 2  # c


def test_multiple_events():
    src = """
event First:
    value: indexed(uint256)

event Second:
    data: uint256

@external
def emit_both():
    log First(value=1)
    log Second(data=2)
    """

    c = loads(src)
    c.emit_both()
    logs = c.get_logs()

    assert len(logs) == 2
    assert logs[0].event == "First"
    assert logs[0].args_obj.value == 1
    assert logs[1].event == "Second"
    assert logs[1].args_obj.data == 2
