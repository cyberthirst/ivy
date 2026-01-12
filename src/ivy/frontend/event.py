from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    address: str  # checksum address
    event_type: Any  # vyper.semantics.types.user.EventT
    event: str  # for compatibility with vyper
    topics: list[Any]  # list of decoded topics
    args: list[Any]  # list of decoded args
    args_obj: Any = field(default=None, init=False)

    def __post_init__(self):
        class Args:
            pass

        self.args_obj = Args()
        for name, value in self.ordered_args():
            setattr(self.args_obj, name, value)

    def ordered_args(self):
        # TODO what about raw_log?
        t_i = 0  # topics list does not include the event_id (excluded by decode_log)
        a_i = 0
        b = []
        # align the evm topic + args lists with the way they appear in the source
        # ex. Transfer(indexed address, address, indexed address)
        for is_topic, k in zip(
            self.event_type.indexed, self.event_type.arguments.keys()
        ):
            if is_topic:
                b.append((k, self.topics[t_i]))
                t_i += 1
            else:
                b.append((k, self.args[a_i]))
                a_i += 1

        return b

    def __repr__(self):
        b = self.ordered_args()
        args = ", ".join(f"{k}={v}" for k, v in b)
        return f"{self.event_type.name}({args})"


@dataclass
class RawEvent:
    event_data: Any
