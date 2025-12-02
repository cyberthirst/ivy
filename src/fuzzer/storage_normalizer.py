from typing import Any, Dict

from vyper.semantics.types import (
    DArrayT,
    HashMapT,
    SArrayT,
    StructT,
    TupleT,
    VyperType,
)

from ivy.expr.default_values import get_default_value
from ivy.frontend.decoder_utils import decode_ivy_object


def normalize_storage_dump(
    storage_dump: Dict[str, Any], contract: Any
) -> Dict[str, Any]:
    """Normalize a storage dump by stripping explicit default values.

    If the two backends produce logically equivalent storage (even when one
    materializes default values and the other does not), the normalized dumps
    will compare equal.
    """

    if storage_dump is None or contract is None:
        return storage_dump

    type_map = _extract_storage_types(contract)
    normalized: Dict[str, Any] = {}

    for name, value in storage_dump.items():
        vyper_type = type_map.get(name)
        assert vyper_type is not None, f"Unknown storage variable: {name}"

        pruned_value = _prune_defaults(value, vyper_type)
        if _is_default_value(pruned_value, vyper_type):
            continue

        normalized[name] = pruned_value

    return normalized


def _extract_storage_types(contract: Any) -> Dict[str, VyperType]:
    """Return a mapping from storage variable name to its Vyper type."""

    module_t = contract.compiler_data.global_ctx
    module_variables = module_t.variables

    storage_types: Dict[str, VyperType] = {}

    for name, varinfo in module_variables.items():
        if varinfo.is_storage:
            storage_types[name] = varinfo.typ

    return storage_types


def _prune_defaults(value: Any, vyper_type: VyperType) -> Any:
    """Remove default-valued entries from complex storage structures."""

    if isinstance(vyper_type, HashMapT) and isinstance(value, dict):
        pruned: Dict[Any, Any] = {}
        for key, entry_value in value.items():
            normalized_entry = _prune_defaults(entry_value, vyper_type.value_type)
            if not _is_default_value(normalized_entry, vyper_type.value_type):
                pruned[key] = normalized_entry
        return pruned

    if isinstance(vyper_type, StructT) and isinstance(value, dict):
        pruned_struct: Dict[str, Any] = {}
        for field_name, field_type in vyper_type.members.items():
            if field_name not in value:
                continue
            pruned_field = _prune_defaults(value[field_name], field_type)
            if _is_default_value(pruned_field, field_type):
                continue
            pruned_struct[field_name] = pruned_field
        return pruned_struct

    if isinstance(vyper_type, (SArrayT, DArrayT)) and isinstance(value, list):
        element_type = vyper_type.value_type
        return [_prune_defaults(item, element_type) for item in value]

    if isinstance(vyper_type, TupleT) and isinstance(value, (tuple, list)):
        items = []
        for (sub_name, sub_type), element in zip(vyper_type.tuple_items(), value):
            items.append(_prune_defaults(element, sub_type))
        return tuple(items) if isinstance(value, tuple) else items

    return value


def _is_default_value(value: Any, vyper_type: VyperType) -> bool:
    """Return True if *value* is indistinguishable from the default for *vyper_type*."""

    if isinstance(vyper_type, HashMapT):
        if not isinstance(value, dict):
            return False
        return all(
            _is_default_value(entry, vyper_type.value_type) for entry in value.values()
        )

    if isinstance(vyper_type, StructT):
        if not isinstance(value, dict):
            return False

        member_types = vyper_type.members
        # Reject unexpected keys so we don't silently hide mutations.
        if any(key not in member_types for key in value):
            return False

        for field_name, field_type in member_types.items():
            if field_name not in value:
                continue
            if not _is_default_value(value[field_name], field_type):
                return False

        return True

    if isinstance(vyper_type, DArrayT):
        if not isinstance(value, list):
            return False
        # A dynamic array differs from default as soon as its length is non-zero.
        return len(value) == 0

    if isinstance(vyper_type, SArrayT):
        if not isinstance(value, list):
            return False
        return all(_is_default_value(item, vyper_type.value_type) for item in value)

    if isinstance(vyper_type, TupleT):
        if not isinstance(value, (list, tuple)):
            return False
        elements = list(value)
        element_types = [subtype for _, subtype in vyper_type.tuple_items()]
        if len(elements) != len(element_types):
            return False
        return all(
            _is_default_value(elem, subtype)
            for elem, subtype in zip(elements, element_types)
        )

    default = get_default_value(vyper_type)
    default = decode_ivy_object(default, vyper_type)
    return value == default
