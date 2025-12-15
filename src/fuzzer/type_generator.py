import random
from typing import List, Optional, Set, Callable, Dict

from vyper.semantics.types import (
    VyperType,
    IntegerT,
    BoolT,
    AddressT,
    BytesT,
    BytesM_T,
    StringT,
    DArrayT,
    SArrayT,
    HashMapT,
    TupleT,
    StructT,
)


class TypeGenerator:
    """Generates random Vyper types for fuzzing."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.struct_counter = 0
        self.source_fragments: List[str] = []

        # Type categories
        self.leaf_types = [BoolT, AddressT, IntegerT, BytesM_T, BytesT, StringT]
        self.complex_types = [SArrayT, DArrayT, TupleT, StructT, HashMapT]

        # Type generator functions
        self.type_generators: Dict[type, Callable] = {
            BoolT: self._generate_bool,
            AddressT: self._generate_address,
            IntegerT: self._generate_integer,
            BytesM_T: self._generate_bytes_m,
            BytesT: self._generate_bytes,
            StringT: self._generate_string,
            SArrayT: self._generate_sarray,
            DArrayT: self._generate_darray,
            HashMapT: self._generate_hashmap,
            TupleT: self._generate_tuple,
            StructT: self._generate_struct,
        }

    def generate_type(
        self,
        nesting: int = 3,
        skip: Optional[Set[type]] = None,
        size_budget: Optional[int] = None,
    ) -> VyperType:
        """Generate a random Vyper type.

        Struct definitions are accumulated in self.source_fragments.

        Args:
            nesting: Maximum nesting depth for complex types
            skip: Types to exclude from generation
            size_budget: Optional memory budget constraint

        Returns:
            Generated VyperType
        """
        assert nesting >= 0

        skip = skip or set()

        # Choose type category based on nesting
        if nesting == 0:
            available_types = [t for t in self.leaf_types if t not in skip]
        else:
            all_types = self.leaf_types + self.complex_types
            available_types = [t for t in all_types if t not in skip]

        if not available_types:
            raise ValueError(f"No available types after filtering. Skipped: {skip}")

        # Choose a type constructor
        type_ctor = self.rng.choice(available_types)

        # Generate the type using the appropriate generator
        generator = self.type_generators[type_ctor]
        return generator(nesting, skip, size_budget)

    # ---------- helpers for budgeted sizing ----------
    def _shrink_length_to_fit(self, initial_len: int, make_type, budget: int) -> int:
        """
        Reduce an integer length until make_type(length).memory_bytes_required <= budget.
        Uses geometric reduction to speed up the search.
        """
        length = max(1, initial_len)
        while length > 1 and make_type(length).memory_bytes_required > budget:
            length = max(1, length // 2)
        return length

    def _max_reps_for_budget(
        self, elem_size: int, budget: int, overhead: int = 0
    ) -> int:
        if budget <= overhead:
            return 1
        return max(1, (budget - overhead) // max(1, elem_size))

    def _generate_elements_with_budget(
        self,
        n: int,
        nesting: int,
        skip: Set[type],
        size_budget: Optional[int],
    ) -> List[VyperType]:
        elements: List[VyperType] = []
        if size_budget is None:
            for _ in range(n):
                elem_t = self.generate_type(nesting - 1, skip, size_budget=None)
                elements.append(elem_t)
            return elements

        remaining = size_budget
        for _ in range(n):
            elem_t = self.generate_type(nesting - 1, skip, size_budget=remaining)
            elements.append(elem_t)
            used = elem_t.memory_bytes_required
            remaining = max(0, remaining - used)
        return elements

    def _generate_bool(
        self,
        _nesting: int,
        _skip: Set[type],
        _size_budget: Optional[int],
    ) -> BoolT:
        return BoolT()

    def _generate_address(
        self,
        _nesting: int,
        _skip: Set[type],
        _size_budget: Optional[int],
    ) -> AddressT:
        return AddressT()

    def _generate_integer(
        self,
        _nesting: int,
        _skip: Set[type],
        _size_budget: Optional[int],
    ) -> IntegerT:
        signed = self.rng.choice([True, False])
        bits = 8 * self.rng.randint(1, 32)  # 8 to 256 bits in steps of 8
        return IntegerT(signed, bits)

    def _generate_bytes_m(
        self,
        _nesting: int,
        _skip: Set[type],
        _size_budget: Optional[int],
    ) -> BytesM_T:
        m = self.rng.randint(1, 32)
        return BytesM_T(m)

    def _generate_bytes(
        self,
        _nesting: int,
        _skip: Set[type],
        size_budget: Optional[int],
    ) -> BytesT:
        max_length = self.rng.randint(1, 1024)
        if size_budget is not None:
            max_length = self._shrink_length_to_fit(
                max_length, lambda x: BytesT(x), size_budget
            )
        return BytesT(max_length)

    def _generate_string(
        self,
        _nesting: int,
        _skip: Set[type],
        size_budget: Optional[int],
    ) -> StringT:
        max_length = self.rng.randint(1, 1024)
        if size_budget is not None:
            max_length = self._shrink_length_to_fit(
                max_length, lambda x: StringT(x), size_budget
            )
        return StringT(max_length)

    def _generate_sarray(
        self,
        nesting: int,
        skip: Set[type],
        size_budget: Optional[int],
    ) -> SArrayT:
        # Skip types that can't be in arrays
        element_skip = skip | {TupleT, BytesT, StringT, HashMapT}
        element_type = self.generate_type(nesting - 1, element_skip, size_budget)
        length = self.rng.randint(1, 6)
        if size_budget is not None:
            elem_size = max(1, element_type.memory_bytes_required)
            max_len = self._max_reps_for_budget(elem_size, size_budget, overhead=0)
            length = max(1, min(length, max_len))
        return SArrayT(element_type, length)

    def _generate_darray(
        self,
        nesting: int,
        skip: Set[type],
        size_budget: Optional[int],
    ) -> DArrayT:
        # Dynamic arrays can't contain tuples or hashmaps
        element_skip = skip | {TupleT, HashMapT}
        element_type = self.generate_type(nesting - 1, element_skip, size_budget)
        max_length = self.rng.randint(1, 16)
        if size_budget is not None:
            elem_size = max(1, element_type.memory_bytes_required)
            allowed = self._max_reps_for_budget(elem_size, size_budget, overhead=32)
            max_length = max(1, min(max_length, allowed))
        return DArrayT(element_type, max_length)

    def _generate_hashmap(
        self,
        nesting: int,
        skip: Set[type],
        _size_budget: Optional[int],
    ) -> HashMapT:
        # Key must be a hashable type (leaf types only)
        hashable_types = [BoolT, AddressT, IntegerT, BytesM_T]
        key_available = [t for t in hashable_types if t not in skip]

        if not key_available:
            raise ValueError(
                f"No hashable types available for HashMap key. Skipped: {skip}"
            )

        key_ctor = self.rng.choice(key_available)
        key_generator = self.type_generators[key_ctor]
        key_type = key_generator(0, skip, None)  # Keys are always leaf types

        # Value can be any type
        value_type = self.generate_type(nesting - 1, skip, size_budget=None)
        return HashMapT(key_type, value_type)

    def _generate_tuple(
        self,
        nesting: int,
        skip: Set[type],
        size_budget: Optional[int],
    ) -> TupleT:
        # Tuples must have at least 1 element
        n_elements = self.rng.randint(1, 6)
        element_skip = skip | {HashMapT}  # Tuples can't contain hashmaps

        elements = self._generate_elements_with_budget(
            n_elements, nesting, element_skip, size_budget
        )

        return TupleT(elements)

    def _generate_struct(
        self,
        nesting: int,
        skip: Set[type],
        size_budget: Optional[int],
    ) -> StructT:
        n_fields = self.rng.randint(1, 6)
        field_skip = skip | {HashMapT}  # Structs can't contain hashmaps

        fields = {}
        element_types = self._generate_elements_with_budget(
            n_fields, nesting, field_skip, size_budget
        )
        for i, field_type in enumerate(element_types):
            field_name = f"x{i}"
            fields[field_name] = field_type

        # Generate unique struct name
        struct_name = f"MyStruct{self.struct_counter}"
        self.struct_counter += 1

        # Create struct type
        struct_type = StructT(struct_name, fields)

        # Generate source fragment and store internally
        if hasattr(struct_type, "def_source_str"):
            fragment = struct_type.def_source_str()
        else:
            # Fallback if the method doesn't exist
            struct_def_lines = [f"struct {struct_name}:"]
            for field_name, field_type in fields.items():
                struct_def_lines.append(f"    {field_name}: {field_type}")
            fragment = "\n".join(struct_def_lines)

        if fragment not in self.source_fragments:
            self.source_fragments.append(fragment)

        return struct_type

    def generate_simple_type(self) -> VyperType:
        return self.generate_type(nesting=0)

    def generate_integer_type(self) -> IntegerT:
        return self._generate_integer(0, set(), None)
