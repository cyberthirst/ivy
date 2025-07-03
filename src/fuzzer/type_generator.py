import random
from typing import List, Tuple, Optional, Set, Callable, Dict

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
        source_fragments: Optional[List[str]] = None,
    ) -> Tuple[VyperType, List[str]]:
        """Generate a random Vyper type and associated source fragments.

        Args:
            nesting: Maximum nesting depth for complex types
            skip: Types to exclude from generation
            source_fragments: List to append struct definitions to

        Returns:
            Tuple of (generated_type, source_fragments)
        """
        assert nesting >= 0

        skip = skip or set()
        if source_fragments is None:
            source_fragments = []

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
        generated_type = generator(nesting, skip, source_fragments)

        return generated_type, source_fragments

    def _generate_bool(self, _: int, _skip: Set[type], _fragments: List[str]) -> BoolT:
        return BoolT()

    def _generate_address(
        self, _: int, _skip: Set[type], _fragments: List[str]
    ) -> AddressT:
        return AddressT()

    def _generate_integer(
        self, _: int, _skip: Set[type], _fragments: List[str]
    ) -> IntegerT:
        signed = self.rng.choice([True, False])
        bits = 8 * self.rng.randint(1, 32)  # 8 to 256 bits in steps of 8
        return IntegerT(signed, bits)

    def _generate_bytes_m(
        self, _: int, _skip: Set[type], _fragments: List[str]
    ) -> BytesM_T:
        m = self.rng.randint(1, 32)
        return BytesM_T(m)

    def _generate_bytes(
        self, _: int, _skip: Set[type], _fragments: List[str]
    ) -> BytesT:
        max_length = self.rng.randint(1, 1024)
        return BytesT(max_length)

    def _generate_string(
        self, _: int, _skip: Set[type], _fragments: List[str]
    ) -> StringT:
        max_length = self.rng.randint(1, 1024)
        return StringT(max_length)

    def _generate_sarray(
        self, nesting: int, skip: Set[type], fragments: List[str]
    ) -> SArrayT:
        # Skip types that can't be in arrays
        element_skip = skip | {TupleT, BytesT, StringT, HashMapT}
        element_type, _ = self.generate_type(nesting - 1, element_skip, fragments)
        length = self.rng.randint(1, 6)
        return SArrayT(element_type, length)

    def _generate_darray(
        self, nesting: int, skip: Set[type], fragments: List[str]
    ) -> DArrayT:
        # Dynamic arrays can't contain tuples or hashmaps
        element_skip = skip | {TupleT, HashMapT}
        element_type, _ = self.generate_type(nesting - 1, element_skip, fragments)
        max_length = self.rng.randint(1, 16)
        return DArrayT(element_type, max_length)

    def _generate_hashmap(
        self, nesting: int, skip: Set[type], fragments: List[str]
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
        key_type = key_generator(0, skip, fragments)  # Keys are always leaf types

        # Value can be any type
        value_type, _ = self.generate_type(nesting - 1, skip, fragments)
        return HashMapT(key_type, value_type)

    def _generate_tuple(
        self, nesting: int, skip: Set[type], fragments: List[str]
    ) -> TupleT:
        # Tuples must have at least 1 element
        n_elements = self.rng.randint(1, 6)
        element_skip = skip | {HashMapT}  # Tuples can't contain hashmaps

        elements = []
        for _ in range(n_elements):
            elem_type, _ = self.generate_type(nesting - 1, element_skip, fragments)
            elements.append(elem_type)

        return TupleT(elements)

    def _generate_struct(
        self, nesting: int, skip: Set[type], fragments: List[str]
    ) -> StructT:
        n_fields = self.rng.randint(1, 6)
        field_skip = skip | {HashMapT}  # Structs can't contain hashmaps

        fields = {}
        for i in range(n_fields):
            field_name = f"x{i}"
            field_type, _ = self.generate_type(nesting - 1, field_skip, fragments)
            fields[field_name] = field_type

        # Generate unique struct name
        struct_name = f"MyStruct{self.struct_counter}"
        self.struct_counter += 1

        # Create struct type
        struct_type = StructT(struct_name, fields)

        # Use the struct's built-in method for source generation
        # Based on vyper codebase, StructT should have def_source_str()
        if hasattr(struct_type, "def_source_str"):
            fragments.append(struct_type.def_source_str())
        else:
            # Fallback if the method doesn't exist
            struct_def_lines = [f"struct {struct_name}:"]
            for field_name, field_type in fields.items():
                struct_def_lines.append(f"    {field_name}: {field_type}")
            fragments.append("\n".join(struct_def_lines))

        return struct_type

    def generate_simple_type(self) -> Tuple[VyperType, List[str]]:
        return self.generate_type(nesting=0)

    def generate_integer_type(self) -> IntegerT:
        return self._generate_integer(0, set(), [])
