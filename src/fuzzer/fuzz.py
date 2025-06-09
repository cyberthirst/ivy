"""
Differential fuzzer for Vyper using test exports.

This module loads test exports, optionally mutates them, and compares
execution between Ivy and the Vyper compiler (via Boa).
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

from ivy.frontend.loader import loads as ivy_loads
from boa import loads as boa_loads

from .mutator import AstMutator
from .export_utils import (
    load_all_exports,
    filter_exports,
    extract_test_cases,
    TestFilter,
)
from ..unparser.unparser import unparse


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class DifferentialFuzzer:
    """Fuzzer that uses Vyper test exports for differential testing."""

    def __init__(
        self,
        exports_dir: Path = Path("tests/vyper-exports"),
        seed: Optional[int] = None,
    ):
        self.exports_dir = exports_dir
        self.rng = random.Random(seed)
        self.mutator = AstMutator(self.rng)

    def load_filtered_exports(self, test_filter: Optional[TestFilter] = None) -> Dict:
        """Load and filter test exports."""
        exports = load_all_exports(self.exports_dir)

        if test_filter:
            exports = filter_exports(exports, test_filter=test_filter)

        return exports

    def mutate_source(self, source: str) -> Optional[str]:
        """Mutate source code and return the mutated version."""
        try:
            # Parse the source into AST
            import vyper

            ast = vyper.ast.parse_to_ast(source)

            # Mutate the AST
            mutated_ast = self.mutator.mutate(ast)

            # Unparse back to source
            return unparse(mutated_ast)
        except Exception as e:
            logging.debug(f"Failed to mutate source: {e}")
            return None

    def run_ivy_calls(self, source: str, calldatas: List[str]) -> dict:
        """
        Compile and execute each calldata against the Ivy interpreter.
        Returns either a load-time exception or a list of per-call results.
        """
        try:
            contract = ivy_loads(source)
        except Exception as e:
            return {"load_error": e}

        results = []
        for hexstr in calldatas:
            data = bytes.fromhex(hexstr)
            try:
                output = contract.message_call(data=data)
                storage = contract.storage_dump()
                results.append({"data": hexstr, "output": output, "storage": storage})
            except Exception as e:
                results.append({"data": hexstr, "runtime_error": e})
        return {"results": results}

    def run_boa_calls(self, source: str, calldatas: List[str]) -> dict:
        """
        Compile and execute each calldata against the Boa interpreter.
        Returns either a load-time exception or a list of per-call results.
        """
        try:
            contract = boa_loads(source)
        except Exception as e:
            return {"load_error": e}

        results = []
        for hexstr in calldatas:
            data = bytes.fromhex(hexstr)
            try:
                output = contract.env.raw_call(to_address=contract.address, data=data)
                storage = contract._storage.to_dict()
                results.append({"data": hexstr, "output": output, "storage": storage})
            except Exception as e:
                results.append({"data": hexstr, "runtime_error": e})
        return {"results": results}

    def compare_runs(
        self, source: str, calldatas: List[str], original_source: Optional[str] = None
    ):
        """Compare execution between Ivy and Boa."""
        ivy_res = self.run_ivy_calls(source, calldatas)
        boa_res = self.run_boa_calls(source, calldatas)

        # Compare load-time behavior first
        ivy_load_err = ivy_res.get("load_error")
        boa_load_err = boa_res.get("load_error")
        if ivy_load_err or boa_load_err:
            if (ivy_load_err is None) != (boa_load_err is None):
                # Skip known risky overlap errors
                if (
                    boa_load_err
                    and hasattr(boa_load_err, "message")
                    and boa_load_err.message == "risky overlap"
                ):
                    return

                logging.error("Load-time mismatch for contract:")
                if original_source:
                    logging.error("Original source:\n%s", original_source)
                logging.error("Mutated source:\n%s", source)
                logging.error("  Ivy error: %r", ivy_load_err)
                logging.error("  Boa error: %r", boa_load_err)
            return

        # Both loaded OK, compare per-call results
        ivy_results = ivy_res["results"]
        boa_results = boa_res["results"]

        for iv, bv in zip(ivy_results, boa_results):
            data = iv["data"]
            ivy_err = iv.get("runtime_error")
            boa_err = bv.get("runtime_error")

            # Exception divergence
            if (ivy_err is None) != (boa_err is None):
                logging.error("Runtime error mismatch for payload %s", data)
                if original_source:
                    logging.error("Original source:\n%s", original_source)
                logging.error("Mutated source:\n%s", source)
                logging.error("  Ivy error: %r", ivy_err)
                logging.error("  Boa error: %r", boa_err)
                continue

            # If both errored, consider them matching
            if ivy_err:
                continue

            # Compare outputs
            if iv["output"] != bv["output"]:
                logging.error("Output mismatch for payload %s", data)
                if original_source:
                    logging.error("Original source:\n%s", original_source)
                logging.error("Mutated source:\n%s", source)
                logging.error("  Ivy output: %r", iv["output"])
                logging.error("  Boa output: %r", bv["output"])

            # Compare storage
            if iv["storage"] != bv["storage"]:
                logging.error("Storage mismatch for payload %s", data)
                if original_source:
                    logging.error("Original source:\n%s", original_source)
                logging.error("Mutated source:\n%s", source)
                logging.error("  Ivy storage: %r", iv["storage"])
                logging.error("  Boa storage: %r", bv["storage"])

    def fuzz_exports(
        self,
        test_filter: Optional[TestFilter] = None,
        max_mutations_per_test: int = 5,
        enable_mutations: bool = True,
    ):
        """Main fuzzing loop using test exports."""
        # Load and filter exports
        exports = self.load_filtered_exports(test_filter)
        logging.info(
            f"Loaded {sum(len(e.items) for e in exports.values())} test items from {len(exports)} files"
        )

        # Extract test cases
        test_cases = extract_test_cases(exports)
        logging.info(f"Extracted {len(test_cases)} test cases")

        # Run differential testing
        for i, (source, calldatas) in enumerate(test_cases):
            if not calldatas:
                continue

            logging.debug(
                f"Testing case {i + 1}/{len(test_cases)} with {len(calldatas)} calls"
            )

            # First, test without mutations to ensure baseline correctness
            logging.debug("Testing original source")
            self.compare_runs(source, calldatas)

            # Then test with mutations if enabled
            if enable_mutations:
                for mutation_round in range(max_mutations_per_test):
                    mutated_source = self.mutate_source(source)
                    if mutated_source and mutated_source != source:
                        logging.debug(f"Testing mutation {mutation_round + 1}")
                        self.compare_runs(
                            mutated_source, calldatas, original_source=source
                        )


def main():
    """Run differential fuzzing with test exports."""
    # Create test filter
    test_filter = TestFilter()
    # Exclude tests with certain patterns
    test_filter.exclude_path(r"test_raw_call")  # Skip raw call tests
    test_filter.exclude_source(r"@nonreentrant")  # Skip nonreentrant tests
    test_filter.exclude_source(r"raw_log\(")  # Skip raw log tests

    # Create and run fuzzer
    fuzzer = DifferentialFuzzer()
    fuzzer.fuzz_exports(
        test_filter=test_filter, max_mutations_per_test=3, enable_mutations=True
    )


if __name__ == "__main__":
    main()
