"""
Main entry point for the differential fuzzer.

This module provides backwards compatibility by importing from the new
differential_fuzzer module.
"""

from src.fuzzer.differential_fuzzer import DifferentialFuzzer, main


if __name__ == "__main__":
    main()
