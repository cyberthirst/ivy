from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from vyper.compiler.phases import CompilerData
from vyper.exceptions import (
    BadArchive,
    JSONError,
    ParserException,
    VyperException,
    VyperInternalException,
)


class CompilationOutcome(Enum):
    SUCCESS = auto()
    COMPILATION_FAILURE = auto()  # invalid code rejected (VyperException)
    COMPILER_CRASH = auto()  # bug in the compiler (ICE or bare Exception)


@dataclass
class CompilationResult:
    outcome: CompilationOutcome
    compiler_data: Optional[CompilerData] = None
    error: Optional[Exception] = None

    @property
    def is_success(self) -> bool:
        return self.outcome is CompilationOutcome.SUCCESS

    @property
    def is_compilation_failure(self) -> bool:
        return self.outcome is CompilationOutcome.COMPILATION_FAILURE

    @property
    def is_compiler_crash(self) -> bool:
        return self.outcome is CompilationOutcome.COMPILER_CRASH


def classify_compilation_error(e: Exception) -> CompilationOutcome:
    if isinstance(e, VyperInternalException):
        return CompilationOutcome.COMPILER_CRASH
    if isinstance(e, (ParserException, JSONError, BadArchive, VyperException)):
        return CompilationOutcome.COMPILATION_FAILURE
    # bare Python exception = unwrapped ICE
    return CompilationOutcome.COMPILER_CRASH


def compile_vyper(source: str) -> CompilationResult:
    try:
        data = CompilerData(source)
        data.bytecode  # trigger full compilation
        data.bytecode_runtime
        return CompilationResult(CompilationOutcome.SUCCESS, compiler_data=data)
    except Exception as e:
        return CompilationResult(classify_compilation_error(e), error=e)
