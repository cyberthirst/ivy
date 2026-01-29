from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DepthConfig:
    """Shared depth control configuration for generators."""

    root_depth: int = 0  # Starting depth for new generation
    decay_base: float = 0.6  # P(continue) = decay_base ^ depth
    max_depth: int = 5  # Hard cap, always terminate at this depth


@dataclass
class ExprGeneratorConfig:
    """Configuration for expression generation probabilities and weights."""

    # Terminal generation
    terminal_var_ref_prob: float = 0.95  # Prefer var refs over literals
    nonconst_expr_retries: int = 3  # Retries for non-constant expressions

    # Strategy weights
    literal_weight: float = 0.01
    ifexp_weight: float = 0.01
    env_var_weight: float = 0.01
    convert_weight: float = 0.01

    # Convert builtin
    convert_literal_prob: float = 0.1
    convert_max_attempts: int = 3

    # Arithmetic generation
    arithmetic_max_attempts: int = 3

    # Variable reference weight scaling: base + scale * count, capped at max
    var_ref_weight_base: float = 0.5
    var_ref_weight_scale: float = 0.1
    var_ref_weight_max: float = 2.0

    # Subscript weight scaling
    subscript_weight_base: float = 0.5
    subscript_weight_scale: float = 0.2
    subscript_weight_max: float = 2.5

    # Subscript chain length controls (separate from depth)
    subscript_chain_max_steps: int = 3
    subscript_random_chain_max_steps: int = 2
    subscript_hashmap_chain_max_steps: int = 1

    # Index generation for sequences
    index_guard_prob: float = 0.60  # Use len()-guarded index
    index_random_prob: float = 0.35  # Use random expression
    # Remaining (0.05) -> OOB literal that triggers compilation xfail

    # DynArray index biases
    dynarray_small_literal_in_guard_prob: float = 0.75
    dynarray_last_index_in_guard_prob: float = 0.2
    dynarray_small_literal_in_random_prob: float = 0.6

    # OOB literal generation
    oob_cap_vs_cap_plus_one_prob: float = 0.5

    # Function call generation
    use_builtin_when_both_available_prob: float = 0.4
    create_new_function_prob: float = 0.1  # When compatible function exists

    # Builtin: slice
    slice_use_bytes32_source_prob: float = 0.25
    slice_valid_prob: float = 0.7
    # Invalid slice path selection thresholds (cumulative)
    slice_invalid_start_at_len_prob: float = 0.34
    slice_invalid_start_plus_rand_prob: float = 0.67  # Cumulative threshold
    # Remaining -> start = len * 2


@dataclass
class StmtGeneratorConfig:
    """Configuration for statement generation probabilities and weights."""

    # Statement injection
    inject_prob: float = 0.3
    min_stmts: int = 1
    max_stmts: int = 3

    # Statement weights
    vardecl_weight: float = 0.3
    assign_weight: float = 2.0
    append_weight: float = 0.2
    augassign_weight: float = 0.4
    if_weight: float = 0.2

    # Type generation bias
    existing_type_bias_prob: float = 0.4

    # Module-level variable location weights (must sum to 1.0)
    storage_location_weight: float = 0.4
    transient_location_weight: float = 0.2
    immutable_location_weight: float = 0.2
    constant_location_weight: float = 0.2

    # Simple statement generation
    simple_stmt_assign_prob: float = 0.6

    # If statement
    generate_else_branch_prob: float = 0.4

    # Loop terminators (break/continue) inside for loops
    loop_terminator_in_if_prob: float = 0.15
    loop_terminator_direct_prob: float = 0.02
    loop_terminator_force_else_prob: float = 0.2

    # Assignment
    subscript_assignment_prob: float = 0.7
    deref_assignment_prob: float = 0.8
    deref_continue_prob: float = 0.5
    deref_chain_max_steps: int = 3
    self_assign_prob: float = 0.0001
    self_assign_max_retries: int = 5

    # For loop generation
    for_weight: float = 0.1
    for_use_range_prob: float = 0.5  # vs array iteration
    for_prefer_existing_array_prob: float = 0.8  # vs literal array
    for_max_range_stop: int = 10  # cap for range(STOP)


@dataclass
class MutatorConfig:
    """Top-level configuration bundling all mutator settings."""

    expr_depth: DepthConfig = field(
        default_factory=lambda: DepthConfig(root_depth=2)
    )
    stmt_depth: DepthConfig = field(default_factory=DepthConfig)
    expr: ExprGeneratorConfig = field(default_factory=ExprGeneratorConfig)
    stmt: StmtGeneratorConfig = field(default_factory=StmtGeneratorConfig)


# Default configuration instance
DEFAULT_CONFIG = MutatorConfig()
