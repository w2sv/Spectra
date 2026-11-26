# Spectra — Engineering Guidelines

Spectra is a Python application for real-time music analysis and visualization. It captures desktop audio, extracts
musical features, and produces RGB frames for a virtual LED strip and physical WLED controllers over DDP.

The project prioritizes clean, modular, readable code and experimentation over unnecessary abstraction or architectural
complexity.

## Python and tooling

- Use Python 3.15 and modern, idiomatic Python.
- Manage Python versions, virtual environments, and dependencies exclusively through **uv**.
- Use **Pyright** for static type checking, **Ruff** for linting and formatting, and **pytest** for testing.
- Keep project and tooling configuration centralized in `pyproject.toml`.
- Use existing standard-library functionality before introducing additional dependencies.
- Do not introduce alternative dependency managers, formatters, linters, or type checkers.
- Keep `uv.lock` synchronized with dependency changes.

## Code quality

- Prioritize readability, simplicity, and maintainability.
- Follow established Python conventions and PEP 8, using Ruff as the formatting authority.
- Prefer descriptive names over abbreviations or unnecessarily verbose identifiers.
- Keep functions focused and reasonably small, but avoid excessive fragmentation.
- Prefer straightforward control flow and explicit behavior over clever abstractions.
- Avoid unnecessary boilerplate, defensive checks, and redundant state.
- Avoid premature optimization. Optimize measured bottlenecks rather than hypothetical ones.
- Use exceptions for exceptional conditions rather than obscuring failures with fallback values.
- Add comments and docstrings when they explain non-obvious behavior, important constraints, or design decisions. Avoid
  restating what the code already expresses.
- Do not introduce compatibility code for Python versions older than the project's minimum supported version.
- Initialize instance attributes explicitly in __init__.

## Architecture and modularity

- Organize code around clear responsibilities and cohesive modules.
- Maintain implementation-agnostic boundaries between audio capture, signal analysis, visualization, and output.
- Keep domain logic independent of specific frameworks, hardware, and transport mechanisms.
- Prefer composition over inheritance.
- Use `Protocol` for meaningful interchangeable implementations, rather than requiring inheritance from abstract base
  classes.
- Do not introduce interfaces for components that have no meaningful reason to be abstracted.
- Prefer ordinary classes and functions over elaborate design patterns.
- Use explicit constructor dependencies where appropriate. Do not introduce dependency-injection frameworks.
- Avoid generic managers, service layers, factories, and other abstractions unless they solve an actual problem.
- Do not create packages or modules speculatively. Let the structure evolve with functionality.
- Avoid catch-all modules such as `utils.py` or `common.py` when functionality has a more appropriate home.

Abstractions should make components easier to understand, test, or replace—not merely increase the number of layers.

## Typing and data modeling

- Use modern Python type annotations consistently.
- Explicitly type public APIs, important internal interfaces, and significant data structures.
- Rely on type inference for straightforward local variables.
- Prefer specific types over `Any`.
- Avoid unnecessary casts, type ignores, and workarounds that conceal actual typing problems.
- Use dataclasses for simple structured data where appropriate.
- Prefer immutable data models for values shared between components.
- Use NumPy arrays for numerical data and annotate their dtypes using `numpy.typing` where useful.
- Document significant array conventions, particularly dimensions, channel ordering, and data types.
- Validate array shapes and other invariants at meaningful boundaries rather than repeatedly throughout internal code.

Write code that works naturally with Pyright rather than weakening type checking to accommodate poor design.

## Real-time processing and concurrency

- Keep audio capture, signal processing, visualization, and presentation logically independent.
- Do not allow blocking audio operations to stall the GUI.
- Keep Qt widget interaction on the main GUI thread.
- Use the simplest concurrency mechanism that satisfies the requirements. Avoid introducing threads, asynchronous
  frameworks, or synchronization primitives unnecessarily.
- Make shared-state ownership and thread boundaries explicit.
- Prefer bounded buffering and predictable behavior over unbounded queues.
- Avoid accumulating stale audio or visualization frames when processing falls behind.
- Preserve discrete musical events until consumed, while allowing continuously updated values to be superseded.
- Use monotonic time for elapsed-time measurements, animation timing, and synchronization.
- Ensure resources such as subprocesses, streams, and threads are released reliably.

Favor predictable latency and responsiveness over processing every intermediate frame.

## Signal processing and visualization

- Keep signal-processing algorithms independent of audio capture backends and UI frameworks.
- Separate continuous audio features from discrete musical events.
- Keep visualization algorithms independent of their output destination.
- Represent rendered LED colors using a consistent RGB frame format.
- Separate configurable visualization parameters from algorithm implementations.
- Prefer deterministic processing where practical.
- Avoid hardcoding genre-specific assumptions into shared audio-analysis components.
- Keep experimental algorithms easy to modify or replace without affecting unrelated components.

Do not introduce unnecessary frameworks for managing visualizations or audio-processing pipelines.

## Testing

Keep automated testing deliberately minimal.

- Test integral, deterministic behavior where incorrect results would compromise the application's fundamental
  functionality.
- Prioritize numerical correctness, data integrity, and important boundary conditions.
- Use synthetic audio signals and other predictable inputs for DSP tests.
- Avoid extensive mocking and tests coupled to implementation details.
- Do not test every configuration parameter or minor implementation detail.
- Avoid brittle assertions about exact visualization colors, animation trajectories, or other intentionally experimental
  behavior.
- Do not add tests merely to increase coverage.
- Prefer manual visual evaluation for rendering behavior and subjective visualization quality.

Tests should protect important invariants without obstructing experimentation.

## Dependencies and configuration

- Add dependencies only when they provide a clear, practical benefit.
- Prefer established libraries for complex functionality such as numerical processing and GUI rendering.
- Avoid adding frameworks to solve problems that require only a small amount of straightforward code.
- Keep configuration simple and introduce persistence only when needed.
- Avoid duplicating configuration values across modules.
- Do not introduce environment variables, configuration files, or command-line options without a concrete use case.

## Development workflow

- Run uv commands needed for the requested task without asking for permission. If the default tooling cache is not
  writable, use a writable repository-local or temporary cache.
- Do not ask for permission for routine file or directory operations inside this repository, including creating
  directories and writing or copying command output and generated artifacts.
- Make focused, cohesive changes that address the requested task.
- Follow existing conventions rather than introducing competing approaches.
- Avoid unrelated refactoring and speculative improvements.
- Remove obsolete code instead of retaining unused compatibility paths.
- Run relevant Ruff and Pyright checks after changes.
- Run focused tests when modifying functionality covered by tests.
- Do not introduce CI pipelines, code-generation tools, task runners, or additional infrastructure without a concrete
  need.
- Follow [the documentation guidelines](docs/documentation-guidelines.md) when creating or modifying documentation
  anywhere in the repository.

## Guiding principle

**Favor the simplest implementation that preserves meaningful architectural boundaries.**

The project should remain easy to understand, modify, and experiment with. Clean architecture is valuable, but
abstractions, tests, and infrastructure must justify their maintenance cost.
