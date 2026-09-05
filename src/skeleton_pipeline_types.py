"""Shared types for the skeleton-driven runtime modules."""

from dataclasses import dataclass


@dataclass
class Navigation:
    node_refs: list[tuple[str, str]]
    broad: bool
