from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, TypeAlias

NetworkNode: TypeAlias = Hashable
GridNode: TypeAlias = NetworkNode
Scenario: TypeAlias = dict[str, Any]
Candidate: TypeAlias = dict[str, Any]
CandidateConstraint: TypeAlias = Callable[[Candidate], str | None]
