"""Deterministic float32 pooling for row-major grid embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OnlineMeanMax:
    """Accumulate mean and max without retaining all 256 patch embeddings.

    Rows are added sequentially in canonical row-major order.  Float32
    accumulation is intentional and recorded in feature-file provenance.
    """

    dimension: int
    count: int = 0
    _sum: np.ndarray = field(init=False, repr=False)
    _max: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        self._sum = np.zeros(self.dimension, dtype=np.float32)
        self._max = np.full(self.dimension, -np.inf, dtype=np.float32)

    def update(self, features: np.ndarray) -> None:
        rows = np.asarray(features, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != self.dimension:
            raise ValueError(
                f"Expected [n, {self.dimension}] float features; got {rows.shape}"
            )
        if not np.isfinite(rows).all():
            raise ValueError("Grid embeddings contain NaN or Inf")
        # Sequential updates make the result independent of inference batch size.
        for row in rows:
            np.add(self._sum, row, out=self._sum, casting="unsafe")
            np.maximum(self._max, row, out=self._max)
            self.count += 1

    def finalize(self, *, expected_count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot pool zero embeddings")
        if expected_count is not None and self.count != expected_count:
            raise ValueError(f"Expected {expected_count} grid embeddings, got {self.count}")
        mean = np.divide(self._sum, np.float32(self.count), dtype=np.float32)
        return mean.copy(), self._max.copy()


def pool_grid_features(
    features: np.ndarray,
    *,
    expected_count: int | None = 256,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"Expected a two-dimensional feature array, got {rows.shape}")
    accumulator = OnlineMeanMax(rows.shape[1])
    accumulator.update(rows)
    return accumulator.finalize(expected_count=expected_count)

