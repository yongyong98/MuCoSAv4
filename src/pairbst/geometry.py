"""Canonical PAIR-BST ROI geometry and deterministic patch sampling."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Iterable, Iterator, Sequence, TypeVar

from PIL import Image


ROI_SIZE = 4096
CENTER_SIZE = 224
CENTER_START = 1936
CENTER_STOP = 2160
GRID_SIDE = 16
GRID_PATCH_SIZE = 256
GRID_PATCH_COUNT = GRID_SIDE * GRID_SIDE


@dataclass(frozen=True)
class GridLocation:
    """One non-overlapping grid cell in row-major order."""

    index: int
    row: int
    column: int
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class SampledPatch:
    sampling: str
    index: int
    row: int | None
    column: int | None
    image: Image.Image


def validate_roi_image(image: Image.Image, *, exact_size: bool = True) -> None:
    expected = (ROI_SIZE, ROI_SIZE)
    if exact_size and image.size != expected:
        raise ValueError(f"PAIR-BST canonical ROI must be exactly {expected}; got {image.size}")
    if not exact_size and (image.width < ROI_SIZE or image.height < ROI_SIZE):
        raise ValueError(f"ROI must be at least {expected}; got {image.size}")


def center_box() -> tuple[int, int, int, int]:
    """The frozen 224 x 224 center crop: [1936:2160, 1936:2160]."""

    return (CENTER_START, CENTER_START, CENTER_STOP, CENTER_STOP)


def center_crop_224(image: Image.Image, *, validate: bool = True) -> Image.Image:
    if validate:
        validate_roi_image(image)
    return image.crop(center_box())


def grid_locations() -> tuple[GridLocation, ...]:
    locations: list[GridLocation] = []
    for row in range(GRID_SIDE):
        top = row * GRID_PATCH_SIZE
        for column in range(GRID_SIDE):
            left = column * GRID_PATCH_SIZE
            index = row * GRID_SIDE + column
            locations.append(
                GridLocation(
                    index=index,
                    row=row,
                    column=column,
                    box=(left, top, left + GRID_PATCH_SIZE, top + GRID_PATCH_SIZE),
                )
            )
    return tuple(locations)


GRID_LOCATIONS = grid_locations()


def iter_grid_patches(image: Image.Image, *, validate: bool = True) -> Iterator[SampledPatch]:
    """Yield 256 non-overlapping 256 x 256 patches in row-major order."""

    if validate:
        validate_roi_image(image)
    for location in GRID_LOCATIONS:
        yield SampledPatch(
            sampling="grid",
            index=location.index,
            row=location.row,
            column=location.column,
            image=image.crop(location.box),
        )


def iter_center_and_grid(image: Image.Image, *, validate: bool = True) -> Iterator[SampledPatch]:
    """Yield the center patch first, followed by the canonical row-major grid."""

    if validate:
        validate_roi_image(image)
    yield SampledPatch(
        sampling="center",
        index=0,
        row=None,
        column=None,
        image=center_crop_224(image, validate=False),
    )
    yield from iter_grid_patches(image, validate=False)


_T = TypeVar("_T")


def batched(items: Iterable[_T], batch_size: int) -> Iterator[tuple[_T, ...]]:
    """Batch an iterator without materializing all 257 sampled patches."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    iterator = iter(items)
    while batch := tuple(islice(iterator, batch_size)):
        yield batch


def assert_row_major(locations: Sequence[GridLocation] = GRID_LOCATIONS) -> None:
    """Fail fast if a modified grid does not cover the canonical ROI exactly once."""

    if len(locations) != GRID_PATCH_COUNT:
        raise ValueError(f"Expected {GRID_PATCH_COUNT} grid locations, got {len(locations)}")
    boxes = set()
    for index, location in enumerate(locations):
        expected_row, expected_column = divmod(index, GRID_SIDE)
        if (location.index, location.row, location.column) != (
            index,
            expected_row,
            expected_column,
        ):
            raise ValueError(f"Grid location {index} is not row-major: {location}")
        boxes.add(location.box)
    if len(boxes) != GRID_PATCH_COUNT:
        raise ValueError("Grid contains duplicate boxes")
    if locations[0].box != (0, 0, 256, 256) or locations[-1].box != (
        3840,
        3840,
        4096,
        4096,
    ):
        raise ValueError("Grid does not span the full 4096 x 4096 ROI")


assert_row_major()

