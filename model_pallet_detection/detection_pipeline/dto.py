from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
Point = Tuple[int, int]


@dataclass
class Box:
    top: int
    right: int
    bottom: int
    left: int
    score: float
    name: str
    id: Optional[int] = None

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> Point:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    @property
    def size(self) -> int:
        return self.width * self.height


@dataclass
class InstMask:
    top: int
    right: int
    bottom: int
    left: int
    mask: Image
    score: float
    name: str
    id: Optional[int] = None

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> Point:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    @property
    def area(self) -> int:
        return self.mask.sum()


@dataclass
class Contour:
    contour: NDArray[np.int32]
    name: str
