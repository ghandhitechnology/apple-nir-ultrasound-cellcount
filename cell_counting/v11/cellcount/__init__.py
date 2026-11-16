# Public interface of the v11 cell counter: configure the settings once, then count an image and
# draw its overlay.

from .core import DEFAULT, SQUARE, configure, segment_area
from .count import Result, count_image, overlay

__version__ = "11.0"
