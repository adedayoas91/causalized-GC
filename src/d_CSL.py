"""Compatibility imports for the historical ``src.d_CSL`` module.

New code should import :class:`causalized_gc.GcStar`, which automatically
selects a CPU or compatible NVIDIA backend. Existing notebooks may continue to
import this module while they migrate.
"""

from causalized_gc.backends.cpu import *  # noqa: F401,F403
