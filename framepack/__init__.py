"""Utilities for FramePack-specific helpers.

This module dynamically loads helper utilities stored in files with
names that are not valid Python identifiers (for example, ``flf-helpers``).
Importing :mod:`framepack` exposes those helpers as attributes so they can be
used as a regular Python module.
"""

from __future__ import annotations

from importlib import util
from pathlib import Path

__all__ = [
    "flf_helpers",
    "load_image",
    "encode_vae",
    "make_schedule",
    "noise_latent_to_sigma",
    "anchor_x0",
]

_module_path = Path(__file__).resolve().with_name("flf-helpers.py")
_spec = util.spec_from_file_location("framepack._flf_helpers", _module_path)
_flf_helpers = util.module_from_spec(_spec)
assert _spec and _spec.loader
del _module_path
_spec.loader.exec_module(_flf_helpers)  # type: ignore[attr-defined]

flf_helpers = _flf_helpers

load_image = _flf_helpers.load_image
encode_vae = _flf_helpers.encode_vae
make_schedule = _flf_helpers.make_schedule
noise_latent_to_sigma = _flf_helpers.noise_latent_to_sigma
anchor_x0 = _flf_helpers.anchor_x0

