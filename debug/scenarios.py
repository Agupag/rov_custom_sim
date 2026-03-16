#!/usr/bin/env python3
"""Deterministic command scenarios shared by debug checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandPhase:
    name: str
    duration_s: float
    surge: float = 0.0
    yaw: float = 0.0
    heave: float = 0.0
    active: float = 1.0


DEFAULT_CONTROL_PATH_SCENARIO = [
    CommandPhase("settle", 1.5, 0.0, 0.0, 0.0),
    CommandPhase("surge_forward", 2.0, 1.0, 0.0, 0.0),
    CommandPhase("yaw_right", 1.5, 0.0, 0.8, 0.0),
    CommandPhase("yaw_left", 1.5, 0.0, -0.8, 0.0),
    CommandPhase("heave_up", 1.5, 0.0, 0.0, 1.0),
    CommandPhase("coast", 1.5, 0.0, 0.0, 0.0),
]


DEFAULT_PHYSICS_SANITY_SCENARIO = [
    CommandPhase("settle", 2.0, 0.0, 0.0, 0.0),
    CommandPhase("surge", 3.0, 1.0, 0.0, 0.0),
    CommandPhase("coast", 2.0, 0.0, 0.0, 0.0),
    CommandPhase("yaw", 2.0, 0.0, 1.0, 0.0),
    CommandPhase("heave", 2.0, 0.0, 0.0, 1.0),
]
