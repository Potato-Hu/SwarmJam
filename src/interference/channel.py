"""RF channel helpers for interference-aware swarm simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CARRIER_FREQUENCY_HZ = 2.4e9
TRANSMIT_POWER_WATTS = 1.0
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
MIN_DISTANCE_M = 1.0
MILLIWATTS_PER_WATT = 1_000.0


def watts_to_dbm(power_watts: np.ndarray | float) -> np.ndarray | float:
    """Convert linear power in watts to dBm."""

    power_array = np.asarray(power_watts, dtype=float)
    clipped = np.maximum(power_array, np.finfo(float).tiny)
    power_dbm = 10.0 * np.log10(clipped * MILLIWATTS_PER_WATT)
    if np.isscalar(power_watts):
        return float(power_dbm)
    return power_dbm


def fspl_received_power_watts(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    transmit_power_watts: float = TRANSMIT_POWER_WATTS,
    frequency_hz: float = CARRIER_FREQUENCY_HZ,
    min_distance_m: float = MIN_DISTANCE_M,
) -> np.ndarray:
    """Compute received power matrix under free-space path loss."""

    tx_array = np.asarray(tx_positions, dtype=float)
    rx_array = np.asarray(rx_positions, dtype=float)
    if tx_array.ndim != 2 or tx_array.shape[1] != 3:
        raise ValueError(f"tx_positions must have shape (N, 3), got {tx_array.shape}.")
    if rx_array.ndim != 2 or rx_array.shape[1] != 3:
        raise ValueError(f"rx_positions must have shape (M, 3), got {rx_array.shape}.")

    if tx_array.shape[0] == 0 or rx_array.shape[0] == 0:
        return np.zeros((tx_array.shape[0], rx_array.shape[0]), dtype=float)

    wavelength_m = SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    deltas = tx_array[:, None, :] - rx_array[None, :, :]
    distances_m = np.linalg.norm(deltas, axis=-1)
    distances_m = np.maximum(distances_m, float(min_distance_m))
    return float(transmit_power_watts) * (wavelength_m / (4.0 * math.pi * distances_m)) ** 2


@dataclass(frozen=True)
class InterferenceSnapshot:
    """Ground-truth aggregate interference powers for the current world state."""

    key_enemy_received_watts: np.ndarray
    key_enemy_received_dbm: np.ndarray
    friendly_received_watts: np.ndarray
    friendly_received_dbm: np.ndarray
    key_enemy_pairwise_watts: np.ndarray


def compute_interference_snapshot(
    friendly_positions: np.ndarray,
    key_enemy_positions: np.ndarray,
    transmit_power_watts: float = TRANSMIT_POWER_WATTS,
    frequency_hz: float = CARRIER_FREQUENCY_HZ,
    min_distance_m: float = MIN_DISTANCE_M,
) -> InterferenceSnapshot:
    """Aggregate interference seen by key enemies and friendly UAVs."""

    friendly_array = np.asarray(friendly_positions, dtype=float)
    key_enemy_array = np.asarray(key_enemy_positions, dtype=float)
    if friendly_array.shape == (0,):
        friendly_array = np.zeros((0, 3), dtype=float)
    if key_enemy_array.shape == (0,):
        key_enemy_array = np.zeros((0, 3), dtype=float)

    friendly_pairwise = fspl_received_power_watts(
        tx_positions=friendly_array,
        rx_positions=friendly_array,
        transmit_power_watts=transmit_power_watts,
        frequency_hz=frequency_hz,
        min_distance_m=min_distance_m,
    )
    if friendly_pairwise.size:
        np.fill_diagonal(friendly_pairwise, 0.0)
    friendly_received_watts = (
        friendly_pairwise.sum(axis=0) if friendly_pairwise.size else np.zeros(friendly_array.shape[0], dtype=float)
    )

    key_enemy_pairwise = fspl_received_power_watts(
        tx_positions=friendly_array,
        rx_positions=key_enemy_array,
        transmit_power_watts=transmit_power_watts,
        frequency_hz=frequency_hz,
        min_distance_m=min_distance_m,
    )
    key_enemy_received_watts = key_enemy_pairwise.sum(axis=0) if key_enemy_pairwise.size else np.zeros(key_enemy_array.shape[0], dtype=float)

    return InterferenceSnapshot(
        key_enemy_received_watts=key_enemy_received_watts,
        key_enemy_received_dbm=np.asarray(watts_to_dbm(key_enemy_received_watts), dtype=float),
        friendly_received_watts=friendly_received_watts,
        friendly_received_dbm=np.asarray(watts_to_dbm(friendly_received_watts), dtype=float),
        key_enemy_pairwise_watts=key_enemy_pairwise,
    )
