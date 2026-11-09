from __future__ import annotations

from math import ceil


def launch_geometry(
    elements: int,
    block_size: int,
    sm_count: int,
    resident_blocks_per_sm: int = 1,
    warp_size: int = 32,
) -> dict[str, float | int]:
    if min(elements, block_size, sm_count, resident_blocks_per_sm, warp_size) < 1:
        raise ValueError("launch dimensions must be positive")

    blocks = ceil(elements / block_size)
    launched_threads = blocks * block_size
    blocks_per_wave = sm_count * resident_blocks_per_sm
    waves = ceil(blocks / blocks_per_wave)
    final_wave_blocks = blocks - (waves - 1) * blocks_per_wave
    warps_per_block = ceil(block_size / warp_size)
    launched_warp_lanes = blocks * warps_per_block * warp_size
    return {
        "elements": elements,
        "block_size": block_size,
        "blocks": blocks,
        "launched_threads": launched_threads,
        "idle_threads": launched_threads - elements,
        "thread_utilization": elements / launched_threads,
        "warps_per_block": warps_per_block,
        "launched_warp_lanes": launched_warp_lanes,
        "idle_warp_lanes": launched_warp_lanes - elements,
        "warp_lane_utilization": elements / launched_warp_lanes,
        "blocks_per_wave": blocks_per_wave,
        "waves": waves,
        "wave_utilization": blocks / (waves * blocks_per_wave),
        "final_wave_utilization": final_wave_blocks / blocks_per_wave,
    }
