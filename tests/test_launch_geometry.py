import pytest

from scripts.launch_geometry import launch_geometry


def test_launch_geometry_reports_tail_and_wave_efficiency() -> None:
    report = launch_geometry(
        elements=1000,
        block_size=256,
        sm_count=2,
        resident_blocks_per_sm=2,
    )

    assert report["blocks"] == 4
    assert report["launched_threads"] == 1024
    assert report["idle_threads"] == 24
    assert report["thread_utilization"] == pytest.approx(1000 / 1024)
    assert report["warps_per_block"] == 8
    assert report["idle_warp_lanes"] == 24
    assert report["warp_lane_utilization"] == pytest.approx(1000 / 1024)
    assert report["waves"] == 1
    assert report["wave_utilization"] == pytest.approx(1.0)
    assert report["final_wave_utilization"] == pytest.approx(1.0)


def test_launch_geometry_tracks_partially_filled_final_wave() -> None:
    report = launch_geometry(4096, 256, sm_count=6, resident_blocks_per_sm=2)

    assert report["blocks"] == 16
    assert report["waves"] == 2
    assert report["wave_utilization"] == pytest.approx(16 / 24)
    assert report["final_wave_utilization"] == pytest.approx(4 / 12)


def test_launch_geometry_accounts_for_partial_warps() -> None:
    report = launch_geometry(100, 48, sm_count=1)

    assert report["blocks"] == 3
    assert report["warps_per_block"] == 2
    assert report["launched_warp_lanes"] == 192
    assert report["idle_warp_lanes"] == 92
    assert report["warp_lane_utilization"] == pytest.approx(100 / 192)
