
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ledc import plot_spatial


def make_spatial_data():
    """Generate synthetic spatial data for testing."""

    rng = np.random.default_rng(42)

    spot_ids = [f"Spot_{i}" for i in range(100)]

    predictions = pd.DataFrame(
        {
            "B cell": rng.uniform(0, 10, 100),
            "T cell": rng.uniform(0, 15, 100),
            "Macrophage": rng.uniform(0, 5, 100)
        },
        index=spot_ids
    )

    positions = pd.DataFrame(
        {
            "pxl_col_in_fullres": np.tile(
                np.arange(10), 10
            ),
            "pxl_row_in_fullres": np.repeat(
                np.arange(10), 10
            ),
            "in_tissue": np.ones(100, dtype=int)
        },
        index=spot_ids
    )

    return predictions, positions


def test_spatial_plot():
    predictions, positions = make_spatial_data()

    fig = plot_spatial(
        predictions,
        positions,
        show=False
    )

    assert fig is not None
    plt.close(fig)


def test_selected_cell_types():
    predictions, positions = make_spatial_data()

    fig = plot_spatial(
        predictions,
        positions,
        cell_types=["B cell", "T cell"],
        vmax=15,
        show=False
    )

    assert fig is not None
    plt.close(fig)


def test_save_spatial_plot(tmp_path):
    predictions, positions = make_spatial_data()

    output = tmp_path / "spatial_test.png"

    fig = plot_spatial(
        predictions,
        positions,
        save_path=output,
        show=False
    )

    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(fig)


def test_spot_alignment():
    predictions, positions = make_spatial_data()

    # Deliberately reorder positions
    shuffled = positions.sample(frac=1, random_state=42)

    fig = plot_spatial(
        predictions,
        shuffled,
        show=False
    )

    assert fig is not None
    plt.close(fig)


def test_missing_cell_type():
    predictions, positions = make_spatial_data()

    with pytest.raises(ValueError):
        plot_spatial(
            predictions,
            positions,
            cell_types=["Unknown cell"],
            show=False
        )
