
import json

import numpy as np
import pandas as pd
import pytest

from ledc import LEDC


def test_expression_scale_default():
    model = LEDC()
    assert model.expression_scale == "unknown"


def test_expression_scale_valid():
    model = LEDC(expression_scale="raw_counts")
    assert model.expression_scale == "raw_counts"


def test_expression_scale_invalid():
    with pytest.raises(ValueError):
        LEDC(expression_scale="invalid")


def test_expression_scale_save_load(tmp_path):
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.poisson(5, size=(30, 4)),
        columns=["g1", "g2", "g3", "g4"]
    )

    Y = pd.DataFrame(
        rng.poisson(3, size=(30, 2)),
        columns=["B cell", "T cell"]
    )

    model = LEDC(
        target="count",
        expression_scale="raw_counts",
        n_estimators=10
    )
    model.fit(X, Y)

    save_dir = tmp_path / "models"
    model.save(save_dir)

    # Check the saved metadata
    metadata = json.loads(
        (save_dir / "metadata.json").read_text()
    )

    assert metadata["expression_scale"] == "raw_counts"

    # Check that loading restores the metadata
    loaded = LEDC.load(save_dir)

    assert loaded.expression_scale == "raw_counts"

    # Check that model predictions are unchanged
    np.testing.assert_allclose(
        model.predict(X).to_numpy(),
        loaded.predict(X).to_numpy()
    )

    # Simulate an older saved model without the new field
    metadata.pop("expression_scale")
    (save_dir / "metadata.json").write_text(
        json.dumps(metadata)
    )

    old_model = LEDC.load(save_dir)

    assert old_model.expression_scale == "unknown"

    np.testing.assert_allclose(
        model.predict(X).to_numpy(),
        old_model.predict(X).to_numpy()
    )
