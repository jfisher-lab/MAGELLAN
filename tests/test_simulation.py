# File for tests of the simulation module
import os
import platform
import tempfile
import warnings
from os.path import join as pj
from pathlib import Path

import pandas as pd
import pytest

from magellan.simulation import check_whitespace, run_simulation

path_root = Path(__file__).parent.parent


# Test check_whitespace --------------
@pytest.fixture(params=["  hello", "hello  ", "he llo", "hello", ""])
def ws_cases(request):
    return request.param


def test_check_whitespace(ws_cases):
    if " " in ws_cases:
        with pytest.raises(NotImplementedError):
            check_whitespace(ws_cases)
    else:
        result = check_whitespace(ws_cases)
        assert result is None


# Test run_simulation --------------
@pytest.fixture
def temp_folder():
    return tempfile.mkdtemp()


def test_run_simulation(temp_folder):
    if platform.system() != "Windows":
        warnings.warn("Skipping test_run_simulation on non-Windows system")
        pytest.skip("Skipping test_run_simulation on non-Windows system")

    spec_file = pj(
        path_root, "tests", "helper_simulation", "helper_simulation_spec_in_1"
    )
    input_file = pj(
        path_root, "tests", "helper_simulation", "helper_simulation_netw_in_1"
    )
    output_file = pj(temp_folder, "output_file")
    bma_console = "C:\PROGRA~2\BMA\BioCheckConsole.exe" # type: ignore
    expected_output = pj(
        path_root, "tests", "helper_simulation", "helper_simulation_out_1"
    )

    if not os.path.exists(bma_console):
        raise FileNotFoundError(
            "User must install BMA command line from biomodelanalyzer.org"
        )

    run_simulation(spec_file, input_file, output_file, bma_console)

    # Check if the output file exists
    assert os.path.exists(output_file + ".csv")

    # Load the expected output CSV file
    expected_df = pd.read_csv(expected_output + ".csv")

    # Load the actual output CSV file
    actual_df = pd.read_csv(output_file + ".csv")

    # Compare the content of the two dataframes
    pd.testing.assert_frame_equal(expected_df, actual_df)
