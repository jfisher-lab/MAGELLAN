# run a simulation of a model vs spec
import argparse
import os
from pathlib import Path

import toml

from magellan.simulation import run_simulation

# Load configuration from TOML file
local_path = Path(__file__).parent.parent
config_path = os.path.join(local_path, "scripts", "example_configs", "example_shortest_path_config.toml")

# Use argparse to get config path from command line
parser = argparse.ArgumentParser(description="Run BMA simulation.")
parser.add_argument(
    "--config",
    type=str,
    default=config_path,
    help="Path to the TOML configuration file.",
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = toml.load(f)


#  Files -----------------------------------------------

pipe_dir = config["DEFAULT"]["pipe_dir"]
data_dir = config["DEFAULT"]["data_dir"]
data_sim_dir = config["DEFAULT"]["simulation_data_dir"]
shortest_results_path = os.path.join(
    local_path, pipe_dir, config["DEFAULT"]["results_dir"]
)
simulation_results_path = os.path.join(
    local_path, pipe_dir, config["DEFAULT"]["results_dir"]
)
path_data_sim = os.path.join(local_path, pipe_dir, data_dir, data_sim_dir)
input_file = os.path.join(
    shortest_results_path, config["simulation_input"]["input_file"]
)
spec_file = os.path.join(path_data_sim, config["simulation_input"]["spec_file"])
output_file = os.path.join(
    simulation_results_path, config["simulation_output"]["output_file"]
)

# Settings --------------------------------------------

time_step = config["simulation_settings"]["time_step"]
max_time_step = config["simulation_settings"]["max_time_step"]

# bma_console = r"C:\Program Files (x86)\BMA\BioCheckConsole.exe"
bma_console = config["simulation_settings"]["bma_console"]

# Check BMA console path exists
if not os.path.exists(bma_console):
    raise FileNotFoundError(
        f"BMA console not found at path: {bma_console}\n\n"
        "Please check the path in your config file and ensure BMA is installed correctly. \n"
        "Note, BMA can most easily be installed on Windows from https://biomodelanalyzer.org/. \n"
        "For other options, see https://github.com/hallba/BioModelAnalyzer"
    )


# Main ------------------------------------------------

run_simulation(
    spec_file=spec_file,
    input_file=input_file,
    output_file=output_file,
    bma_console=bma_console,
    time_step=time_step,
    max_time_step=max_time_step,
)
