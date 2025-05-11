import subprocess
import warnings
from os.path import exists
from pathlib import Path
from time import sleep

import alive_progress as ap
import pandas as pd

from magellan.utils.file_io import read_json


def check_whitespace(input_string):
    input_string = str(input_string)  # Convert PosixPath to string

    if " " in input_string:
        raise NotImplementedError(
            "Paths to input and output files cannot contain whitespace characters."
        )
    return None


def get_cmd(input_file, output_file, node_ko, bma_console, time_step=30):
    """
    Generate command for BMA console

    :param input_file: str, directory of input .json file
    :param output_file: str, directory of output .json file
    :param node_ko: str, format 'node id [space] ko value'
    :param time_step: int, # iterations in bma simulation
    :param bma_console: to BioCheckConsole.exe (see README for installation instructions)

    :return: str, generated command

    """

    if input_file.split(".")[-1] != "json":
        input_file += ".json"

    check_whitespace(input_file)
    check_whitespace(output_file)
    cmd = "%s -model %s -engine SIMULATE -simulate_time %d -simulate %s.csv -ko %s" % (
        bma_console,
        input_file,
        time_step,
        output_file,
        node_ko,
    )

    if "Program Files (x86)" in cmd:
        cmd = cmd.replace("Program Files (x86)", "PROGRA~2")
        warnings.warn(
            "Sending commands from python to console with spaces leads to errors. We have automatically\n "
            "replaced 'Program Files (x86)' with 'PROGRA~2' in the command. If this does not work,\n "
            "instead please ensure that the path to the BioCheckConsole.exe does not contain any spaces."
        )

    return cmd


def run_cmd(cmd):
    """
    Run BMA command in command line
    :param cmd: str, command to run

    """
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)


def convert_min_max(state_val, low=0, high=2):
    """
    Convert the value in a spec to numeric format if the value is min/max

    :param low: setting for node for min
    :param high: setting for node for max
    :param state_val: str or int
    :return state_val_convert: int

    """

    dic = {"min": low, "max": high, "mid": int((low + high) / 2.0)}
    state_val_convert = dic[state_val] if isinstance(state_val, str) else int(state_val)

    return state_val_convert


# def check_int_str(x):
#     """
#     Check if a variable is str or int
#
#     :param x: input variable
#     :return: True if x is str or int, False otherwise
#
#     """
#
#     return True if isinstance(x, str) or isinstance(x, int) else False


def check_nan(x):
    """
    Check if a variable is numeric or NaN

    :param x: input variable
    :return: True if x is numeric or NaN, False otherwise

    """

    return True if pd.isna(x) else False


def extract_spec_val(spec_val):
    """
    Extract and construct a sub-dict for spec_dic value

    :param spec_val: list, a value item from spec_val
    :return spec_sub_dic: dict

    """

    spec_sub_dic = {"perturbation": {}, "expectation": {}}

    for ele in spec_val:
        if not check_nan(ele[-1]):  # expected node
            spec_sub_dic["expectation"][ele[0]] = convert_min_max(ele[-1])
        elif not check_nan(ele[1]):  # perturbation
            spec_sub_dic["perturbation"][ele[0]] = convert_min_max(ele[1])
        else:
            raise ValueError("Invalid value in spec file: %s" % ele)

    return spec_sub_dic


def extract_node(input_file):
    """
    Extract node id from input json file

    :param input_file: .json, network model
    :return: dict, key: node name, val: node id

    """

    net_dic = read_json(input_file)["Model"]["Variables"]
    node_dic = {ele["Name"]: ele["Id"] for ele in net_dic}

    return node_dic


def parse_spec(spec_file):
    """
    Read spec file and construct spec dictionary (without node id)

    :param spec_file: .csv spec file
    :return spec_dic: dict,
        structure: {experiment:
        {perturbation: {node name: {id: perturbed node id, perturbation: perturbed value (converted to numeric)}},
        expectation: {node names with expectation: expected value}}}

        e.g. single item
        'trivial apoptosis':
        {perturbation: {BCL2: 0, BCL2L1: 0, BCL2L2: 0, MCL1: 0},
        expectation: {BAK1: 2, BAX: 2}

    """
    spec_path = Path(spec_file)
    if spec_path.suffix != ".csv":
        spec_path = spec_path.with_suffix(".csv")

    df_spec = pd.read_csv(spec_file)
    spec_dic = (
        df_spec.groupby("experiment_particular")[
            ["gene", "perturbation", "expectation_bma"]
        ]
        .apply(lambda x: x.values.tolist())
        .to_dict()
    )

    spec_dic = {k: extract_spec_val(v) for k, v in spec_dic.items()}

    return spec_dic


def spec_in_id(spec_file, input_file):
    """
    Construct spec dictionary

    :param spec_file: .csv spec file
    :param input_file: .json network model

    :return spec_dic: dict,
        structure: {experiment:
        {perturbed node: {node name: {id: perturbed node id, perturbation: perturbed value (converted to numeric)}},
        expectation: {node names with expectation: expected value}}}

        e.g. single item
        'trivial apoptosis':
        {perturbation: {BCL2: {id: 1, ko: 0}, BCL2L1: {id: 2, ko: 0},
                        BCL2L2: {id: 3, ko: 0}, MCL1: {id: 4, ko: 0}},
        expectation: {BAK1: 2, BAX: 2}

    """

    node_dic = extract_node(input_file)
    spec_dic = parse_spec(spec_file)

    for k, v in spec_dic.items():
        for kk, vv in v["perturbation"].items():
            if kk in node_dic:
                spec_dic[k]["perturbation"][kk] = {"id": node_dic[kk], "ko": vv}
            else:
                raise KeyError("%s not found in the input network .json file." % kk)

    return spec_dic, node_dic


def read_result(sub_output, node_dic):
    """
    Construct node-indexed data frame from .csv bma results

    :param sub_output: str, directory of .csv file (sub???)
    :param node_dic: OrderedDict, key: node name, value: node id (???)

    :return: pandas.DataFrame, table of bma results, row: node name (???), column: bma simulation iteration

    """

    if sub_output.split(".")[-1] != "csv":
        sub_output += ".csv"

    df = pd.read_csv(sub_output, header=None, sep=",|;", engine="python")
    df = df[range(2, df.shape[1], 2)].T
    df.columns = list(range(df.shape[1]))
    df.index = node_dic.keys()

    return df


def extract_result(df):
    """
    Extract results from constructed df

    :param df: pandas.DataFrame
    :return: list, [pandas.DataFrame, int, pandas.DataFrame, pandas.DataFrame], to be explained???

    """

    # compare last two columns
    if all(df.iloc[:, -1] == df.iloc[:, -2]):  # stabilised
        return df.iloc[:, -1]

    else:  # unstabilised
        df_loop = None
        for idx in range(-2, -df.shape[1] + 1, -1):
            if all(df.iloc[:, -1] == df.iloc[:, idx]):
                df_loop = df.iloc[:, idx + 1 :]
                break

        if df_loop is None:
            raise ValueError(
                "Insufficient time steps. Please set a higher value for -simulation_time."
            )

        df_loop_mean = df_loop.mean(axis=1)
        df_loop_mean_hl = (df_loop.max(axis=1) + df_loop.min(axis=1)) / 2

        return [df_loop, idx + 1, df_loop_mean, df_loop_mean_hl]


def get_result(sub_output, node_dic):
    """
    Obtain results

    :param sub_output: str, directory of bma output (sub???)
    :param node_dic: OrderedDict (???), key: node name, value: node id (???)

    :return: list, see function extract_result(.)

    """

    df = read_result(sub_output, node_dic)
    result = extract_result(df)

    return result


def prep_result(result, loop_strategy="mean"):
    if isinstance(result, list):  # unstabilised
        if loop_strategy == "mean":
            df = result[-2]
        else:
            df = result[-1]
    else:
        df = result

    return df


def compare_result(df_result, exp_dic, show_all=False):
    """
    Compare result (stabilised or mean over unstabilised) vs spec expectation

    :param df: pandas.DataFrame, shape: (n_node, 1). Stabilised result or mean over unstabilised loops
    :param exp_dic: dict, exp value item in spec_dic, corresponds to the expected values under current exp
    :return ?: pandas.DataFrame, shape: (n_expectation, ?): Comparison between BMA results and expectation

    """

    df_compare = {}

    for node, val in exp_dic["expectation"].items():
        result = df_result.loc[node]
        df_compare[node] = {"expectation": val, "mean": result, "diff": result - val}
        # if result != val:  # bma result doesn't meet expectation
        #     df_compare[node] = {'expectation': val, 'mean': result, 'diff': result - val}

    df_compare = pd.DataFrame.from_dict(df_compare).T

    # remove same-valued results
    if not show_all:
        df_compare = df_compare[df_compare["diff"] != 0]

    return df_compare


def run_bma(spec_dic, input_file, output_file, bma_console, time_step=30):
    """
    Run BMA in command line against given network model and spec

    :param spec_dic: dict of spec
    :param input_file: .json file, input file for network model
    :param output_file: .csv file directory and name
    :param time_step: int, n_iterations of simulation in BMA
    :param bma_console: to BioCheckConsole.exe (see README for installation instructions)

    """
    with ap.alive_bar(len(spec_dic.items())) as bar:
        for exp, exp_dic in spec_dic.items():
            sub_output = get_sub_output(output_file, exp)

            # extract perturbation node and ko values
            node_ko = [
                "%d %d" % (v["id"], v["ko"]) for v in exp_dic["perturbation"].values()
            ]
            node_ko = " -ko ".join(node_ko)

            # run bma in command line
            cmd = get_cmd(input_file, sub_output, node_ko, bma_console, time_step)
            run_cmd(cmd)
            bar()


def get_bma(spec_dic, node_dic, output_file, loop_strategy="mean", show_all=False):
    """
    Get BMA results

    :param spec_dic: dict of spec
    :param node_dic: dict of node name and id
    :param output_file: .csv file directory and name
    :param loop_strategy: str, 'mean' or 'high_low'. the mean strategy when a loop is discovered in unstabilised result
    :param show_all: boolean. if false, only node-exp pairs that do not meet expecation will be returned

    :return: .csv file, output file comparing BMA results and expectation

    """

    df_final = pd.DataFrame()
    for exp, exp_dic in spec_dic.items():
        sub_output = get_sub_output(output_file, exp)
        while not exists(str(sub_output) + ".csv"):
            sleep(0.01)

        # obtain result
        result = get_result(sub_output, node_dic)
        # df = prep_result(result, loop_strategy)

        if isinstance(result, list):  # unstabilised
            if loop_strategy == "mean":
                df = result[-2]
            else:
                df = result[-1]
        else:
            df = result

        df_compare = compare_result(df, exp_dic, show_all)
        df_compare["experiment"] = exp

        df_final = pd.concat([df_final, df_compare], axis=0)

    return df_final


def bma_scripts(
    spec_file,
    input_file,
    output_file,
    bma_console,
    time_step=30,
    loop_strategy="mean",
    show_all=False,
):
    """
    Run BMA using BMA console (BMA must be pre-installed)

    :param spec_file: str, directory of spec .csv file
    :param input_file: str, directory of input .json file
    :param output_file: str, directory of output file
    :param time_step: int, number of iterations in BMA simulation
    :param loop_strategy: str, how to calculate bma results if in loop
           'mean': average of all results within a single loop
    :param bma_console: to BioCheckConsole.exe (see README for installation instructions)

    :return: pandas.DataFrame, BMA simulation result

    """

    spec_dic, node_dic = spec_in_id(spec_file, input_file)

    run_bma(spec_dic, input_file, output_file, bma_console, time_step)
    df_final = get_bma(spec_dic, node_dic, output_file, loop_strategy, show_all)

    return df_final


def run_simulation(
    spec_file, input_file, output_file, bma_console, time_step=50, max_time_step=1000
):
    """
    Run a simulation of a model vs spec
    :param spec_file: specification file
    :param input_file: input JSON file for network
    :param output_file: name of output files
    :param bma_console: path to BioCheckConsole.exe without whitespace if possible
    :param time_step: time step for simulation
    :param max_time_step: cap on time step to increase to automatically
    :return: writes csv files to output_file
    """

    # node_dic = extract_node(input_file)
    while True:
        try:
            df_final = bma_scripts(
                spec_file,
                input_file,
                output_file,
                time_step=time_step,
                show_all=True,
                bma_console=bma_console,
            )
            break
        except ValueError:
            if time_step < max_time_step:
                time_step += 50
                print(f"increase time step by 50, current time_step: {time_step}")
            else:
                raise ValueError(
                    "Time step too large, and possibly other errors, please scroll up."
                )
    # df_final = df_final.rename_axis('node').reset_index()
    # # this is so that what is written out as a csv will be
    # # read in the same, otherwise the row names get turned into an unnamed column, so `df_final != pd.read_csv(
    # # df_final.csv)`
    df_final.to_csv(output_file + ".csv")
    return df_final


def get_sub_output(output_file, exp):
    """
    Generate sub output file name for each experiment
    :param output_file: out_put filename
    :param exp: experiment
    :return: sub output file name
    """
    output_file = Path(output_file)  # Ensure it's a Path object
    return (
        output_file.parent
        / f"{output_file.stem}_{exp.replace(' ', '_')}_RAW_SIM{output_file.suffix}"
    )
