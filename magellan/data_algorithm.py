from collections import Counter, OrderedDict
from itertools import chain

import matplotlib.pyplot as plt
import pandas as pd


def get_count(df, col="sources", sep=";", reverse=False):
    df = df[col].str.split(sep).to_list()
    count_dic = Counter(list(chain(*df)))

    count_dic = OrderedDict(
        sorted(count_dic.items(), key=lambda t: t[1], reverse=reverse)
    )

    return count_dic


def plot_count(
    count_dic=None,
    df=None,
    col="sources",
    sep=";",
    reverse=False,
    figsize=None,
    to_save=None,
    **kwargs,
):
    if not count_dic:
        count_dic = get_count(df, col, sep, reverse)

    if figsize:
        plt.figure(figsize=figsize)

    plt.plot(count_dic.values(), count_dic.keys(), **kwargs)
    plt.tight_layout()

    if to_save:
        plt.savefig(to_save, dpi=150)
    else:
        plt.show()


def _extract_spec(df, spec_type):
    dic = (
        df.dropna(subset=[spec_type])
        .groupby("experiment_particular")[["gene", spec_type]]
        .apply(lambda x: x.values.tolist())
        .to_dict()
    )
    dic = {k: {kk: vv for kk, vv in v} for k, v in dic.items()}

    return dic


def extract_spec(df, dic, min_range=0, max_range=4):
    """
    Extract info from a .csv spec file

    :param df: pandas.DataFrame or str, spec df or file directory of the spec file
    :param dic: dict, constant nodes

    :return spec_dic: dict,
        {experiment: {'pert': {perturbation node: perturbed value},
                      'exp': {expectation node: expected value}}

    """

    if isinstance(df, str):
        df = pd.read_csv(df)

    # replace min/max
    df[["perturbation", "expected_result_bma"]] = (
        df[["perturbation", "expected_result_bma"]]
        .replace({"min": min_range, "max": max_range})
        .astype(float)
    )

    # if isinstance(dic, str):
    #     dic = read_pickle(dic)

    # create spec dic
    pert_dic = _extract_spec(df, "perturbation")
    exp_dic = _extract_spec(df, "expected_result_bma")

    spec_dic = {k: {"pert": pert_dic[k], "exp": exp_dic[k]} for k in pert_dic}

    for v in spec_dic.values():
        # v['pert'].update(dic)
        for k in set(dic.keys()) - set(
            v["pert"].keys()
        ):  # only update constant nodes that are not perturbed
            v["pert"][k] = dic[k]

    return spec_dic


def extract_spec_bma(
    df: pd.DataFrame | None = None,
    path: str | None = None,
    file_name: str | None = None,
):
    if not df:
        if file_name and file_name[-4:] != ".csv":
            file_name += ".csv"

        df = pd.read_csv(f"{path}{file_name}")

    spec_dic = (
        df.groupby("experiment_particular")[["gene", "mean_result"]]
        .apply(  # type: ignore
            lambda x: x.values.tolist()
        )
        .to_dict()
    )
    spec_dic = {k: {ele[0]: ele[1] for ele in v} for k, v in spec_dic.items()}

    df = pd.DataFrame.from_dict(spec_dic)
    df = df.sort_index(axis=0, ascending=True)

    return df


def extract_val(df, grouby, col):
    """
    Construct perturbation/expectation df from full spec df

    :param df: pandas.DataFrame, spec df
    :param grouby: str, column name in df to groupby, usually 'experiment_particular'
    :param col: list, column names in df to form rows and values in the new df.
                usually ['gene', 'perturbation] for perturbation node
                and ['gene', 'expected_result_bma'] for expectation node

    :return X: pandas.DataFrame, constructed perturbation df
    :return y: pandas.DataFrame, constructed expectation df

    """

    df = (
        df.groupby(grouby)[col]
        .apply(lambda x: x.dropna().set_index(col[0])[col[1]].to_dict())
        .to_dict()
    )

    return pd.DataFrame.from_dict(df).fillna(
        -1
    )  # fill with -1 not 0 to distinguish from 0 in BMA


def gen_data(df=None, path=None, file_name=None, min=0, max=2):
    """
    Generate input X and labels y based on spec df
    X: pertubation node value
    y: expectation node value

    :param df: pandas.DataFrame. spec df.
               Note that df is the spec file used for BMA, NOT the processed spec (as returned by extract_spec_bma)
    :param path: str, directory of spec df. needed when df is not passed
    :param file_name: str, .csv spec file. needed when df is not passed
    :param min: int, the lower bound of BMA net
    :param max: int, the upper bound of BMA net

    :return X: pandas.DataFrame. row: node names in ascending order, col: experiments.
               Only perturbation nodes in each experiment are non-zero
    :return y: pandas.DataFrame. row: node names in ascending order, col: experiments.
               Only expectation nodes in each experiment are non-zero

    """

    if not df:
        if file_name and file_name[-4:] != ".csv":
            file_name += ".csv"

        df = pd.read_csv(f"{path}{file_name}")

    col = ["gene", "perturbation", "expected_result_bma"]

    df = df.dropna(subset=col, how="all")
    df = df.replace("min", min)
    df = df.replace("max", max)
    df = df.replace("mid", (min + max) / 2.0)

    X = extract_val(df, "experiment_particular", ["gene", "perturbation"])
    y = extract_val(df, "experiment_particular", ["gene", "expected_result_bma"])

    ## NEED TO FILL IN OTHER NODES WITH ZEROS
    ## NEED TO CHANGE -1 TO 0 IN X (FOR COMPUTATIONAL REASONS) -- necessary to indicate perturbed nodes?

    return X, y
