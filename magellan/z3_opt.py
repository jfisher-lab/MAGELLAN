import re
import time
from itertools import chain

import networkx as nx
import numpy as np
import z3


def sign_to_num(sign):
    return 1 if sign == "Activator" else -1


def sign_to_op(sign_):
    return "+" if sign_ == "Activator" else "-"


def aggregate_z3(G, v, local_dic, max_range=2):
    t = len(G.nodes[v]["agg"])
    u_list = sorted(list(G.predecessors(v)))

    n_parent = len(u_list)

    if n_parent == 0:
        msg = G.nodes[v]["agg"][t - 1]

    else:
        msg = 0.0
        all_inh = True

        for u in u_list:
            sign_ = sign_to_op(G[u][v]["sign"])

            if all_inh:
                all_inh &= sign_ == "-"

            # msg += '%sw_%s_%s * (%s)' % (sign_, u, v, G.nodes[u]['agg'][t-1])
            local_dic = {**local_dic, **locals().copy()}
            exec(
                "msg_temp = z3.simplify(%sw_%s_%s * G.nodes[u]['agg'][t-1])"
                % (sign_, u, v),
                globals(),
                local_dic,
            )
            msg += local_dic["msg_temp"]

        msg = round_minmax(msg)

        if all_inh:
            msg = max_range - msg

    try:
        msg = z3.simplify(msg)
    except Exception:
        pass

    return msg


def aggregate(G, v, max_range=2):
    """
    Aggregate messages from parent nodes of node v

    :param G: networkx.DiGraph
    :param v: str, name of the target node
    :param max_range: int,

    :return msg: str, aggregated messages in text

    """

    # obtain parent nodes of node v
    t = len(G.nodes[v]["agg"])
    u_list = list(G.predecessors(v))

    n_parent = len(u_list)

    # if v has no parents, return the message from t - 1 timestep (current timestep: t)
    # as topology remains unchanged across timesteps, v will retain its initialised message
    if n_parent == 0:
        msg = G.nodes[v]["agg"][t - 1]

    # aggregate messages from v's parent nodes
    else:
        msg = ""
        all_inh = True  # whether all parents of v are inhibitors
        for u in u_list:
            sign_ = sign_to_op(G[u][v]["sign"])

            if all_inh:
                all_inh &= sign_ == "-"

            if G.nodes[u]["agg"][t - 1] == "":
                msg += ""
            else:
                msg += "%sw_%s_%s * (%s)" % (sign_, u, v, G.nodes[u]["agg"][t - 1])

        msg_len = len(msg) > 0
        if msg_len:
            msg = "round_minmax(%s)" % msg

        if all_inh:  # make msg = max_range - msg if all parents of v are inhibitors
            if msg_len:
                # use plus here instead of minus bc signs are absorbed in msg
                msg = msg.replace("round_minmax(", "round_minmax(%d" % max_range, 1)
            else:
                msg = "%d." % max_range

        if msg_len and msg[0] == "+":
            msg = msg[1:]  # remove leading plus '+'

        msg = msg.replace("- -", "+")  # replace double minus to a single plus

        # if n_parent > 1:
        #     msg = '(%s) / %.1f' % (msg, len(u_list))

    return msg


def init_weight(weights, dtype):
    local_dic = locals().copy()
    for w in weights:
        exec('%s = z3.%s("%s")' % (w, dtype, w), globals(), local_dic)

    return local_dic


def remove_dash(s):
    return s.replace("-", "").replace("_", "")


def mpn(G, pert_node=None, max_iter=10**3, max_range=2):
    """
    Message passing network update. The update stops when

    :param G: networkx.DiGraph
    :param pert_node: dic, key: perturbation node, value: perturbed value
    :param max_iter: int, maximum iteration when there are loops (thus messages will not "converge")
    :param max_range: int, upper bound of nodes

    :return: a set of equations of expectation being expressed in other node values and weights
        each equation corresponds to a single expectation node in one experiment

    """

    G_copy = G.copy()
    nx.set_node_attributes(
        G_copy, values={n: [] for n in G_copy.nodes}, name="agg"
    )  # use dict to avoid list change together

    # message passing
    # initialise messages with an empty string
    for n in G_copy.nodes:
        G_copy.nodes[n]["agg"].append("")

    if pert_node is not None:
        to_remove = []
        for v, val in pert_node.items():
            G_copy.nodes[v]["agg"][0] = str(val)

            # remove parent node for perturbation node
            to_remove += [(u, v) for u in G_copy.predecessors(v)]

        G_copy.remove_edges_from(to_remove)

    t = 1
    while True:
        print("iteration: %d" % t)
        cond = True
        for n in G_copy.nodes:
            # G_copy.nodes[n]['agg'].append(aggregate_z3(G_copy, n, local_dic, max_range))
            G_copy.nodes[n]["agg"].append(aggregate(G_copy, n, max_range))
            cond &= np.all(
                str(G_copy.nodes[n]["agg"][-1]) == str(G_copy.nodes[n]["agg"][-2])
            )

        if cond or t == max_iter:
            break

        t += 1

    return G_copy


def agg_to_eq(G, X, y):
    """
    Convert aggregated results to equations with weights

    :param G: nx.DiGraph
    :param n: str, node name
    :param X: pandas.DataFrame, row: node name in ascending order, col: experiment
        non-perturbation nodes have values at -1 to distinguish those with 0
    :param y: pandas.DataFrame, row: node name in ascending order, col: experiment
        non-expectation nodes have values at -1 to distinguish those with 0

    # :param pert_node: list, nodes to be perturbed
    # :param exp_node: list, nodes with expectations
    # :param exp: str, experiment of interest

    :return: str, converted aggregated results, will be used in z3 solver

    """

    exp_list = []

    for exp_col in X.columns:
        pert_node = X.index[X[exp_col] != -1].tolist()
        exp_node = y.index[y[exp_col] != -1].tolist()

        for n_exp in exp_node:
            exp = str(G.nodes[n_exp]["agg"][-1]).replace("'", "")
            exp = re.sub(
                r"\(([0-9A-Za-z]+), ([0-9A-Za-z]+)\)",
                lambda x: "w_%s_%s" % (x.group(1), x.group(2)),
                exp,
            )  # replace tuple keys (u, v) to w_uv
            exp = (
                exp.replace(": ", " *")
                .replace(", ", " + ")
                .replace("{", "(")
                .replace("}", ")")
            )

            for n_pert in pert_node:
                exp = exp.replace("%s_0" % n_pert, str(X.at[n_pert, exp_col]))

            # replace non perturbed nodes with zeros
            exp = re.sub(r"\*[0-9A-Za-z]+_0", "*0", exp)

            exp_list.append(exp + " == %f" % y.at[n_exp, exp_col])

    return exp_list


def extract_weight_single(exp):
    return re.findall("w_[0-9A-Za-z]+_[0-9A-Za-z]+", exp)


def extract_weight(exp_list):
    weights = [extract_weight_single(exp_list[i]) for i in range(len(exp_list))]
    return set(chain(*weights))


def gen_cond(weights, operator, val):
    return ", ".join(["%s %s %.1f" % (w, operator, val) for w in weights])


def find_init(exp):
    return set(re.findall(r"[0-9A-Za-z]+_0", exp))


def set_init(exp, val_dic):
    init_node = find_init(exp)
    init_dic = {k: 0 for k in init_node}

    for k, v in val_dic.items():
        init_dic[k] = v

    return init_dic


def replace_init(exp, val_dic):
    init_dic = set_init(exp, val_dic)

    for k, v in init_dic.items():
        exp = exp.replace(k, "%.1f" % v)

    return exp


# min between val and 2
def min_2(val):
    val = z3.If(val < 2, val, 2)

    return val


# max between val and 0
def max_0(val):
    val = z3.If(val > 0, val, 0)

    return val


def min_max(val):
    return min_2(max_0(val))


def round_z3(val):
    val = z3.If(val - z3.ToInt(val) >= 0.5, z3.ToInt(val) + 1, z3.ToInt(val))

    return val


def round_minmax(val):
    """
    Round up number to integer, and restrict number between 0 to 2

    :param val: float
    :return: float, rounded up number between 0 to 2

    """

    return round_z3(min_max(val))


def z3_check(exp_dic, G, operate=">=", cond=0):
    """
    Check if a set of equations and additional constraints are consistent
    e.g whether equations as in exp_dic.values, with unknowns are a subset of w indexed by G.edges follow
    constraints defined by w [operate] [cond], w >= 0 in the default case

    :param exp_dic: dict, key: str, perturbation & exp, value: str, equation
        (expression of the expectation nodes == expected values). equations of constraint
    :param G: networkx.DiGraph. graph of network, use to initialise Z3 variables
    :param operate: str, operator for w constraints, default '>=' (w >= 0)
    :param cond: int, w constraint value, default 0 (w >= 0)

    :return: str. sat: consistent, unsat: inconsistent (unsolvable)

    """

    # initialise
    w = ["w_%s_%s" % (u, v) for (u, v) in G.edges]
    exec('%s = z3.Reals("%s")' % (",".join(w), " ".join(w)))

    # generate conditions
    cond = gen_cond(["w_%s_%s" % (u, v) for (u, v) in G.edges], operate, cond)
    exp_all = ", ".join(exp_dic.values())

    s = z3.Solver()
    exec("s.add(%s, %s)" % (cond, exp_all))

    return str(s.check())


def z3_to_float(num, prec=3):
    """
    Convert z3.z3.ArithRef to float

    :param num: z3.z3.ArithRef
    :param prec: int, precision

    :return: float

    """

    return float(num.as_decimal(prec=prec).replace("?", ""))


def extract_z3_result(m):
    """
    Extract results from z3.Solver().model()

    :param m: z3.Solver().model(), model results
    :return: dict of z3 results. key:

    """

    return {str(ele): z3_to_float(m[ele]) for ele in m}


def estimate_w(exp_dic, G):
    """
    Estimate weights with z3 solver

    :param exp_dic:
    :param G: nx.DiGraph, resulted network graph. Only edge info is used in order to generate z3 variables

    :return: if a solution exists, dict: key: str, weight 'w_u_v', value: float, estimated weights.
             if a solution does not exist, return s.check() --> z3.unsat

    """

    for u, v in G.edges:
        w = "w_%s_%s" % (u, v)
        exec('%s = z3.Real("%s")' % (w, w))
        time.sleep(0.01)

    cond = gen_cond(["w_%s_%s" % (u, v) for (u, v) in G.edges], ">=", 0)

    if "==" in list(exp_dic.values())[0]:
        exp_all = ", ".join(exp_dic.values())
    else:
        exp_all = ", ".join(["%s == %.1f" % (exp, val) for exp, val in exp_dic.items()])

    # exec('z3.solve(%s, %s)' % (cond, exp_all))

    s = z3.Solver()
    exec("s.add(%s, %s)" % (cond, exp_all))

    if s.check() == z3.sat:
        m = s.model()
        dic = extract_z3_result(m)

        return dic

    else:
        print("no solution")
        return s.check()


def min_max_py(val):
    return np.max([np.min([val, 2]), 0])


def round_minmax_py(val):
    return np.floor(min_max_py(val) + 0.5)


def weight_to_dic(weight_str):
    dic = {}
    weight_str = (
        weight_str.replace(" = ", "': ")
        .replace(",", ", '")
        .replace("]", "}")
        .replace("[", "{'")
    )
    exec("dic = %s" % weight_str, dic)

    return dic["dic"]


def replace_dic(exp_str, weight_dic):
    for k, v in weight_dic.items():
        exp_str = exp_str.replace(k, str(v))

    exp_str = exp_str.replace("round_minmax", "round_minmax_py")

    return exp_str
