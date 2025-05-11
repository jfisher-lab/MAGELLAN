import csv
from typing import Iterable

import omnipath as op
import pandas as pd

from magellan.graph import Graph

# from BMATool.graph_algorithm import get_direction
from magellan.path_algorithm import (
    expand_path,
    find_path,
    get_direction,
    get_style,
    remove_dup,
)
from magellan.utils.file_io import save_excel


class ShortestPath:
    """
    Search for weighted shortest paths

    :param cut_off: False or tuple. If tuple, e.g. ('curation_effort', 15),
        then only include interactions with >= 15 curation_effort
    :param pre_filter: tuple, e.g. ('n_primary_sources', 3) means remove all interactions with n_primary_sources <= 3
    :param filter_source_prime: None or database name. If database name, e.g. 'SIGNOR',
        then only use interactions in the source database
    :param filter_source: None or list. If list, e.g. ['HPRD', 'SPIKE'],
        then only use interactions in at least one of the databases

    :param weights: tuple of potential scored fields AND ’sign’.
        Default ('sign', 'curation_effort', 'n_references', 'n_sources', 'n_primary_sources')
    :param weighted_edge: weighted_edge: edge weight, can be string, None, or tuple ('hybrid', string).
        If str, it refers to the weight used in Dijkstra's algorithm. e.g. 'reciprocal_n_references’.
            Note that there is no need to calculate ‘reciprocal_n_references' explicitly,
            as long as ’n_references’ is included in weights above.
        If None, all shortest paths (determined by length only) will be searched.
        If tuple ('hybrid', string), look for the weighted shortest paths among all shortest paths (not chosen by weights).
            E.g. ('hybrid', 'reciprocal_n_references'), then first look for all shortest paths
            (non-weighted, determined by length, equivalent to weighted_edge = None),
            then select the path with the smallest sum reciprocal_n_references among these paths.

    :param remove_cycle: Boolean, whether remove 2-edge cycles. Default True
    :param remove_cycle_by: if remove_cycle is True, then  remove 2-edge cycles by remove_cycle_by. Default: ’n_references'

    :param remove_sign: Boolean, whether remove redundant sign for interactions that are both activation and inhibition.
        If False, create two entries for the both interactions: one activation and one inhibition. Default: True
    :param consensus_sign: Boolean, whether use consensus sign if an interaction is both activation and inhibition
    :param sign_to_keep: ‘Activator’ or ‘Inhibitor’. Interactions that are both an activation or an inhibition
        will be changed to sign_to_keep (after processed with consensus sign)
    :param remove_neither: Boolean, whether remove interactions that are neither an activation or an inhibition.
        Default True
    :param remove_undir: Boolean, whether remove undirected edges. Default True

    :param to_combine: default ‘mut_deg_pheno’ to link mutations to DEGs to phenotypes.
        Can be changed to ‘mut_pheno’ which does not force DEGs between mutations and phenotypes
    :param replace_gene: False or dictionary.
        If dictionary, replace gene names with corresponding proteins based on their Uniprot ID
        (only available with Omnipath).
        e.g. replace CDKN2A with CDKN2A_arf and CDKN2A_p16 based on their Uniprot ID in Omnipath.
    :param filter_gene: False/None or iterable. filter by selected genes.
        if there exists paths from u to v that consists of only selected genes then discard other paths

    """

    def __init__(
        self,
        replace_gene: bool | dict = False,
        weighted_edge: bool | str | tuple[str, str] = False,
        filter_gene: bool | Iterable[str] = False,
        cut_off: bool | tuple[str, int] = False,
        remove_undir: bool = True,
        filter_source_prime: bool | str = False,
        filter_source: bool | Iterable[str] = False,
        pre_filter: bool | tuple[str, int] = ("n_primary_sources", 3),
        remove_cycle: bool = True,
        remove_sign: bool = True,
        remove_neither: bool = True,
        weights=(
            "sign",
            "curation_effort",
            "n_references",
            "n_sources",
            "n_primary_sources",
        ),
        sign_to_keep="Activator",
        consensus_sign=False,
        remove_cycle_by="n_references",
    ):
        # if (not isinstance(weighted, bool)) or (not isinstance(weighted, str)):
        #     raise TypeError('weighted must be either boolean or string')

        self.replace_gene = replace_gene
        self.weighted_edge = weighted_edge
        self.filter_gene = filter_gene
        self.cut_off = cut_off
        self.remove_undir = remove_undir
        self.filter_source_prime = filter_source_prime
        self.filter_source = filter_source
        self.pre_filter = pre_filter

        self.remove_cycle = remove_cycle
        self.remove_sign = remove_sign
        self.remove_neither = remove_neither

        self.weights = weights
        self.sign_to_keep = sign_to_keep
        self.consensus_sign = consensus_sign
        self.remove_cycle_by = remove_cycle_by

    def _get_op(self, int_op=None):
        """
        Extract data from Omnipath, with options to remove undirected edges/ambiguous edges/etc

        :param int_op: pandas.DataFrame, Omnipath database
        :return int_op: pandas.DataFrame, Omnipath database

        """

        if int_op is None:
            int_op = op.interactions.OmniPath.get(  # type: ignore
                organism="human", genesymbols=True, directed=False
            )

        if self.filter_source_prime:  # only use interactions contain at least one reference from filter_source_prime
            int_op = int_op[
                (~int_op["sources"].isna())
                & (int_op["sources"].str.contains(self.filter_source_prime))
            ]

        if self.filter_source:
            self.filter_source = set(self.filter_source)  # type: ignore
            int_op["source_filter"] = int_op["sources"].str.split(";")
            int_op["source_filter"] = int_op["source_filter"].apply(
                lambda x: list(set(x).intersection(self.filter_source))  # type: ignore
            )
            int_op = int_op[int_op["source_filter"].map(lambda x: len(x)) > 0]

        # replace duplicated genes
        if self.replace_gene:
            if not isinstance(self.replace_gene, dict):
                raise TypeError("replace_gene must be dict or boolean False")

            for k, v in self.replace_gene.items():
                for col in ["source", "target"]:
                    int_op.loc[(int_op[col] == k), "%s_genesymbol" % col] = v

        # undirected edges and edges with undefined direction are removed in this function (_get_op)
        # instead of the following function (_filter_op)
        # because undirected and undefined edges are currently removed by default
        # we can save the df after running this function

        # replace None in n_reference column with zero/0 (actually use a small number to avoid numeric errors)
        int_op.infer_objects(copy=False).replace(
            {"n_references": {None: 10**-13}}, inplace=True
        )

        # remove undirected edges
        if self.remove_undir:
            int_op = int_op[int_op["is_directed"]]  # remove undirected edges
        # if undirected is not remove, add additional edges representing the opposite directions
        # e.g. if A--B, add B->A and change A->B
        else:
            int_undir = int_op[~int_op["is_directed"]]
            int_undir = int_undir.rename(
                columns={
                    "source": "target",
                    "target": "source",
                    "source_genesymbol": "target_genesymbol",
                    "target_genesymbol": "source_genesymbol",
                }
            )
            int_op = pd.concat([int_op, int_undir], axis=0, ignore_index=True)

        int_op = get_direction(int_op)

        # remove neither
        if self.remove_neither:
            int_op = int_op[int_op["sign"] != "Neither"]

        # set index to (source gene, target gene)
        int_op.index = zip(int_op["source_genesymbol"], int_op["target_genesymbol"])  # type: ignore

        return int_op

    def _filter_op(self, int_op=None):
        """
        Filter Omnipath database

        :param int_op: pandas.DataFrame, Omnipath database
        :return int_op: pandas.DataFrame, filtered Omnipath database by specified columns and thresholds

        """

        if int_op is None:
            int_op = self._get_op()
        else:
            int_op.index = zip(int_op["source_genesymbol"], int_op["target_genesymbol"])

        # filter by pre_filter, usually n_primary_sources > 3
        if isinstance(self.pre_filter, tuple):
            int_op = int_op[
                int_op[self.pre_filter[0]] > self.pre_filter[1]
            ]  # filter entries

        # filter by cut_off
        if isinstance(self.cut_off, tuple):
            int_op = int_op[int_op[self.cut_off[0]] >= self.cut_off[1]]

        # remove sign
        if self.remove_sign:
            if self.consensus_sign:
                int_op.loc[int_op["sign"] == "Both", "sign"] = (
                    int_op.loc[
                        int_op["sign"] == "Both",
                        ["consensus_stimulation", "consensus_inhibition"],
                    ]
                    .apply(lambda x: ",".join(x.index[x]), axis=1)
                    .replace(
                        {
                            "consensus_stimulation": "Activator",
                            "consensus_inhibition": "Inhibitor",
                            "consensus_stimulation,consensus_inhibition": "Both",
                            "": "Both",
                        }
                    )
                )

            int_op.loc[int_op["sign"] == "Both", "sign"] = self.sign_to_keep

        return int_op

    def _get_graph(self, int_op):
        """
        Construct a graph from Omnipath data

        :param int_op: pandas.DataFrame, Omnipath database
        :return G: BMATool.graph.Graph, constructed graph with edges being the interactions in Omnipath

        """

        G = Graph(
            remove_cycle=self.remove_cycle,
            remove_sign=self.remove_sign,
            remove_neither=self.remove_neither,
        )
        G.gen_graph(
            df=int_op,
            weights=self.weights,
            sign_to_keep=self.sign_to_keep,
            remove_cycle_by=self.remove_cycle_by,
        )

        return G

    def shortest_path(
        self,
        gene_sets: dict,
        to_combine: list | str,
        int_op: pd.DataFrame | None = None,
        thre: int = 2,
        file_path: str | bool = False,
    ):
        """
        Extract the shortest paths between gene sets

        :param gene_sets: dict, key: type of sets, e.g. mut (mutant), pheno (phenotype), value: a gene set under corresponding type
        :param int_op: None or pandas.DataFrame.
            If None, automatically use Omnipath,
            else use the interactions and scores from passed DataFrame
        :param thre: max # path length linking same-type genes
            (mutation-mutation OR phenotype-phenotype).
            Default 2, which means no additional genes are allowed between two mutation nodes OR two phenotype nodes)
        :param to_combine: list or str.
            If list, e.g. [['mut', 'deg'], ['deg', 'pheno']], return every possible combination of
            mutation-DEG pairs and DEG-phenotype pairs.
            If str, e.g. ‘mut’, return every possible combination of mutation-mutation pairs.
        :param file_path: Boolean False or excel file path. If False, file will not be saved

        """

        # obtain and filter omnipath data
        int_op = self._filter_op(int_op)

        # construct a graph from omnipath filtered data
        # to extract shortest paths
        G = self._get_graph(int_op)

        # find shortest paths between genes sets in G
        df = find_path(
            G, gene_sets, to_combine, self.weighted_edge, self.filter_gene, thre
        )

        # expand shortest paths
        df_expand = expand_path(df)

        # remove duplicated entries in expanded paths
        df_remove_dup = remove_dup(df_expand)

        # merge df_remove_dup with int_op to obtain relevant info (n_ref, curation_efforts, etc)
        weights = list(self.weights)
        weights.remove("sign")

        df_remove_dup = pd.merge(
            df_remove_dup,
            int_op[weights],
            how="left",
            left_index=True,
            right_index=True,
        )

        # add style to df
        df = get_style(df)

        file_path_csv = f"{file_path}.csv" if isinstance(file_path, str) else False

        # write out df as csv to file file_path but do not append
        with open(file_path_csv, "w") as f:
            df_remove_dup.to_csv(
                f, header=True, index=False, quoting=csv.QUOTE_NONNUMERIC
            )

        # save to excel
        if file_path:
            for to_save, sheet_name in zip(
                (df, df_expand, df_remove_dup),
                ("shortest path", "expand", "remove dup"),
            ):
                save_excel(file_path, to_save, sheet_name=sheet_name)

        return df_remove_dup
