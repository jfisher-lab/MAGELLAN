# create shortest path network from Omnipath
import argparse
import os.path
from pathlib import Path

import pandas as pd
import toml  # Changed from configparser

from magellan.graph import Graph
from magellan.path import ShortestPath
from magellan.pydot_io import graph_to_pydot
from magellan.utils.file_io import save_pickle

##############
# initialise #
##############

# config = configparser.ConfigParser() # Removed
# config.read(os.path.join(path_root, "scripts", "mcconfig.ini")) # Removed

local_path = Path(__file__).parent.parent

default_config_path = os.path.join(
    local_path, "scripts", "example_shortest_path_config.toml"
)

# Use argparse to get config path from command line
parser = argparse.ArgumentParser(description="Generate shortest path network.")
parser.add_argument(
    "--config",
    type=str,
    default=default_config_path,
    help="Path to the TOML configuration file.",
)
args = parser.parse_args()

with open(
    args.config, "r"
) as f:  # "rb" is recommended for tomli, good practice for toml
    config = toml.load(f)


pipe_dir = config["DEFAULT"]["pipe_dir"]
data_dir = config["DEFAULT"]["data_dir"]
path_data = os.path.join(
    local_path, pipe_dir, data_dir, config["DEFAULT"]["shortest_path_data_dir"]
)
results_path = os.path.join(local_path, pipe_dir, config["DEFAULT"]["results_dir"])
os.makedirs(results_path, exist_ok=True)

# max n_genes in mut-mut paths or pheno-pheno paths (incl. mut/pheno genes at both ends)
thre = config["shortest_path_settings"]["thre"]
# if tuple, e.g. ('curation_effort', 15), then only include interactions with >= 15 curation_effort
cut_off = config["shortest_path_settings"]["cut_off"]
# use reciprocal n_ref as edge weight in Dijkstra's algorithm
weighted_edge = config["shortest_path_settings"]["weighted_edge"]
# use consensus sign if an interaction is both activation AND inhibition
consensus_sign = config["shortest_path_settings"]["consensus_sign"]
# link mutations to DEGs to phenotypes

# if source name, e.g. 'SIGNOR', then only use interactions in the source database
filter_source_prime = config["shortest_path_settings"].get("filter_source_prime")
# if list, e.g. ['HPRD', 'SPIKE'], then only use interactions in at least one of the databases
filter_source = config["shortest_path_settings"].get("filter_source")

#################
# load bma data #
#################

# replace proteins with the same gene names
replace_gene = config["shortest_path_settings"]["replace_gene"]  # No longer need eval()
# convert string to dict

# load bma genes
# df_gene = pd.read_csv(path_data + 'gene_node_table.csv')
gene_node_table_file = config["shortest_path_input"]["gene_node_table_file"]
df_gene = pd.read_csv(os.path.join(path_data, gene_node_table_file))
gene_replace_dict = config["shortest_path_settings"][
    "gene_replace_dict"
]  # No longer need eval()
df_gene.replace(
    to_replace=gene_replace_dict, inplace=True
)  # replace ERBB1 with EGFR as Omnipath only has EGFR

# replace duplicated terms in omnipath (mainly CDKN2A)
for v in replace_gene.values():
    df_gene.loc[
        (df_gene["node"].str.lower() == v.split("_")[-1].lower()), "gene_name"
    ] = v

# extract bma genes by type: mut (mutants), pheno (phenotypes), deg (differentially expressed genes)
mut = set(df_gene.loc[df_gene["mutant"] == "Y", "gene_name"].dropna())
pheno = set(df_gene.loc[df_gene["phenotype"] == "Y", "gene_name"].dropna())
deg = set(df_gene.loc[df_gene["key"] == "Y", "gene_name"].dropna())
gene_sets = {"mut": mut, "deg": deg, "pheno": pheno}

# manually set KRAS and HRAS as DEGs (alternatively the corresponding entries can be changed in gene_node_table.csv
# gene_sets['deg'].add('KRAS')
# gene_sets['deg'].add('HRAS')

# set of bma genes
bma_gene = set(df_gene["gene_name"].dropna())

####################
# generate network #
####################

# load df_merge (created by data related/filter omnipath by signor.py)
# only include interactions with a score >= signor_score in SIGNOR

# signor_score = 0.0
# df_merge = pd.read_csv(path_data + 'signor_omnipath (signor_23_11_22).csv.zip', compression='zip')
# df_merge = df_merge[df_merge['SCORE'] >= signor_score]


# generate the shortest path network
file_name = config["shortest_path_output"]["shortest_path_out_file_name"]

path = ShortestPath(
    weighted_edge=weighted_edge,
    cut_off=cut_off,
    filter_source_prime=filter_source_prime,
    filter_source=filter_source,
    filter_gene=bma_gene,
    replace_gene=replace_gene,
    consensus_sign=consensus_sign,
)
df_remove = path.shortest_path(
    int_op=None,  # or replace with a pre-defined DataFrame such as consensus score csv
    gene_sets=gene_sets,
    to_combine=[["mut", "deg"], ["deg", "pheno"]],
    thre=thre,
    file_path=os.path.join(results_path, file_name),
)

# generate graph and json
G = Graph()
G.gen_graph(df_remove)
save_pickle(os.path.join(results_path, file_name), G)

# see generate_json/sample_code.py and BMATool.json_io for more parameter usage
G.to_json(
    to_dir=results_path, model_name=file_name, func_type="default", gene_sets=gene_sets
)

# output svg and png of graph for automated documentation e.g. latex notebook
graph_to_pydot(
    G,
    out_path=os.path.join(results_path, file_name),
    mut=list(mut),  # Convert set to list
    pheno=list(pheno),  # Convert set to list
    deg=list(deg),  # Convert set to list
    format="svg",
)
graph_to_pydot(
    G,
    out_path=os.path.join(results_path, file_name),
    mut=list(mut),  # Convert set to list
    pheno=list(pheno),  # Convert set to list
    deg=list(deg),  # Convert set to list
    format="png",
)
