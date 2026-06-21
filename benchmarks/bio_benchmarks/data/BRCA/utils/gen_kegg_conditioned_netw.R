library(BMAlayout)
library(biotubemapR)
library(tidyverse)
library(igraph)
library(jsonlite)
library(ggraph)
library(xml2)
library(jsonlite)

knitr::opts_chunk$set(echo = TRUE)


combine_graphs <- function(g1, g2, edge_attr_name) {
  # Merge the two graphs
  g_merged <- igraph::union(g1, g2)
  g1_attr_name <- paste0(edge_attr_name, "_1")
  g2_attr_name <- paste0(edge_attr_name, "_2")

  # If both graphs have the edge attribute, merge meaningfully (prioritize g1)
  if (edge_attr_name %in% edge_attr_names(g1) && edge_attr_name %in% edge_attr_names(g2)) {
    edge_attr(g_merged)[[edge_attr_name]] <- ifelse(is.na(edge_attr(g_merged)[[g1_attr_name]]), edge_attr(g_merged)[[g2_attr_name]], edge_attr(g_merged)[[g1_attr_name]])
  }

  # Clean up redundant attributes only if they exist
  if (g1_attr_name %in% edge_attr_names(g_merged)) {
    g_merged <- delete_edge_attr(g_merged, g1_attr_name)
  }
  if (g2_attr_name %in% edge_attr_names(g_merged)) {
    g_merged <- delete_edge_attr(g_merged, g2_attr_name)
  }

  # Remove redundant edges (edges with the same source, target, and type)
  g_merged <- igraph::simplify(g_merged, remove.loops = FALSE, edge.attr.comb = "first")

  return(g_merged)
}

# Function to add phenotype associations layer and remove redundant edges
add_phenotype_layer <- function(graph, phenotype_associations) {
  # Combine phenotype associations into a single data frame
  phenotype_associations_df <- bind_rows(phenotype_associations) %>%
    mutate(EFFECT = case_when(
      EFFECT == "down-regulates" ~ "Inhibitor",
      EFFECT == "up-regulates" ~ "Activator",
      EFFECT == "up-regulates activity" ~ "Activator",
      TRUE ~ EFFECT # Keep other values unchanged
    ))

  # Create phenotype graph using the edge list (only include relevant edges)
  pheno_graph <- data.frame(
    from = phenotype_associations_df$ENTITYA,
    to = phenotype_associations_df$ENTITYB,
    type = phenotype_associations_df$EFFECT
  )

  # Filter phenotype edges to those whose source nodes exist in the main graph
  pheno_graph <- unique(pheno_graph[pheno_graph$from %in% V(graph)$name, ])
  phenotype_names <- unique(pheno_graph$to)

  # Create phenotype graph using the filtered edge list
  g_pheno <- graph_from_data_frame(pheno_graph[, 1:2], directed = TRUE)
  E(g_pheno)$Type <- pheno_graph$type

  # Merge the main graph and the phenotype graph
  g_merged <- combine_graphs(graph, g_pheno, edge_attr_name = "Type")

  # Remove redundant edges (edges with the same source, target, and type)
  g_merged <- igraph::simplify(g_merged, remove.loops = FALSE, edge.attr.comb = "first")

  return(list(g_merged = g_merged, phenotype_names = phenotype_names))
}


# Function to create the BMA JSON with layout
make_bma_json_with_layout <- function(g_final, phenotype_names) {
  # Classify nodes
  node_classification <- classify_nodes(V(g_final)$name)
  node_classification$type[node_classification$name %in% phenotype_names] <- "phenotype"

  # Generate pathway layout
  pLayout <- pathwayLayout(g_final, type_df = node_classification)

  # Convert graph to BMA JSON
  model_json <- convert_graph_to_json_bma(g_final, pLayout$layout_matrix)
  node_table <- json_to_node_table(model_json)

  # Identify ligand names from the pLayout
  ligand_names <- pLayout$node_types$name[pLayout$node_types$type == "ligand"]

  # Update ligand nodes to be always active
  is_ligand <- node_table[, "Name"] %in% ligand_names
  node_table[is_ligand, "Formula"] <- "1"

  # Update model JSON with modified node table
  model_json$Model$Variables <- table_to_json(node_table)

  return(model_json)
}
# function to add value to list of values
add_value <- function(model_json, list_of_nodes, value) {
  node_table <- json_to_node_table(model_json)
  # Update nodes to value
  is_node <- node_table[, "Name"] %in% list_of_nodes
  node_table[is_node, "Formula"] <- value
  # Update model JSON with modified node table
  model_json$Model$Variables <- table_to_json(node_table)
  return(model_json)
}

# Sample inputs for phenotype associations
phenotype_associations <- list(
  apoptosis = read.delim(file = "./path/to/signor_phenotype_associations/apoptosis.tsv", sep = "\t"),
  proliferation = read.delim(file = "./path/to/signor_phenotype_associations/proliferation.tsv", sep = "\t"),
  survival = read.delim(file = "./path/to/signor_phenotype_associations/survival.tsv", sep = "\t")
)

model_json <- read_json(path = "./path/to/kegg_with_phenotype.json")
g_kegg <- json_to_igraph(model_json)$graph
# Load shortest path graph
shortest_path_df <- read.csv(file = "./path/to/brca_w_kinase_shortest_path.csv")
shortest_path_df$sign[shortest_path_df$sign == "inhibition"] <- "Inhibitor"
g_shortestpath <- graph_from_data_frame(shortest_path_df[, 1:2], directed = TRUE)
E(g_shortestpath)$Type <- shortest_path_df$sign

# Combine KEGG graph with shortest path graph
g_shortestpath_with_kegg <- combine_graphs(g_kegg, g_shortestpath, edge_attr_name = "Type")

g_shortestpath_with_kegg <- simplify(g_shortestpath_with_kegg, remove.loops = TRUE, edge.attr.comb = "first")
# Add phenotype layer to KEGG graph and g_shortestpath_with_kegg
shortestpath_with_kegg_with_pheno <- add_phenotype_layer(g_shortestpath_with_kegg, phenotype_associations)
shortestpath_with_kegg_with_pheno$g_merged <- delete_edges(
  shortestpath_with_kegg_with_pheno$g_merged,
  E(shortestpath_with_kegg_with_pheno$g_merged)[.from("MTOR") & .to("Proliferation")]
)
shortestpath_with_kegg_with_pheno$g_merged <- delete_edges(
  shortestpath_with_kegg_with_pheno$g_merged,
  E(shortestpath_with_kegg_with_pheno$g_merged)[.from("CTNNB1") & .to("Proliferation")]
)

# Function to trim the network, keeping only nodes with a path to phenotype nodes
trim_network_to_phenotypes <- function(graph, phenotype_names) {
  # Find all nodes that have a direct or indirect path to any phenotype node
  reachable_nodes <- unique(unlist(
    lapply(phenotype_names, function(phenotype) {
      if (phenotype %in% V(graph)$name) {
        # Find all nodes that can reach the phenotype (reverse for incoming paths if directed)
        subcomponent(graph, phenotype, mode = "in")
      } else {
        NULL
      }
    })
  ))

  # Filter the graph to include only reachable nodes
  trimmed_graph <- induced_subgraph(graph, vids = reachable_nodes)

  return(trimmed_graph)
}

E(shortestpath_with_kegg_with_pheno$g_merged)$Type[
  E(shortestpath_with_kegg_with_pheno$g_merged)$Type == "Inhibitor"
] <- "inhibition"
E(shortestpath_with_kegg_with_pheno$g_merged)$type <- E(shortestpath_with_kegg_with_pheno$g_merged)$Type
# usage
shortestpath_with_kegg_with_pheno$g_merged <- trim_network_to_phenotypes(shortestpath_with_kegg_with_pheno$g_merged, shortestpath_with_kegg_with_pheno$phenotype_names)


# Generate BMA JSONs for both graphs
json_keggShortestPath_withPheno <- make_bma_json_with_layout(shortestpath_with_kegg_with_pheno$g_merged, shortestpath_with_kegg_with_pheno$phenotype_names)



write_json(json_keggShortestPath_withPheno,
  path = "./path/to/kegg_shortest_path_with_phenotype.json", auto_unbox = TRUE
)
