knitr::opts_chunk$set(echo = TRUE)

library(dplyr)
library(reticulate)
library(readxl)
library(jsonlite)
library(purrr)
library(stringr)
library(benchmarKIN)
library(tidyr)
# Load necessary libraries
library(ComplexHeatmap)
library(circlize)
library(ggplot2)
library(viridis)
library(stringr)
library(purrr)

# Read the CSV file
model_init <- read.csv(file = "/path/to/model_init.csv")
model_init <- model_init[, c("Cell.Line", "Gene", "Perturbation")]

# Remove rows with NA values in Perturbation
model_init <- model_init[!is.na(model_init$Perturbation), ]
# Reshape the data into a hierarchical list format, grouped by Cell.Line
perturbations_dict <- lapply(unique(model_init$Cell.Line), function(cell_line) {
  # Filter data for the current cell line
  cell_line_data <- model_init[model_init$Cell.Line == cell_line, ]
  # Convert to named list (dictionary) with gene names as keys
  gene_perturbations <- setNames(cell_line_data$Perturbation, cell_line_data$Gene)
  # Remove NA values if any
  gene_perturbations <- gene_perturbations[!is.na(gene_perturbations)]
  gene_perturbations
})
# Set names of the list for each cell line
names(perturbations_dict) <- unique(model_init$Cell.Line)



# Read the CSV file
model_basal_behaviours <- read.csv(file = "/path/to/model_basal_behaviours.csv")
model_basal_behaviours <- model_basal_behaviours[grep(model_basal_behaviours$code, pattern = "basal"), ]
# Create a list of experimental outcomes by Cell.Line
exp_dict <- lapply(unique(model_basal_behaviours$cell_line), function(cell_line) {
  # Filter data for the current cell line
  cell_line_data <- model_basal_behaviours[model_basal_behaviours$cell_line == cell_line, ]
  # Convert to named list (dictionary) with the phenotype as keys and expect as values
  exp_outcomes <- setNames(
    as.numeric(unlist(map(str_split(cell_line_data$expect, pattern = "="), 2))),
    unlist(map(str_split(cell_line_data$expect, pattern = "="), 1))
  )
  exp_outcomes
})
# Set names of the list for each cell line
names(exp_dict) <- unique(model_basal_behaviours$cell_line)
# Combine both pert and exp data into the final structure
basal_specification <- lapply(names(perturbations_dict), function(cell_line) {
  list(
    pert = perturbations_dict[[cell_line]], # Perturbation data
    exp = exp_dict[[cell_line]] # Experimental outcomes
  )
})
# Set names of the final structure to match the cell lines
names(basal_specification) <- paste0(names(perturbations_dict), "[basal]")
basal_spec_df_list <- lapply(basal_specification, stack)
basal_spec_df_list <- lapply(basal_spec_df_list, function(x) {
  x$node <- rownames(x)
  return(x)
})
basal_spec_df <- bind_rows(basal_spec_df_list, .id = "exp")
rownames(basal_spec_df) <- NULL


cell_line_info <- read.csv(file = "./path/to/hijazi_hernandez/cell_line_info.csv")
hernandez_meta_data <- read_xlsx(path = "./path/to/hijazi_hernandez/hernandez_meta_data.xlsx", sheet = 2)
drug_summary <- read.csv(file = "./path/to/drug_summary.csv")
drug_summary$primary_target <- unlist(map(str_split(drug_summary$Direct_Target, pattern = "/"), first))
# Add the simplified_drug_name column based on the Activity column
drug_summary$simplified_drug_name <- ifelse(drug_summary$Activity == 0,
  paste(drug_summary$primary_target, "inhibitors"),
  ifelse(drug_summary$Activity == 2,
    paste(drug_summary$primary_target, "activators"),
    NA
  )
)

# Load meta data
meta <- load_meta()
# summary of cell lines from cancer in both
hernandez_meta_data$Biological_Sample <- gsub("\\s*\\(.*?\\)", "", hernandez_meta_data$Biological_Sample)
breast_hernandez_meta <- hernandez_meta_data[hernandez_meta_data$Biological_Sample %in% cell_line_info$Cell.Line[cell_line_info$Tissue == "Breast"], ]
breast_hernandez_meta$Description <- gsub(" ", "_", breast_hernandez_meta$Description, fixed = TRUE)
cell_line_To_id <- data.frame(
  cell.line = breast_hernandez_meta$Biological_Sample,
  id = breast_hernandez_meta$Condition
)
meta[is.na(meta$cell_line), ]$cell_line <- cell_line_To_id$cell.line[match(meta[is.na(meta$cell_line), ]$id, cell_line_To_id$id)]
dim(meta)
meta <- meta[!is.na(meta$cell_line), ]
dim(meta)

meta[is.na(meta$treatment), ]$treatment <- unlist(map(strsplit(unlist(meta$Description[is.na(meta$treatment)]), split = " "), 1))
meta <- meta[meta$cell_line %in% unique(breast_hernandez_meta$Biological_Sample), ]
meta$direct_target <- drug_summary$Direct_Target[match(meta$treatment, drug_summary$Compound)]
meta$primary_target <- unlist(map(str_split(meta$direct_target, pattern = "/"), first))
meta$drug <- meta$treatment
meta$treatment <- NULL

#
# #get kinase activity scores
mat <- load_perturbData()
phosphositeplus$target <- paste(phosphositeplus$target, phosphositeplus$target_protein, phosphositeplus$position, sep = "|")
ppsp <- phosphositeplus %>%
  dplyr::select(source, target, mor) %>%
  dplyr::distinct()
# get activities from zscore
act_scores <- run_zscore(mat = mat, network = ppsp)
write.csv(x = act_scores, file = "./path/to/kinase_activity_z_scores_hijazi.csv")
# # or

# act_scores<-read.csv(file = "./path/to/kinase_activity_z_scores_hijazi.csv", row.names = 1)

# Function to compute z-scores and discretize into 0, 1, or 2
discretize_by_zscore <- function(values, abs_cut_off = 2) {
  z_scores <- scale(values) # Compute z-scores

  # Discretization based on z-scores: -1 < z < 1 as "1", else "0" or "2"
  discretized <- ifelse(z_scores < -abs_cut_off, 0,
    ifelse(z_scores > abs_cut_off, 2, 1)
  )

  return(discretized)
}
# Use check.names = FALSE to avoid column name transformation
discretized_kinase_data <- as.data.frame(lapply(act_scores, discretize_by_zscore), check.names = FALSE)
# Add the row names (genes) back as a column
discretized_kinase_data$Kinase <- rownames(act_scores)
# Use tidyr::pivot_longer to melt the data and filter out Activity == 1 and NA values
melted_data <- discretized_kinase_data %>%
  pivot_longer(cols = -Kinase, names_to = "id", values_to = "Activity") %>%
  filter(Activity != 1 & !is.na(Activity))
melted_data$pert <- paste0(meta$cell_line, ".", meta$drug)[match(melted_data$id, meta$id)]
melted_data$pert_activity <- drug_summary$Activity[match(meta$drug[match(melted_data$id, meta$id)], drug_summary$Compound)]

melted_data$direct_target <- meta$primary_target[match(melted_data$id, meta$id)]
melted_data <- melted_data[!is.na(melted_data$pert), ]
# Group by direct target and kinase
consistent_kinase_activity <- melted_data %>%
  group_by(direct_target, Kinase) %>%
  summarize(
    no_of_drugs = n(), # Count total activities (number of drugs)
    unique_activities = n_distinct(Activity), # Count unique activities to check consistency
    consistent_activity = ifelse(unique_activities == 1, first(Activity), NA), # If consistent, return that activity
    drugs_used = paste(pert, collapse = ";"), # List all drugs used (concatenated by ;)
    pert_activity = unique(pert_activity)
  ) %>%
  filter(!is.na(consistent_activity)) # Keep only kinases with consistent activity
# get strongly supported kinase activities
strong_kinase_activity <- consistent_kinase_activity[consistent_kinase_activity$no_of_drugs >= 2, ]
cell_line_drug_clean <- gsub("([^.]+)\\.(.*)", "\\1[treatment=\\2]", strong_kinase_activity$drugs_used)
strong_kinase_activity$drugs_used <- gsub("(\\w+)\\.", "", cell_line_drug_clean)
# remove the observations were kinases are a direct target of the drug (having studied them to make sure they make sense - above)
# Convert direct_target to a list of targets and filter rows where Kinase is in direct_target
strong_kinase_activity <- strong_kinase_activity %>%
  rowwise() %>%
  mutate(targets_list = list(strsplit(direct_target, "/")[[1]])) %>%
  filter(!Kinase %in% unlist(targets_list)) %>%
  select(-targets_list) # Remove the intermediate column


# construct specification and add the model initialisations for the relevant cell line models
# Reshape the data into a hierarchical list format, grouped by Experiment
spec_dict <- lapply(unique(strong_kinase_activity$drugs_used), function(experiment) {
  # Filter data for the current experiment
  experiment_data <- strong_kinase_activity[strong_kinase_activity$drugs_used == experiment, ]
  # Convert to named list (dictionary) with kinases and their activities
  gene_perturbations <- setNames(experiment_data$consistent_activity, experiment_data$Kinase)
  # Remove NA values if any
  gene_perturbations <- gene_perturbations[!is.na(gene_perturbations)]
  gene_perturbations
})
names(spec_dict) <- unique(strong_kinase_activity$drugs_used)

# Combine both pert and exp data into the final structure
drug_specification <- lapply(names(spec_dict), function(experiment) {
  unique_targets <- strong_kinase_activity %>%
    filter(drugs_used == experiment) %>%
    select(direct_target, pert_activity) %>%
    mutate(direct_target = str_split(direct_target, "/")) %>%
    unnest(direct_target) %>%
    distinct(direct_target, pert_activity) %>%
    arrange(direct_target)

  # Create a named list directly
  unique_targets <- setNames(unique_targets$pert_activity, unique_targets$direct_target)

  # Get the perturbation data and replace values with unique_targets where applicable
  perturbations <- perturbations_dict[[gsub("\\[.*", "", experiment)]]

  # Replace values in perturbations with those in unique_targets if they share the same name
  combined_perturbations <- setNames(
    sapply(names(perturbations), function(name) {
      if (name %in% names(unique_targets)) {
        unique_targets[[name]]
      } else {
        perturbations[[name]]
      }
    }),
    names(perturbations)
  )

  list(
    pert = c(combined_perturbations, unique_targets[!names(unique_targets) %in% names(combined_perturbations)]), # Perturbation data
    exp = spec_dict[[experiment]] # Experimental outcomes
  )
})
names(drug_specification) <- names(spec_dict)


stacked_spec <- lapply(drug_specification, stack)
stacked_spec <- bind_rows(stacked_spec, .id = "experiment")
stacked_spec$Gene <- gsub("\\.\\.\\..*$", "", rownames(stacked_spec)) # Remove everything after the first occurrence of '...'
rownames(stacked_spec) <- NULL

kinases_measured_at_least_3_times <- stacked_spec %>%
  filter(ind == "exp") %>% # Filter for 'exp' rows
  group_by(Gene) %>% # Group by Gene
  summarize(count = n(), .groups = "drop") %>% # Count occurrences
  filter(count > 3) %>% # Filter for counts greater than 3
  pull(Gene)

genes_perturbed_in_some_way <- unique(stacked_spec$Gene[stacked_spec$ind == "pert"])
genes_mutated_in_some_way <- model_init$Gene

all_genes <- unlist(unique(c(
  kinases_measured_at_least_3_times,
  genes_perturbed_in_some_way,
  genes_mutated_in_some_way
)))
node_list <- data.frame(
  gene_name = all_genes,
  mut = all_genes %in% genes_mutated_in_some_way,
  pheno = all_genes %in% kinases_measured_at_least_3_times,
  deg = all_genes %in% genes_perturbed_in_some_way
)
# write.csv(node_list, file = "../data/shortest_path_node_list.csv")


# Prepare the DataFrame
rows <- list()

# combine literature specification and drug specification
res_specification <- c(drug_specification)

for (experiment in names(res_specification)) {
  pert_values <- res_specification[[experiment]]$pert
  exp_values <- res_specification[[experiment]]$exp

  # Create a row for perturbation
  pert_df <- data.frame(
    cell_line = gsub("\\[.*?\\]", "", experiment),
    source = "",
    experiment_particular = experiment,
    gene = names(pert_values),
    perturbation = unname(pert_values),
    expectation_bma = "",
    mean_result = unname(pert_values)
  )
  exp_df <- data.frame(
    cell_line = gsub("\\[.*?\\]", "", experiment),
    source = "",
    experiment_particular = experiment,
    gene = names(exp_values),
    perturbation = "",
    expectation_bma = unname(exp_values),
    mean_result = unname(exp_values)
  )

  # Add the row to the DataFrame
  rows[[experiment]] <- rbind(pert_df, exp_df)
}

# Display the DataFrame
spec_df <- bind_rows(rows)
spec_df[spec_df$gene == "ABL1", ]$gene <- "SRC"

# Extract drug names from experiment_particular and store them in a new column
spec_df <- spec_df %>%
  mutate(
    real_drug_names = str_extract_all(experiment_particular, "(?<=treatment=)[^;\\]]+|(?<=;)[^;\\]]+")
  )
# Map the drug names to simplified names based on primary_target inhibitor/activator
# Join `spec_df` with `drug_summary` on Direct_Target or primary_target and use simplified_drug_name
# Prepare a lookup table for drug name to simplified name
drug_lookup <- drug_summary %>%
  select(Compound, simplified_drug_name) %>%
  unique()
# Apply the mapping
spec_df <- spec_df %>%
  rowwise() %>%
  mutate(
    experiment_particular = paste(unique(drug_lookup$simplified_drug_name[drug_lookup$Compound %in% unlist(real_drug_names)]), collapse = "; "),
    real_drug_names = paste(unique(unlist(real_drug_names)), collapse = "; ")
  ) %>%
  ungroup()
# Add formatted column with the desired format
spec_df <- spec_df %>%
  mutate(experiment_particular = paste0(cell_line, ":[Treatment=", experiment_particular, "]"))

write.csv(spec_df,
  file = "/path/to/drug_specification.csv",
  quote = F
)

# read and combine

literature_curated_spec <- read.csv(file = "/path/to/literature_curated_specification.csv")
literature_curated_spec$real_drug_names <- ""
combined_spec <- rbind(
  read.csv(file = "/path/to/drug_specification.csv"),
  literature_curated_spec
)
combined_spec$X <- NULL
combined_spec[is.na(combined_spec)] <- ""
write.csv(combined_spec,
  file = "/path/to/combined_specification.csv",
  quote = F
)


library(ggpubr)
library(wesanderson)
library(patchwork)
library(cowplot)

dot_in <- consistent_kinase_activity %>%
  mutate(
    real_drug_names = str_extract_all(drugs_used, "(?<=\\.)[^;]+")
  )
drug_lookup <- drug_summary %>%
  select(Compound, simplified_drug_name) %>%
  unique()
dot_in <- dot_in %>%
  rowwise() %>%
  mutate(
    experiment_particular = paste(unique(drug_lookup$simplified_drug_name[drug_lookup$Compound %in% unlist(real_drug_names)]), collapse = "; "),
    real_drug_names = paste(unique(unlist(real_drug_names)), collapse = "; ")
  ) %>%
  ungroup()

# Sort `experiment_particular` by `Kinase` and `no_of_drugs` from the scatter plot
ordered_experiments <- dot_in %>%
  filter(no_of_drugs >= 2) %>%
  arrange(desc(no_of_drugs)) %>%
  distinct(experiment_particular) %>%
  pull(experiment_particular)

# Expand drugs_used column by separating at ";"
expanded_df <- dot_in %>%
  separate_rows(drugs_used, sep = ";") %>%
  group_by(experiment_particular) %>%
  summarise(count = n_distinct(drugs_used)) %>%
  arrange(desc(count))

# Set the same factor levels for `experiment_particular` across both datasets for alignment
dot_in$experiment_particular <- factor(dot_in$experiment_particular, levels = ordered_experiments)
expanded_df$experiment_particular <- factor(expanded_df$experiment_particular, levels = ordered_experiments)

# Ensure alignment on shared axis

# Define the number of top kinases to display
n <- 30 

# Filter the data for top n kinases based on no_of_drugs
top_n_kinases <- dot_in %>%
  filter(no_of_drugs >= 2) %>%
  group_by(Kinase) %>%
  summarise(max_drugs = max(no_of_drugs)) %>%
  arrange(desc(max_drugs)) %>%
  slice_head(n = n) %>%
  pull(Kinase)

# Filter the original data to include only the top n kinases
dot_in_top <- dot_in %>%
  filter(no_of_drugs >= 2 & Kinase %in% top_n_kinases)
