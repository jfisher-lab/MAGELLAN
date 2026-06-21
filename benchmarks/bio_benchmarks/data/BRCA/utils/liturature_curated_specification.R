knitr::opts_chunk$set(echo = TRUE)
setwd("~/Desktop/proteomic_nsclc_model/")

library(dplyr)
library(reticulate)
library(readxl)
library(jsonlite)
library(purrr)
library(stringr)
library(benchmarKIN)
library(tidyr)

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


# Create a list of experimental outcomes by Cell.Line
exp_dict <- lapply(unique(model_basal_behaviours$code), function(code) {
  cat(code, "\n")
  # Filter data for the current cell line
  cell_line_data <- model_basal_behaviours[model_basal_behaviours$code == code, ]

  # Convert to named list (dictionary) with the phenotype as keys and expect as values
  exp_outcomes <- setNames(
    unlist(purrr::map(stringr::str_split(cell_line_data$expect, pattern = "="), 2)),
    unlist(purrr::map(stringr::str_split(cell_line_data$expect, pattern = "="), 1))
  )
  # Initialize exp_perturbations to NULL
  exp_perturbations <- NULL
  # Check if the 'if.' column has any value that is not "NULL"
  if (any(cell_line_data$if. != "NULL")) {
    # Split the perturbations and process them
    seperate_perturbations <- unlist(stringr::str_split(cell_line_data$if., pattern = ","))
    # Handle the splitting and creation of named vector
    if (length(seperate_perturbations) > 0) {
      # Split into key-value pairs
      keys <- unlist(purrr::map(stringr::str_split(seperate_perturbations, pattern = "="), 1))
      values <- unlist(purrr::map(stringr::str_split(seperate_perturbations, pattern = "="), 2))
      # Remove duplicates by keeping the first occurrence of each key
      unique_keys <- !duplicated(keys)
      # Create exp_perturbations with unique keys
      exp_perturbations <- setNames(values[unique_keys], keys[unique_keys])
    }
  }
  list(
    pert = exp_perturbations, # Perturbation data
    exp = exp_outcomes # Experimental outcomes
  )
})


# Print the result to check
names(exp_dict) <- unique(model_basal_behaviours$code)

# Combine both pert and exp data into the final structure
basal_specification <- lapply(unique(model_basal_behaviours$code), function(code) {
  cell_line <- unlist(map(str_split(code, pattern = "\\."), 1))
  exp_pert <- c(perturbations_dict[[cell_line]], exp_dict[[code]]$pert)
  exp_outcomes <- exp_dict[[code]]$exp
  # Overwrite values in exp_pert only for proteins that exist in both dictionaries
  for (protein_name in names(exp_pert)) {
    if (protein_name %in% names(exp_dict[[code]]$pert)) {
      exp_pert[[protein_name]] <- exp_dict[[code]]$pert[[protein_name]]
    }
  }
  list(
    pert = exp_pert[!duplicated(names(exp_pert))], # Perturbation data
    exp = exp_outcomes[!duplicated(names(exp_outcomes))] # Experimental outcomes
  )
})

# Set names of the final structure to match the cell lines
names(basal_specification) <- unique(model_basal_behaviours$code)




rows <- list()

for (experiment in names(basal_specification)) {
  pert_values <- basal_specification[[experiment]]$pert
  exp_values <- basal_specification[[experiment]]$exp

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
spec_df$cell_line <- unlist(map(str_split(spec_df$cell_line, pattern = "\\."), first))

to_include <- c(
  "MCF10A.basal", "MCF7.basal", "MDA-MB-231.basal",
  "MCF7.estradiol.stimulation", "MCF7.estradiol.control",
  "MCF7.insulin.stimulation", "MCF7.insulin.control",
  "MCF10A.progesterone.stimulation", "MCF10A.progesterone.control"
)

write.csv(spec_df # [spec_df$gene != "CDKN2A",]
  ,
  file = "/path/to/literature_curated_specification.csv",
  quote = F
)
# basal spec
write.csv(
  spec_df[ # spec_df$gene != "CDKN2A"&
    grepl(spec_df$experiment_particular, pattern = "basal"),
  ],
  file = "/path/to/basal_specification.csv",
  quote = F
)
