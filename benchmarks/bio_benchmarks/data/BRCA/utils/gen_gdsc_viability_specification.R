library(dplyr)
library(ggplot2)
# Load necessary libraries
library(dplyr)
library(tidyr)
library(stringr)
library(BMAlayout) # for extracting kegg graphd
library(biotubemapR)
library(jsonlite)

# from GDSC2 https://www.cancerrxgene.org/cellline/T47D/905945?screening_set=GDSC2

drug_viability <- rbind(
  data.frame(read.csv(file = "./path/to/MCF7.csv"),
    cell.line = "MCF7"
  ),
  data.frame(read.csv(file = "./path/to/MDA-MB-231.csv"),
    cell.line = "MDA-MB-231"
  ),
  data.frame(read.csv(file = "./path/to/T47D.csv"),
    cell.line = "T47D"
  ),
  data.frame(read.csv(file = "./path/to/BT-474.csv"),
    cell.line = "BT-474"
  ),
  data.frame(read.csv(file = "./path/to/MDA-MB-468.csv"),
    cell.line = "MDA-MB-468"
  ),
  data.frame(read.csv(file = "./path/to/HCC1954.csv"),
    cell.line = "HCC1954"
  )
)

# Add the cell.line.rank column
drug_viability <- drug_viability %>%
  group_by(cell.line) %>% # Group by cell line
  mutate(cell.line.rank = rank(Z.Score, ties.method = "first")) # Rank Z-Scores for each cell line

drug_viability_subset <- drug_viability[abs(drug_viability$Z.Score) > 1, ]
drug_viability_subset <- drug_viability_subset[drug_viability_subset$Targets != "", ]



# Separate each target into its own row
drug_viability_expanded <- drug_viability_subset %>%
  separate_rows(Targets, sep = ",\\s*") # Split Targets into separate rows

# Create a column for activity based on Z.Score
drug_viability_expanded <- drug_viability_expanded %>%
  mutate(activity = ifelse(Z.Score > 0, "Insensitive (Basal)", "Sensitive"))

# Group by targets and cell line and check for consistency
activity_summary <- drug_viability_expanded %>%
  group_by(Targets, cell.line) %>%
  summarise(
    consistent = n_distinct(activity) == 1, # TRUE if all activities are the same within the cell line
    activities = paste(unique(activity), collapse = ", "),
    drugs = paste(unique(Drug.Name), collapse = ", "),
    .groups = "drop"
  )

# find targets that are contained within our model.
model_json <- read_json(path = "./path/to/kegg_literature_run_synthetic_weight_est_realSpec.json")
model_json_out <- json_to_igraph(model_json)
activity_summary <- activity_summary[activity_summary$Targets %in% model_json_out$layout$Name, ]

# Separate consistent and inconsistent activities
consistent_activities <- activity_summary %>%
  filter(consistent == TRUE) %>%
  select(Targets, cell.line, activities, drugs)

inconsistent_activities <- activity_summary %>%
  filter(consistent == FALSE) %>%
  select(Targets, cell.line, activities, drugs)



consistent_activities <- consistent_activities %>%
  mutate(label = paste(activities, "(by", drugs, "inhibition)"))

# Print the results
print("Consistent Activities:")
print(consistent_activities)

print("Inconsistent Activities:")
print(inconsistent_activities)

ggplot(drug_viability, aes(x = cell.line.rank, y = Z.Score)) +
  geom_point(aes(color = abs(Z.Score) > 1, alpha = abs(Z.Score) <= 1)) + # Different color and alpha
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "black")) + # Highlight high Z-scores in red
  scale_alpha_manual(values = c("TRUE" = 1, "FALSE" = 0.3)) + # Reduce alpha for points near 0
  labs(
    title = "Relative Sensitivity of IC50 Z-Score vs Drugs (Ranked by Drug Sensitivity within Each Cell Line)",
    x = "Drug Rank (by Sensitivity within Cell Line)",
    y = "IC50 Z-Score"
  ) +
  facet_wrap(~cell.line) + # Facet by Cell Line
  cowplot::theme_cowplot() +
  theme(
    plot.title = element_text(size = 15, face = "bold"),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
    axis.text.x = element_text(angle = 90, hjust = 1),
    legend.position = "none" # Rotate x-axis labels
  ) +
  grids(linetype = "dashed") +
  geom_hline(yintercept = 0, color = "black") + # Dotted line at y = 0
  geom_hline(yintercept = 1, linetype = "dotted", color = "red") + # Dotted line at y = 0
  geom_hline(yintercept = -1, linetype = "dotted", color = "red") # Dotted line at y = 0

library(ComplexHeatmap)
library(dplyr)
library(tidyr)
library(circlize)

# Convert activities to numeric values
consistent_activities <- consistent_activities %>%
  mutate(activity_numeric = case_when(
    activities == "Sensitive" ~ "Sensitive",
    activities == "Insensitive (Basal)" ~ "Insensitive",
    TRUE ~ NA_character_ # Keep NA for missing values
  ))

# Pivot the data to a wide format
activity_matrix <- consistent_activities %>%
  group_by(cell.line, Targets) %>%
  summarise(activity_numeric = first(activity_numeric), .groups = "drop") %>% # Ensuring uniqueness
  pivot_wider(names_from = Targets, values_from = activity_numeric) %>%
  column_to_rownames("cell.line")

# Convert to matrix format
activity_matrix <- as.matrix(activity_matrix)

# Define discrete color mapping
activity_colors <- c("Sensitive" = "darkgreen", "Insensitive" = "red", "NA" = "darkgrey")

# Create heatmap with custom titles for rows and columns
Heatmap(
  activity_matrix,
  col = activity_colors,
  rect_gp = gpar(col = "white", lwd = 5), # White grid lines for separation
  na_col = "darkgrey", # Grey for missing values
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  row_names_side = "left",
  column_names_side = "top",
  row_title = "Cell Line", # Label for rows
  column_title = "Drug Inhibition Target", # Label for columns
  heatmap_legend_param = list(title = "Activity")
)
