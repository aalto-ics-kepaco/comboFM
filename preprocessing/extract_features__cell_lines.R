library(tidyverse)

# Read drug combination data
df <- 
  readr::read_csv(
    file = "data/NCI-ALMANAC_subset_555300.csv",
    col_types = readr::cols()
  ) 

# One-hot encoding --------------------------------------------------------

df_onehot <- df %>% 
  dplyr::select(.data$CellLine) %>% 
  dplyr::mutate(CellLine = as.factor(.data$CellLine)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot()


# Save the features
readr::write_csv(
  x = df_onehot,
  path = "data/cell_lines__one-hot_encoding.csv"
)

# Gene expression ---------------------------------------------------------

# Read gene expression data (obtained from cellmineR R package)
df_expr <-
  readr::read_delim(
    file = "data/NCI-60__gene_expression.txt",
    delim = " ",
    col_types = cols()) %>% 
  dplyr::rename(gene = .data$X1) %>% 
  tidyr::drop_na()

# Compute variance for each gene across cell lines
df_expr <- df_expr %>% 
  dplyr::mutate(
    var = matrixStats::rowVars(
      x = df_expr %>% 
        select(-.data$gene) %>% 
        as.matrix, 
      na.rm = T)
  ) %>% 
  # Arrange by descending variance
  dplyr::arrange(desc(var)) 

# Select 0.5% of genes with the highest variance
n_genes <- round(0.005 * nrow(df_expr))

# Get the threshold for variance
var_thresh <- df_expr[n_genes, ] %>% 
  dplyr::pull(var) %>% 
  as.numeric()

df_expr <- df_expr %>% 
  dplyr::filter(.data$var >= var_thresh) %>%
  dplyr::select(-.data$var) %>% 
  tidyr::gather(key = "CellLine", value = "gene_expression", -.data$gene) %>% 
  dplyr:: mutate(
    CellLine = str_remove_all(.data$CellLine, ".*:"),
    CellLine = str_replace_all(.data$CellLine, "_", "-"),
    # Manually rename some of the cell lines to match those in the drug combination data
    CellLine = dplyr::case_when(
    .data$CellLine == "A549" ~ "A549/ATCC",
    .data$CellLine == "COLO205" ~ "COLO 205",
    .data$CellLine == "CSF-268" ~ "SF-268",
    .data$CellLine == "HL-60" ~ "HL-60(TB)",
    .data$CellLine == "HS578T" ~ "HS 578T",
    .data$CellLine == "LOXIMVI" ~ "LOX IMVI",
    .data$CellLine == "MDA-MB-231" ~ "MDA-MB-231/ATCC",
    .data$CellLine == "MDA-N" ~ "MDA-MB-468",
    .data$CellLine == "NCI-ADR-RES" ~ "NCI/ADR-RES",
    .data$CellLine == "RXF-393" ~ "RXF 393",
    .data$CellLine == "T47D" ~ "T-47D",
    TRUE ~ as.character(.data$CellLine))
  ) %>% 
  tidyr::spread(key = .data$gene, value = .data$gene_expression)

# Match the gene expression data with the rows in the drug combination data
df_expr <- df %>% 
  dplyr::select(.data$CellLine) %>% 
  dplyr::left_join(
    df_expr,
    by = "CellLine"
  ) %>% 
  dplyr::select(-.data$CellLine)

# Save the features
readr::write_csv(
  x = df_expr,
  path = "data/cell_lines__gene_expression.csv"
)






