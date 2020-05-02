library(tidyverse)

# Read drug combination data
df <- 
  readr::read_csv(
    file = "data/NCI-ALMANAC_subset_555300.csv",
    col_types = readr::cols()
  ) 

# One-hot encoding --------------------------------------------------------

onehot_levels <- unique(c(df$Drug1, df$Drug2))

# Drug 1 concentrations
df_onehot_drug1 <- df %>% 
  dplyr::select(.data$Drug1) %>% 
  dplyr::mutate(Drug1 = factor(.data$Drug1, levels = onehot_levels)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot()

# Save the features
readr::write_csv(
  x = df_onehot_drug1,
  path = "data/drug1__one-hot_encoding.csv"
)

# Drug 2 concentrations
df_onehot_drug2 <- df %>% 
  dplyr::select(.data$Drug2) %>% 
  dplyr::mutate(Drug2 = factor(.data$Drug2, levels = onehot_levels)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot()

# Save the features
readr::write_csv(
  x = df_onehot_drug2,
  path = "data/drug2__one-hot_encoding.csv"
)

# Molecular fingerprints --------------------------------------------------

# Read computed fingerprints
drug_fingerprints <-
  readr::read_csv(
    file = "data/drugs__estate_fingerprints.csv",
    col_types = readr::cols(),
    col_names = FALSE
  )

# Remove bits with zero variance across drugs
drug_fingerprints <- drug_fingerprints %>% 
  dplyr::select(-which(apply(drug_fingerprints, 2, var) == 0))


# Drug 1

# Match fingerprints with drugs in the drug combination data
drug1_fingerprints <- df %>% 
  dplyr::select(.data$Drug1) %>% 
  dplyr::left_join(
    drug_fingerprints %>% 
      rename(Drug1 = .data$X1),
    by = "Drug1"
  ) %>% 
  dplyr::select(-.data$Drug1) 

# Save the features
readr::write_csv(
  x = drug1_fingerprints,
  path = "data/drug1__estate_fingerprints.csv"
)


# Drug 2

# Match fingerprints with drugs in the drug combination data
drug2_fingerprints <- df %>% 
  dplyr::select(.data$Drug2) %>% 
  dplyr::left_join(
    drug_fingerprints %>% 
      rename(Drug2 = .data$X1),
    by = "Drug2"
  ) %>% 
  dplyr::select(-.data$Drug2)

# Save the features
readr::write_csv(
  x = drug2_fingerprints,
  path = "data/drug2__estate_fingerprints.csv"
)
