library(tidyverse)

data_dir <- "../comboFM_data/"
# Read drug combination data
df <- 
  readr::read_csv(
    file = paste0(data_dir,  "NCI-ALMANAC_subset_2225137__validation_train.csv"),
    col_types = readr::cols()
  ) 

df_validation <- 
  readr::read_csv(
    paste0(data_dir, "NCI-ALMANAC_subset_2540334__validation_test.csv"),
    col_types = readr::cols()
  ) 

# One-hot encoding --------------------------------------------------------

unique_drugs <- unique(c(df_validation$Drug1, df_validation$Drug2))

# Drug 1 concentrations
df_onehot_drug1 <- df %>% 
  dplyr::select(.data$Drug1) %>% 
  dplyr::mutate(Drug1 = factor(.data$Drug1, levels = unique_drugs)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot(sparsifyNAs = TRUE)

# Save the features
readr::write_csv(
  x = df_onehot_drug1,
  path = paste0(data_dir, "validation_data_train/drug1__one-hot_encoding.csv")
)

# Drug 2 concentrations
df_onehot_drug2 <- df %>% 
  dplyr::select(.data$Drug2) %>% 
  dplyr::mutate(Drug2 = factor(.data$Drug2, levels = unique_drugs)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot(sparsifyNAs = TRUE)

# Save the features
readr::write_csv(
  x = df_onehot_drug2,
  path = paste0(data_dir, "validation_data_train/drug2__one-hot_encoding.csv")
)

# Molecular fingerprints --------------------------------------------------

# Read computed fingerprints
drug_fingerprints <-
  readr::read_csv(
    file = paste0(data_dir, "additional_data/drugs__estate_fingerprints.csv"),
    col_types = readr::cols()
  )

drug_fingerprints <- drug_fingerprints %>% 
  dplyr::select(-which(apply(drug_fingerprints %>%  filter(.data$Drug %in% unique_drugs), 2, var) == 0))

# Drug 1

# Match fingerprints with drugs in the drug combination data
drug1_fingerprints <- df %>% 
  dplyr::select(.data$Drug1) %>% 
  dplyr::left_join(
    drug_fingerprints %>% 
      rename(Drug1 = .data$Drug),
    by = "Drug1"
  ) %>% 
  dplyr::select(-.data$Drug1) 

# Save the features
readr::write_csv(
  x = drug1_fingerprints,
  path = paste0(data_dir, "validation_data_train/drug1__estate_fingerprints.csv")
)


# Drug 2

# Match fingerprints with drugs in the drug combination data
drug2_fingerprints <- df %>% 
  dplyr::select(.data$Drug2) %>% 
  dplyr::left_join(
    drug_fingerprints %>% 
      rename(Drug2 = .data$Drug),
    by = "Drug2"
  ) %>% 
  dplyr::select(-.data$Drug2)

# Save the features
readr::write_csv(
  x = drug2_fingerprints,
  path = paste0(data_dir, "validation_data_train/drug2__estate_fingerprints.csv")
) 




# Validation set ----------------------------------------------------------


# Drug 1 concentrations
df_onehot_drug1 <- df_validation %>% 
  dplyr::select(.data$Drug1) %>% 
  dplyr::mutate(Drug1 = factor(.data$Drug1, levels = unique_drugs)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot(sparsifyNAs = TRUE)

# Save the features
readr::write_csv(
  x = df_onehot_drug1,
  path = paste0(data_dir, "validation_data/drug1__one-hot_encoding.csv")
)

# Drug 2 concentrations
df_onehot_drug2 <- df_validation %>% 
  dplyr::select(.data$Drug2) %>% 
  dplyr::mutate(Drug2 = factor(.data$Drug2, levels = unique_drugs)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot(sparsifyNAs = TRUE)

# Save the features
readr::write_csv(
  x = df_onehot_drug2,
  path = paste0(data_dir, "validation_data/drug2__one-hot_encoding.csv")
)


# Drug 1

# Match fingerprints with drugs in the drug combination data
drug1_fingerprints <- df_validation %>% 
  dplyr::select(.data$Drug1) %>% 
  dplyr::left_join(
    drug_fingerprints %>% 
      rename(Drug1 = .data$Drug),
    by = "Drug1"
  ) %>% 
  dplyr::select(-.data$Drug1) 

# Save the features
readr::write_csv(
  x = drug1_fingerprints,
  path = paste0(data_dir, "validation_data/drug1__estate_fingerprints.csv")
)


# Drug 2

# Match fingerprints with drugs in the drug combination data
drug2_fingerprints <- df_validation %>% 
  dplyr::select(.data$Drug2) %>% 
  dplyr::left_join(
    drug_fingerprints %>% 
      rename(Drug2 = .data$Drug),
    by = "Drug2"
  ) %>% 
  dplyr::select(-.data$Drug2)

# Save the features
readr::write_csv(
  x = drug2_fingerprints,
  path = paste0(data_dir, "validation_data/drug2__estate_fingerprints.csv")
) 

