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

unique_concentrations <- unique(c(df_validation$Conc1, df_validation$Conc2))

# Drug 1 concentrations
df_onehot_conc1 <- df %>% 
  dplyr::select(.data$Conc1) %>% 
  dplyr::mutate(Conc1 = factor(.data$Conc1, levels = unique_concentrations)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot(sparsifyNAs = TRUE)

# Save the features
readr::write_csv(
  x = df_onehot_conc1,
  path = paste0(data_dir, "validation_data_train/drug1_concentration__one-hot_encoding.csv")
)

# Drug 2 concentrations
df_onehot_conc2 <- df %>% 
  dplyr::select(.data$Conc2) %>% 
  dplyr::mutate(Conc2 = factor(.data$Conc2, levels = unique_concentrations)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot(sparsifyNAs = TRUE)

# Save the features
readr::write_csv(
  x = df_onehot_conc2,
  path = paste0(data_dir, "validation_data_train/drug2_concentration__one-hot_encoding.csv")
)

# Concentration values ----------------------------------------------------

# Drug concentrations
df_conc_vals <- df %>% 
  dplyr::select(.data$Conc1, .data$Conc2) 

# Save the features
readr::write_csv(
  x = df_conc_vals,
  path = paste0(data_dir, "validation_data_train/drug1_drug2_concentration__values.csv")
)

# Drug concentrations, flip order
df_conc_vals <- df %>% 
  dplyr::select(.data$Conc2, .data$Conc1) 

# Save the features
readr::write_csv(
  x = df_conc_vals,
  path = paste0(data_dir,  "validation_data_train/drug2_drug1_concentration__values.csv")
)


# Validation set ----------------------------------------------------------



# One-hot encoding 

# Drug 1 concentrations
df_onehot_conc1 <- df_validation %>% 
  dplyr::select(.data$Conc1) %>% 
  dplyr::mutate(Conc1 = factor(.data$Conc1, levels = unique_concentrations)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot(sparsifyNAs = TRUE)

# Save the features
readr::write_csv(
  x = df_onehot_conc1,
  path = paste0(data_dir, "validation_data/drug1_concentration__one-hot_encoding.csv")
)

# Drug 2 concentrations
df_onehot_conc2 <- df_validation %>% 
  dplyr::select(.data$Conc2) %>% 
  dplyr::mutate(Conc2 = factor(.data$Conc2, levels = unique_concentrations)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot(sparsifyNAs = TRUE)

# Save the features
readr::write_csv(
  x = df_onehot_conc2,
  path = paste0(data_dir, "validation_data/drug2_concentration__one-hot_encoding.csv")
)

# Concentration values ----------------------------------------------------

# Drug concentrations
df_conc_vals <- df_validation %>% 
  dplyr::select(.data$Conc1, .data$Conc2) 

# Save the features
readr::write_csv(
  x = df_conc_vals,
  path = paste0(data_dir, "validation_data/drug1_drug2_concentration__values.csv")
)

# Drug concentrations, flip order
df_conc_vals <- df_validation %>% 
  dplyr::select(.data$Conc2, .data$Conc1) 

# Save the features
readr::write_csv(
  x = df_conc_vals,
  path = paste0(data_dir, "validation_data/drug2_drug1_concentration__values.csv")
)









