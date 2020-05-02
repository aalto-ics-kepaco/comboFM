library(tidyverse)

# Read drug combination data
df <- 
  readr::read_csv(
    file = "data/NCI-ALMANAC_subset_555300.csv",
    col_types = readr::cols()
  ) 

# One-hot encoding --------------------------------------------------------

onehot_levels <- unique(c(df$Conc1, df$Conc2))

# Drug 1 concentrations
df_onehot_conc1 <- df %>% 
  dplyr::select(.data$Conc1) %>% 
  dplyr::mutate(Conc1 = factor(.data$Conc1, levels = onehot_levels)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot()

# Save the features
readr::write_csv(
  x = df_onehot_conc1,
  path = "data/drug1_concentration__one-hot_encoding.csv"
)

# Drug 2 concentrations
df_onehot_conc2 <- df %>% 
  dplyr::select(.data$Conc2) %>% 
  dplyr::mutate(Conc2 = factor(.data$Conc2, levels = onehot_levels)) %>% 
  data.table::as.data.table() %>% 
  mltools::one_hot()

# Save the features
readr::write_csv(
  x = df_onehot_conc2,
  path = "data/drug2_concentration__one-hot_encoding.csv"
)

# Concentration values ----------------------------------------------------

# Drug concentrations
df_conc_vals <- df %>% 
  dplyr::select(.data$Conc1, .data$Conc2) 

# Save the features
readr::write_csv(
  x = df_conc_vals,
  path = "data/drug1_drug2_concentration__values.csv"
)

# Drug concentrations, flip order
df_conc_vals <- df %>% 
  dplyr::select(.data$Conc2, .data$Conc1) 

# Save the features
readr::write_csv(
  x = df_conc_vals,
  path = "data/drug2_drug1_concentration__values.csv"
)








