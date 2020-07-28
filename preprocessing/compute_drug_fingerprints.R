library("rJava")
library("rcdk")
library("fingerprint")
library("tidyverse")

data_dir <- "../comboFM_data/"

df <- 
  readr::read_csv(
    #file = paste0(data_dir, "additional_data/drugs__SMILES.csv"),
    file ="drugs__SMILES.csv",
    comment = "",
    col_types = readr::cols()
  )

# Read SMILES from text file 
smiles <- df %>% 
  dplyr::pull(.data$SMILE)

# Parse a vector of SMILES to generate a list of IAtomContainer objects
molecules = parse.smiles(smiles)

fps = list()
fps_vectors = list()

for (i in 1:length(molecules)){
  
  do.typing(molecules[[i]])
  do.aromaticity(molecules[[i]])
  do.isotopes(molecules[[i]])
  
  fps[[i]] = get.fingerprint(molecules[[i]],  type="estate", verbose=TRUE)
  fps_vectors[[i]] = bit.spectrum(fps[i])
}

df_fps <- df %>% 
  dplyr::select(.data$Drug) %>% 
  dplyr::bind_cols(
    fps_vectors %>% 
      as.data.frame() %>% 
      t() %>% 
      as.data.frame() 
  )

readr::write_csv(
  x = df_fps,
  path = "drugs__estate_fingerprints.csv",
  col_names = FALSE
)