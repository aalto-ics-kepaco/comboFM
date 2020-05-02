library("rJava")
library("rcdk")
library("fingerprint")

# Read SMILES from text file 
smiles <-
  readr::read_csv(
    file = "drugs__SMILES.csv",
    comment = "",
    col_names = FALSE,
    col_types = readr::cols()
  )

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

readr::write_csv(
  x = fps_vectors,
  path = "drugs__estate_fingerprints.csv",
  col_names = FALSE
)