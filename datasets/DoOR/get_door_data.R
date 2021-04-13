rm(list=ls())

# install DoOR if not already present
library(devtools)
install_github('ropensci/DoOr.functions')
library(DoOR.functions)

# load data
load_door_data()

# save dataset info
write.csv(door_dataset_info, 'DoOR_datasets_info.csv')

# save raw OR responses
raw_or_response_tables_dir <- 'raw_or_response_tables/'
dir.create(raw_or_response_tables_dir)
or_names <- names(door_response_matrix)
for (i in 1:length(or_names)){
  tab <- get(or_names[i])
  tab_fname <- paste(raw_or_response_tables_dir, or_names[i], '.csv', sep='')
  write.csv(tab, tab_fname)
}

# save table relating mapping from
# columns:
# receptor, sensillum, OSN, glomerulus, etc.
write.csv(door_mappings, 'DoOR_mappings.csv', row.names=FALSE)

# save table relating odors to their DoOR InChIKey code
write.csv(odor, 'DoOR_odor.csv', row.names=FALSE)

# get table relating odors to glomerular responses
# get response matrix NOT adjusted for spontaneous firing rate
default_response_matrix <- door_default_values('door_response_matrix')
write.csv(default_response_matrix, 'DoOR_response_matrix.csv', row.names=TRUE)

# adjust for spontaneuous firing rate
sfr <- door_default_values('zero')
sfr_adjusted_response_matrix <- reset_sfr(default_response_matrix, sfr)
# save table
write.csv(sfr_adjusted_response_matrix, 'DoOR_SFR_response_matrix.csv', row.names=TRUE)

