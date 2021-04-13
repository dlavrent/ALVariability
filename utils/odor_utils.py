"""
Defines utility functions for odor-related matters,
like loading from the DoOR dataset
"""
import pandas as pd
import os

def load_door_data(home_dir='.'):
    '''
    Loads DoOR data from the tables generated by 
    ./datasets/DoOR/get_door_data.R
    Returns pandas dataframes:
        df_door_mappings: index is receptor name, 
                          columns include DoOR code, ORN, glomerulus, etc.
        df_door_odor: index is InChiKey (chemical key) of an odor, 
                      columns include name, physical properties
        df_door_response_matrix: index is odorant InChIkey,
                                 columns are receptor responses
        df_odor_response_matrix: index is odorant common name, 
                                 columns are receptor responses
    '''
    # table relating receptors to glomeruli / ORNs / DoOR codes
    datasets_dir = os.path.join(home_dir, 'datasets/DoOR')
    df_door_mappings = pd.read_csv(os.path.join(datasets_dir, 'DoOR_mappings.csv'), index_col=0)
    # table relating odors to their chemical key (InChiKey) + their physical properties
    df_door_odor =  pd.read_csv(os.path.join(datasets_dir, 'DoOR_odor.csv'), index_col=2)
    # table relating odorants to glomerular activations
    df_door_response_matrix =  pd.read_csv(os.path.join(datasets_dir, 'DoOR_SFR_response_matrix.csv'), index_col=0)
    df_door_response_matrix.index.name = 'InChIKey'

    # get mappings of InChIKey to common name
    door_key_to_odor_name = df_door_odor['Name']

    # replace InChiKey with common odor name
    df_odor_response_matrix = df_door_response_matrix.\
                                join(door_key_to_odor_name).\
                                set_index('Name')
    df_odor_response_matrix.index.name = 'odor'

    return df_door_mappings, df_door_odor, df_door_response_matrix, df_odor_response_matrix