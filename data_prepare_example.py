from data_prepare import ALLCCS_PATH, FINAL_DATA_PATH
from data_prepare.data_module import Data_reader_ALLCCS
import pandas as pd

from data_prepare.data_utils import restful_pub_inchi_finder, restful_pub_name_finder, tran_iupac2smiles_fun



# # Example for processing the entire dataset:
# # Preprocessing, including dataset reading, preprocessing, and saving
# allccs_tester = Data_reader_ALLCCS(ALLCCS_PATH,
#                                     fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))

# # Select only 'Experimental CCS' type data
# allccs_tester.selected_proprties({'Type': 'Experimental CCS'})
# # You can also select specific ion types, e.g.: allccs_tester.selected_proprties({'Type': 'Experimental CCS','Addcut': '[M+H]+'})

# # Post-processing: convert missing data by querying SMILES from compound names or InChI
# # Here we use iupac2smiles function to convert IUPAC names to SMILES, col_name is the SMILES column name, supply_name is the IUPAC column name
# allccs_tester.iupac2smiles(col_name='smiles', supply_name='Name')

# # ALLCCS specific processing - filter by confidence level
# allccs_tester.data = allccs_tester.data[allccs_tester.data['Confidence level'] != 'Conflict']

# # Save processed ALLCCS data
# allccs_tester.data.to_csv(FINAL_DATA_PATH + 'ALLCCS.csv', index=False)


# # Randomly split the dataset
# # test, valid = (test_size, valid_size)
# count = (int(len(allccs_tester.data)*0.2),int(len(allccs_tester.data)*0.1))
# random_data_test,random_data_valid = allccs_tester.random_split(count)
# random_data_test.to_csv(FINAL_DATA_PATH + 'ALLCCS_test.csv', index=False)
# random_data_valid.to_csv(FINAL_DATA_PATH + 'ALLCCS_valid.csv', index=False)


# Example B: Using utility functions to process individual data:
tran_iupac2smiles_fun('Clomazone')
# Example: Query using InChI through PubChem
restful_pub_inchi_finder("InChI=1S/C16H15ClN2O3/c1-10-3-5-13(11(2)7-10)16(21)22-9-15(20)19-14-6-4-12(17)8-18-14/h3-8H,9H2,1-2H3,(H,18,19,20)")