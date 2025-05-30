import time
import pandas as pd
import random
from concurrent.futures.thread import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from .data_utils import tran_iupac2smiles_fun, tran_iso2can_rdkit, restful_pub_finder, \
    SMILES_BASE_FINDER, tran_iupac2can_smiles_cir, restful_pub_name_finder

def multithreader(func):
    """
    A decorator for multithreaded execution of class methods.
    Splits the input dataframe into chunks and processes them in parallel.
    
    Args:
        func: The method to be executed in parallel
        
    Returns:
        Decorated function that runs in parallel threads
    """
    def decorator(self, *args, **kwargs):
        data_list = []
        
        # Adjust number of workers if more than data size
        if self.max_workers >= len(self.data):
            self.max_workers = len(self.data)

        each_num = len(self.data) // self.max_workers

        print(f"Starting multithreaded execution:\r\n"
              f"Data length: {len(self.data)}\r\n"
              f"Function: {func.__name__}\r\n"
              f"Workers: {self.max_workers}\r\n"
              f"Items per worker: {each_num}\r\n")

        # Split data into chunks for each worker
        for i in range(self.max_workers):
            data_list.append(self.data.iloc[i * each_num: (i + 1) * each_num])

        # Handle remainder data (not evenly divisible by worker count)
        remainder = len(self.data) % self.max_workers
        if remainder > 0:
            print(f"Remainder chunk size: {remainder}")
            data_list.append(self.data.iloc[-remainder:])

        print(f"Total chunks: {len(data_list)}\r\n")

        # Process data in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers + 1) as pool:
            futures = []

            for data in data_list:
                # Add small random delay to prevent API rate limiting
                random_time_rest = random.randint(1, 2)
                time.sleep(random_time_rest)
                futures.append(pool.submit(func, self, data, *args, **kwargs))

            # Collect results from all threads
            for future in futures:
                try:
                    result = future.result()
                    print(f"Task result: {result}")
                except Exception as e:
                    print(f"An error occurred: {e}")

    return decorator


class Data_reader:
    """
    Base class for reading and processing various chemical databases.
    
    Provides functionality for:
    - Reading data from multiple sources
    - Data preprocessing and cleaning
    - SMILES conversion and handling
    - Data splitting and filtering
    """

    def __init__(self, path_list, target_colnames, max_workers=32, fun=None):
        """
        Initialize the Data_reader class.
        
        Args:
            path_list: List of paths to data files
            target_colnames: Columns to read from data files
            max_workers: Maximum number of worker threads for parallel processing
            fun: Custom function to read data files (must return a pandas DataFrame)
        """
        self.target_colnames = target_colnames
        self.path_list = path_list
        self.read_fun = fun
        self.max_workers = max_workers
        self.data = None
        self.data_list = []

        # Initialize data loading pipeline
        self._read_data_list()
        self._aggregate()
        self._preprocess()
    
    def _read_data_list(self):
        """
        Read data from all files in path_list using the provided read function.
        Stores results in self.data_list.
        """
        try:
            if hasattr(self.path_list, '__len__') and len(self.path_list) >= 1:
                self.data_list = [self.read_fun(path, self.target_colnames) for path in self.path_list]
        except IOError:
            print('Data file error: path is not correct:\r\n', str(self.path_list))
        else:
            print(f'Data reading complete. Files processed: {len(self.path_list)}')

    def _aggregate(self):
        """
        Combine multiple dataframes from data_list into a single dataframe.
        Verifies that no data is lost during aggregation.
        """
        if len(self.data_list) >= 1:
            self.data = pd.concat(self.data_list, axis=0, ignore_index=True)
            # Verify row count is preserved
            expected_rows = sum([len(data) for data in self.data_list])
            assert expected_rows == self.data.shape[0], f"Row count mismatch: {expected_rows} vs {self.data.shape[0]}"

    def _preprocess(self):
        """
        Perform initial data cleaning:
        - Remove duplicate entries
        - Remove rows with missing values
        - Reset index
        """
        # Remove duplicates and missing values
        self.data.drop_duplicates(None, keep='first', inplace=True, ignore_index=True)
        self.data.dropna(axis=0, how='any', inplace=True)
        self.data = self.data.reset_index(drop=True)
        
        print("Preprocessing complete. Data summary:")
        self.data.info()

    def iupac2smiles(self, col_name='smiles', supply_name='Molecule Name'):
        """
        Convert IUPAC names to SMILES notation.
        
        Args:
            col_name: Column name to store SMILES strings
            supply_name: Column name containing IUPAC names
        """
        self.data[col_name] = self.data[supply_name].apply(tran_iupac2smiles_fun)

    def iso2can_smiles_offline(self, col_name):
        """
        Convert isomeric SMILES to canonical SMILES using RDKit (offline).
        
        Args:
            col_name: Column name containing SMILES strings to convert
        """
        self.data[col_name] = self.data[col_name].apply(tran_iso2can_rdkit)

    def iso2can_smiles_cir(self, col_name='smiles'):
        """
        Convert isomeric SMILES to canonical SMILES using Chemical Identifier Resolver (online).
        
        Args:
            col_name: Column name containing SMILES strings to convert
        """
        self.data[col_name] = self.data[col_name].apply(tran_iupac2can_smiles_cir)

    @multithreader
    def supply_smiles(self, target_data, col_name='smiles', supply_name='Molecule Name', transformer=None):
        """
        Fill missing SMILES by querying based on molecule names or other identifiers.
        Uses multithreading for faster processing.
        
        Args:
            target_data: DataFrame chunk to process
            col_name: Column name for SMILES data
            supply_name: Column name with alternative identifier
            transformer: Custom function for SMILES lookup
        
        Returns:
            Number of SMILES entries added
        """
        print("Running SMILES lookup for missing entries")

        # Set transformer function - default uses molecule name lookup
        if transformer:
            func = lambda x: transformer(x[supply_name])
        else:
            func = lambda x: restful_pub_name_finder(x[supply_name])

        # add more smiles to fill the empty
        target_data.loc[target_data[col_name].isna(), col_name] = target_data.loc[target_data[col_name].isna()].apply(
            func, axis=1)

    def selected_proprties(self, selected):
        """
        Filter data to keep only rows matching specified criteria.
        
        Args:
            selected: Dictionary of {column_name: value} pairs to filter by
        """
        for key, value in selected.items():
            self.data = self.data[self.data[key] == value]
            
        # Reset index after filtering
        self.data = self.data.reset_index(drop=True)
        print(f"Data filtered by {list(selected.keys())}. Remaining rows: {len(self.data)}")
        
    def print_uniques(self):
        """
        Print unique values for each column in the dataset.
        Useful for exploring categorical variables.
        """
        for col in self.data.columns:
            unique_vals = self.data[col].unique()
            print(f'Column: {col}\r\n'
                  f'Unique values: {len(unique_vals)}\r\n'
                  f'Values: {unique_vals}\r\n')

    def random_split(self, count):
        """
        Randomly split the dataset into test and validation sets.
        
        Args:
            count: Tuple (test_size, validation_size)
            
        Returns:
            Tuple (test_dataframe, validation_dataframe)
        """
        assert len(count) == 2, "Count must be a tuple of (test_size, validation_size)"
        
        # Sample rows for test set
        random_numbers_test = random.sample(range(len(self.data)), count[0])
        random_data_test = self.data.loc[random_numbers_test]
        
        # Remove test rows from main dataset
        self.data = self.data.drop(index=random_numbers_test)
        self.data.reset_index(drop=True, inplace=True)
        print(f"Test set size: {len(random_data_test)}, Remaining data size: {len(self.data)}")

        # Sample rows for validation set
        random_numbers_valid = random.sample(range(len(self.data)), count[1])
        random_data_valid = self.data.loc[random_numbers_valid]
        print(f"Validation set size: {len(random_data_valid)}")

        return random_data_test, random_data_valid

    @multithreader
    def can2iso_smiles_pub(self, target_data, col_name='smiles', supply_name='Molecule Name', transformer=None):
        """
        Convert canonical SMILES to isomeric SMILES using PubChem.
        Uses multithreading for faster processing.
        
        Args:
            target_data: DataFrame chunk to process
            col_name: Column name containing SMILES strings
            supply_name: Additional column for identification (not used by default)
            transformer: Custom function for SMILES conversion
        
        Returns:
            Number of SMILES strings processed
        """
        print("Converting canonical to isomeric SMILES")
        
        # Verify required columns exist
        try:
            assert col_name in target_data.columns
            if transformer is None:  # Only check supply_name if using default transformer
                assert supply_name in target_data.columns
        except AssertionError:
            print(f"Error: Column '{col_name}' or '{supply_name}' not found in data")
            return 0

        # Set default transformer to use PubChem
        transformer = transformer if transformer else lambda x: restful_pub_finder(x, SMILES_BASE_FINDER)

        # Apply transformation
        initial_count = len(target_data)
        target_data.loc[:, col_name] = target_data.loc[:, col_name].apply(transformer)
        
        print(f"Processed {initial_count} SMILES strings")
        return initial_count

    def isomer_finder(self, group_name_list, index, save_index=False):
        """
        Find repeated SMILES with different identifiers (potential isomers).
        Sets SMILES to None for duplicate identifiers.
        
        Args:
            group_name_list: List of columns to group by (first is assumed to be SMILES)
            index: Column containing unique identifiers 
            save_index: Whether to save duplicate indices to CSV
            
        Returns:
            Series containing duplicate index values
        """
        # Group by SMILES (and other columns) and count unique identifiers
        grouped = self.data.groupby(group_name_list)[index]
        smiles_column_name = group_name_list[0]
        
        # Print groups with duplicate identifiers
        duplicate_count = 0
        for group, data in grouped:
            if data.nunique() > 1:
                duplicate_count += 1
                print(f"Group: {group}")
                print(data)

        # Find all groups with duplicate identifiers
        repeated_smiles = grouped.filter(lambda x: x.nunique() > 1)
        
        if duplicate_count > 0:
            print(f"Found {duplicate_count} groups with duplicate identifiers")
            
            # Save to CSV if requested
            if save_index:
                repeated_smiles.to_csv('duplicate_identifiers.csv', index=False)

            # Set SMILES to None for rows with duplicate identifiers
            for idx in repeated_smiles:
                if idx in self.data[index].values:
                    self.data.loc[self.data[index] == idx, smiles_column_name] = None
                    
            print(f"Set SMILES to None for {len(repeated_smiles)} entries with duplicate identifiers")

        return repeated_smiles
    
    def select_adduct_fre(self, threshold=0.01):
        """
        Filter data based on adduct frequency, keeping only common adduct types.
        
        Args:
            threshold: Minimum frequency (0-1) to retain an adduct type
        """
        # Calculate normalized frequency for each adduct type
        adduct_freq = self.data["Adduct"].value_counts(normalize=True)
        
        # Convert to percentage strings for display
        adduct_freq_percent = (adduct_freq * 100).round(2).astype(str) + '%'
        
        # Print all adduct frequencies
        print("Adduct frequency distribution (%):")
        print(adduct_freq_percent.to_string())  
        
        # Plot original distribution
        self._plot_adduct_frequency(threshold, bar_hat=False, name='adduct_distribution_original')
        
        # Get adduct types that meet threshold
        valid_adducts = adduct_freq[adduct_freq >= threshold].index
        
        # Print low-frequency adducts that will be filtered out
        low_freq_adducts = adduct_freq[adduct_freq < threshold]
        low_freq_percent = (low_freq_adducts * 100).round(2).astype(str) + '%'
        print(f"\nAdducts with frequency < {threshold*100}%: \n{low_freq_percent.to_string()}")
        
        # Filter dataset to keep only common adduct types
        self.data = self.data[self.data["Adduct"].isin(valid_adducts)]
        self.data = self.data.reset_index(drop=True)
        
        # Plot filtered distribution
        self._plot_adduct_frequency(threshold, name='adduct_distribution_filtered')

        # Print retained adduct types
        valid_freq_percent = (adduct_freq[valid_adducts] * 100).round(2).astype(str) + '%'
        print(f"\nRetained adduct types ({len(valid_adducts)} total): \n{valid_freq_percent.to_string()}")

    def _plot_adduct_frequency(self, threshold=0.01, figsize=(12, 5), bar_hat=True, name='adduct_distribution'):
        """
        Plot a bar chart of adduct frequency distribution.
        
        Args:
            threshold: Frequency threshold to display as a horizontal line
            figsize: Figure size (width, height)
            bar_hat: Whether to display percentage values above bars
            name: Filename for saving the plot
        """
        # Calculate frequencies
        adduct_freq = self.data["Adduct"].value_counts(normalize=True)
        adduct_freq_percent = adduct_freq * 100  # Convert to percentage
        
        # Create gradient colormap from blue to red
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "blue_to_red", ["#5391f5", "#ff1653"]
        )
        
        # Assign gradient colors to bars
        colors = cmap(np.linspace(0, 1, len(adduct_freq)))

        # Create figure and axes
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        # Plot bars with gradient colors
        bars = plt.bar(
            adduct_freq_percent.index.astype(str),  # X-axis: adduct types
            adduct_freq_percent.values,             # Y-axis: frequency percentages
            color=colors,                           # Gradient colors
            edgecolor='black',                      # Bar border color
            width=0.4,
            alpha=0.8                               # Transparency
        )
        
        # Remove top and right border
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add threshold line
        plt.axhline(
            y=threshold * 100,
            color='red', 
            linestyle='--', 
            linewidth=1,
            label=f'Threshold ({threshold*100:.1f}%)'
        )
        
        # Add percentage labels above bars if requested
        if bar_hat:
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,  # X position (centered)
                    height + 0.3,                       # Y position (above bar)
                    f'{height:.1f}%',                   # Percentage label
                    ha='center',                        # Horizontal alignment
                    va='bottom',                        # Vertical alignment
                    size=16
                )

        # Style the plot
        plt.xticks(rotation=50, ha='right', fontsize=12)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=18)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"./{name}", dpi=500)
        plt.close()


class Data_reader_METLIN(Data_reader):
    """
    Example :

    using the Data_reader_METLIN:

    meltin_tester = Data_reader_METLIN(MELTIN_PATH,
                                        fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))
    meltin_tester.isomer_finder(
        ['smiles', 'Adduct'],
        'Molecule Name',
        'Molecule Name')

    select specific data
    meltin_tester.selected_proprties({'Adduct': 1,
                                      'Dimer.1': 'Monomer'})
    write data as csv
    meltin_tester.data.to_csv(TEMP_SAVE + 'METLIN.CSV', index=False)

    Args :

    path_list : list each element is a string to describe a path
    target_colnames : support for pandas dataframe read specific cols default is  ['Molecule Name','CCS_AVG','Adduct','Dimer.1','inchi','smiles']
    fun : read funs must return a dataframe or using the pandas api

    """

    def __init__(self, path_list, target_colnames=None, max_workers=2, fun=None):
        """
        Initialize METLIN dataset reader with default column selection.
        
        Args:
            path_list: List of paths to METLIN data files
            target_colnames: Columns to read (uses defaults if None)
            max_workers: Maximum number of parallel workers
            fun: Custom function to read data files
        """
        target_colnames = target_colnames if target_colnames else [
            'Molecule Name', 'CCS_AVG', 'Adduct', 'Dimer.1', 'inchi', 'smiles', 'm/z'
        ]
        super().__init__(path_list, target_colnames, max_workers, fun)


class Data_reader_ALLCCS(Data_reader):
    """
    Specialized reader for ALLCCS dataset.
    
    Example usage:
        allccs_tester = Data_reader_ALLCCS(ALLCCS_PATH,
                                          fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))
        allccs_tester.selected_proprties({'Type': 'Experimental CCS'})
        allccs_tester.data = allccs_tester.data[allccs_tester.data['Confidence level'] != 'Conflict']
        allccs_tester.isomer_finder(['Structure', 'Adduct'], 'AllCCS ID')
        allccs_tester.data.to_csv('ALLCCS.csv', index=False)
    """

    def __init__(self, path_list, target_colnames=None, max_workers=32, fun=None):
        """
        Initialize ALLCCS dataset reader with default column selection.
        
        Args:
            path_list: List of paths to ALLCCS data files
            target_colnames: Columns to read (uses defaults if None)
            max_workers: Maximum number of parallel workers
            fun: Custom function to read data files
        """
        target_colnames = target_colnames if target_colnames else [
            'AllCCS ID', 'Name', 'Formula', 'Type', 'Adduct',
            'm/z', 'CCS', 'Confidence level', 'Structure'
        ]
        super().__init__(path_list, target_colnames, max_workers, fun)
