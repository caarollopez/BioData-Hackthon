
import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from openpyxl import load_workbook
def open_file(file_name, datasets_path, dfs):
    file_path = os.path.join(datasets_path, file_name)

    # Specify the engine to avoid format detection issues
    # load all sheets with pandas.read_excel(sheet_name=None)
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

    # Save each sheet in `dfs` with a unique name combining the file and sheet name
    for sheet_name, data in all_sheets.items():
        df_name = f"{file_name.replace('.xlsx', '')}_{sheet_name}"
        dfs[df_name] = data
def preprocess_dataset(dataset, columns, display_info=True, check_missing=True, check_duplicates=True, check_outliers=True, show_correlation=True):
    """
    This function processes the given dataset by renaming the columns and displaying relevant information.
    
    Parameters:
        dataset (pd.DataFrame): The dataset to process.
        columns (list): The list of column names to rename the dataset columns to.
        display_info (bool): Whether to display basic info about the dataset (default is True).
        check_missing (bool): Whether to check for missing values (default is True).
        check_duplicates (bool): Whether to check for duplicate rows (default is True).
        check_outliers (bool): Whether to check for outliers in numeric columns (default is True).
        show_correlation (bool): Whether to display correlation for numeric columns (default is True).
        
    Returns:
        pd.DataFrame: The processed dataset with renamed columns.
    """
    
    dataset.columns = columns
    
    # Display basic information
    if display_info:
        print("------ Dataset Info ------")
        print(f"Shape: {dataset.shape}")
        print(f"Columns: {dataset.columns.tolist()}")
        print(f"First few rows:\n{dataset.head()}")
        print(f"Summary statistics:\n{dataset.describe()}")
        print("--------------------------")
    
    # Check for missing values and impute
    if check_missing:
        missing_values = dataset.isnull().sum()
        print("\n------ Missing Values ------")
        print(missing_values[missing_values > 0])  # Only show columns with missing values
        print("--------------------------")
        
        # Impute missing values
        for col in dataset.columns:
            if dataset[col].dtype in ['float64', 'int64']:  # For numerical columns
                mean_value = dataset[col].mean()
                dataset[col].fillna(mean_value, inplace=True)
                print(f"Imputed missing values in '{col}' with mean value: {mean_value}")
            elif dataset[col].dtype == 'object':  # For categorical columns
                mode_value = dataset[col].mode()[0]
                dataset[col].fillna(mode_value, inplace=True)
                print(f"Imputed missing values in '{col}' with mode value: {mode_value}")
        print("\n------ Missing Values After Imputation ------")
        print(dataset.isnull().sum())
        print("--------------------------")
    
    # Check for duplicate rows
    if check_duplicates:
        duplicate_rows = dataset.duplicated().sum()
        print(f"\n------ Duplicate Rows ------")
        print(f"Duplicate Rows: {duplicate_rows}")
        print("--------------------------")
    
    # Check for outliers in numeric columns
    if check_outliers:
        numeric_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            q1 = dataset[col].quantile(0.25)
            q3 = dataset[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = dataset[(dataset[col] < lower_bound) | (dataset[col] > upper_bound)]
            print(f"\n------ Outliers in '{col}' ------")
            print(f"Outliers: {outliers.shape[0]} rows")
            if outliers.shape[0] > 0:
                print(outliers[[col]].head())
            print("--------------------------")
    
    # Show correlation matrix (only for numeric columns)
    if show_correlation:
        numeric_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            correlation_matrix = dataset[numeric_cols].corr()
            print("\n------ Correlation Matrix ------")
            print(correlation_matrix)
            print("--------------------------")
    
    return dataset
def load_centrifuga_data(id_centrifuga, dfs):
    file_name = f'Centrífuga {id_centrifuga}_Datos'
    try:
        centrifuga_df = dfs[file_name]
        return centrifuga_df
    except KeyError:
        # Si el archivo no existe, retornar un DataFrame vacío o NaN
        return pd.DataFrame()  
def calculate_mean_for_lote(lote, fecha_inicio, fecha_fin, id_centrifuga, dfs):
    centrifuga_df = load_centrifuga_data(id_centrifuga, dfs)
    if centrifuga_df.empty:
        return np.nan  # Si no hay datos, devolver NaN
    mask = (pd.to_datetime(centrifuga_df['DateTime']) >= pd.to_datetime(fecha_inicio)) & (pd.to_datetime(centrifuga_df['DateTime']) <= pd.to_datetime(fecha_fin))
    filtered_data = centrifuga_df[mask]
    return filtered_data[f'{id_centrifuga}_D01916047.PV'].mean()
def load_bioreactor_data(id_bioreactor, dfs):
    file_name = f'Biorreactor {id_bioreactor}_Datos'
    try:
        # Attempt to load the DataFrame of the bioreactor
        bioreactor_df = dfs[file_name]
        return bioreactor_df
    except KeyError:
        # If the file does not exist, return an empty DataFrame
        return pd.DataFrame()  # Alternatively, return np.nan if preferred
# Function to calculate the mean for the specified lot and bioreactor
def calculate_mean_for_lote2(lote, fecha_inicio, fecha_fin, id_bioreactor, dfs):
    bioreactor_df = load_bioreactor_data(id_bioreactor, dfs)
    
    if bioreactor_df.empty:
        return np.nan, np.nan, np.nan  # Return NaN for all means if there's no data

    # Filter the data based on the specified date range
    mask = (pd.to_datetime(bioreactor_df['DateTime']) >= pd.to_datetime(fecha_inicio)) & \
           (pd.to_datetime(bioreactor_df['DateTime']) <= pd.to_datetime(fecha_fin))
    filtered_data = bioreactor_df[mask]
    
    # Calculate means for the specified columns
    mean_temp = filtered_data[f'{id_bioreactor}_FERM0101.Temperatura_PV'].mean() if not filtered_data.empty else np.nan
    mean_ph = filtered_data[f'{id_bioreactor}_FERM0101.Single_Use_pH_PV'].mean() if not filtered_data.empty else np.nan
    mean_do = filtered_data[f'{id_bioreactor}_FERM0101.Single_Use_DO_PV'].mean() if not filtered_data.empty else np.nan
    
    return mean_temp, mean_ph, mean_do
def clean_and_convert(column):
    # Reemplazar ',' por '.' y eliminar el signo '+'
    column = column.str.replace(',', '.', regex=False)
    column = column.str.replace('+', '', regex=False)
    # Convertir a float, forzando errores a NaN
    return pd.to_numeric(column, errors='coerce')
