Project README
Overview
This project involves data processing and analysis of production phases from various Excel files. It focuses on merging and analyzing data related to pre-inoculum, inoculum, and final cultivation phases, along with centrifuge and bioreactor data. The goal is to compute average measurements and prepare the data for further analysis.


Key Steps
Loading Excel Files:

The script reads all Excel files in the datasets directory.
Each sheet within the files is loaded into a dictionary (dfs) with unique keys based on the filename and sheet name.
Data Merging:

The pre-inoculum, inoculum, and final cultivation DataFrames are accessed and merged on the LOTE column.
Centrifuge Data Processing:

A function (load_centrifuga_data) retrieves centrifuge data based on its ID.
Another function (calculate_mean_for_lote) calculates the mean for a specific LOTE within a given date range and centrifuge ID.
The mean values are then added as a new column (media_PV) to the merged DataFrame.
Bioreactor Data Processing:

Similar to centrifuge data, functions are defined to load bioreactor data and calculate means for temperature, pH, and dissolved oxygen (DO) based on specified dates.
These mean values are added as new columns (media_temp_bioreactor, media_ph_biorreactor, media_PO_biorreactor) to the merged DataFrame.
Finalizing the Data:

Any remaining NaN values are filled with "NA".
The final DataFrame is saved to a CSV file (dataset.csv).

Project Structure (model_analysis)
Data Loading:

Loads training and test datasets from CSV files.
Data Cleaning:

Identifies and fills missing values in the dataset.
Converts specific columns to the appropriate numeric types, handling potential formatting issues.
Feature and Target Definition:

Defines features (X) and the target variable (y) for both training and test datasets.
Data Splitting:

Splits the training data into training, validation, and test sets (60% training, 20% validation, 20% testing).
Model Training:

Test diferent model for validation and test sets the best of all seems to be RandomForestRegressor.
Final Predictions:

Generates predictions for the test dataset using the best-performing model from the RandomForestRegressor.
