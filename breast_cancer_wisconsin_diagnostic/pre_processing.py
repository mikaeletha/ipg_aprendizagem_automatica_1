import pandas
import matplotlib.pyplot as plt
# import seaborn

def remove_duplicate_rows(df):
    # Count the number of rows before removing duplicates
    initial_count = len(df)

    # Remove duplicates
    df_cleaned = df.drop_duplicates()

    # Count the number of rows after removing duplicates
    final_count = len(df_cleaned)

    # Calculate the number of duplicates removed
    duplicates_removed = initial_count - final_count

    # Print the number of duplicates removed
    if duplicates_removed > 0:
        print({duplicates_removed})
    else:
        print("No duplicate samples were found.")

    # print(f"Dataset contains {final_count} Samples")

    return df_cleaned

column_names = ['ID', 'Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']

dataset = pandas.read_csv("./data/wdbc.data", header=None, names = column_names)

dataset = remove_duplicate_rows(dataset)

print(dataset.describe(include="all"))

num_variables = len(dataset.columns)
num_output_variables = 1
number_input_variables = num_variables - num_output_variables

input_columns = dataset.columns[0:number_input_variables]

dataset.to_csv("pre_processed/wdbc.csv", index=False)
