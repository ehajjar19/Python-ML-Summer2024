import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import pickle
from pycirclize import Circos
from pathlib import Path

def plot_loss(mlp):
    """
    Plotting the training loss curve from mlp object
    """
    plt.plot(mlp.loss_curve_)
    plt.title('Training Loss Curve')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.savefig('mlp_training_loss_curve1.png')


def plot_all_metrics(infile, outfile='metrics_plots.png'):
    """
    Read a metrics CSV then plot the barchart of r2, pearson_corr, p-value, mse,
    in a single PNG file with 4 subplots.
    """
    # Read the CSV data into a DataFrame
    df = pd.read_csv(infile)

    # Set up the matplotlib figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Plot each metric on a separate subplot
    metrics = ['r2_score', 'pearson_correlation', 'p-value', 'mse']
    for ax, metric in zip(axs.flat, metrics):
        ax.bar(df['brain_region'], df[metric])
        ax.set_title(metric.replace('_', ' ').title())
        # ax.set_xlabel('Brain Region')
        ax.set_ylabel(metric.title())
        ax.tick_params(labelrotation=45)

    # Save the figure to a single PNG file
    plt.savefig(outfile)

def plot_scatter(predicted_values, actual_values):
    """
    Generate a scatter plot of predicted vs. actual values
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_values, predicted_values, color='blue', label='Actual vs Predicted')

    # Plotting the line y=x
    min_value = min(min(actual_values), min(predicted_values))
    max_value = max(max(actual_values), max(predicted_values))
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label='y=x')

    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file named scatter.png
    plt.savefig('scatter.png', bbox_inches='tight')


def scatter_corr(x, y, corr):
    '''
    Create a simple scatterplot with correlation annotation.
    '''
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.scatter(x, y, color='blue')  # Scatter plot of x vs y
    plt.title('Scatter Plot of x vs. y')
    plt.xlabel('x')
    plt.ylabel('y')

    # Annotate the plot with the Pearson correlation
    plt.annotate(f'Pearson Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))

    plt.grid(True)  # Add grid for better readability
    plt.savefig("scatter.png")


def read_coef(infile):
    '''Read the 180*180 coefficients of a ...Model_details.json file'''
    with open(infile, 'r') as f:
        data = json.load(f)
        return data['coef']


def plot_corr_his(infile, outfile='corr_his.png'):
    '''
    Plot the histogram of all the 180*180 coefficients of a Model_details.json file.
    '''
    correlations = read_coef(infile)

    plt.figure(figsize=(8, 6))
    plt.hist(correlations, bins=50, color='blue', edgecolor='black')  # You can adjust the number of bins and color here
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 70)
    plt.savefig(outfile, bbox_inches='tight')


def plot_sorted_line(infile, outfile='sorted_line.png'):
    '''
    Sorted line plot of all the 180*180 coefficients of a Model_details.json file.
    '''
    data = read_coef(infile)
    data = np.abs(data)
    sorted_data = np.sort(data)

    # Create the plot
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(sorted_data, marker='o', linestyle='-', color='b')  # Line plot with markers
    plt.title('Sorted Line Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.xlim(14000, 16110)

    plt.grid(True)  # Add grid for better readability
    plt.savefig(outfile, bbox_inches='tight')


def get_column_name(column_file="updated_180.csv", area=5, mapped=True):
    """
    Get the column names of fMRI, in either 5/22/180 categories.
    Mapped defaults to True, which will return the string of col names.
    """
    df = pd.read_csv(column_file, sep=',', header=None) # TODO this is redundant when area == 180, consider remove.
    if area == 5:
        out_col = df.iloc[:, 1].tolist()
        if mapped:       
            mapping_5 = {1: 'Early and intermediate visual cortex', 2: 'Sensorimotor', 3: 'Auditory', 
                        4: 'Rest of posterior cortex', 5: 'Rest of anterior cortex'}
            out_col = [mapping_5[i] for i in out_col]
    elif area == 180:
        df = pd.read_csv("HCPMMP1_on_MNI152_ICBM2009a_nlin.txt", sep=' ', header=None)
        #TODO right now it only return the mapped.
        out_col = df.iloc[:, 1].tolist()
    else:
        out_col = df.iloc[:, 0].tolist()
        if mapped:
            mapping_22 = {
                            1: 'Primary_Visual',
                            2: 'Early_Visual',
                            3: 'Dorsal_Stream_Visual',
                            4: 'Ventral_Stream_Visual',
                            5: 'MT+_Complex_and_Neighboring_Visual_Areas',
                            6: 'Somatosensory_and_Motor',
                            7: 'Paracentral_Lobular_and_Mid_Cingulate',
                            8: 'Premotor',
                            9: 'Posterior_Opercular',
                            10: 'Early_Auditory',
                            11: 'Auditory_Association',
                            12: 'Insular_and_Frontal_Opercular',
                            13: 'Medial_Temporal',
                            14: 'Lateral_Temporal',
                            15: 'Temporo-Parieto-Occipital_Junction',
                            16: 'Superior_Parietal',
                            17: 'Inferior_Parietal',
                            18: 'Posterior_Cingulate',
                            19: 'Anterior_Cingulate_and_Medial_Prefrontal',
                            20: 'Orbital_and_Polar_Frontal',
                            21: 'Inferior_Frontal',
                            22: 'Dorsolateral_Prefrontal'
                        }
            out_col = [mapping_22[i] for i in out_col]
    return out_col


def get_DTI_col_name(group=False):
    """
    Get the DTI column names.
    """
    items = [
        "global_mori",
        "GenuCorpus","BodyCorpus","SplnCorpus","Tapetum",
        "PontineCros","InfCblmPed","MidCblmPed","SupCblmPed","CerebralPed",
        "CortspnlTrct","MedLemniscus","AntIntCap","PosIntCap","RetroIntCap","AntCoronaRad","SupCoronaRad","PosCoronaRad","PosThalamRad","ExternalCap",
        "Fornix","StriaTerminali","CingAntMid","CingInf",
        "SupLongFasc","SupFrntOccFasc","SagStratum","UncinateFac"
        ]
    #TODO add the case when group = True.
    return items


def get_fMRI_22_name():
    return ['Primary_Visual',
            'Early_Visual',
            'Dorsal_Stream_Visual',
            'Ventral_Stream_Visual',
            'MT+_Complex_and_Neighboring_Visual_Areas',
            'Somatosensory_and_Motor',
            'Paracentral_Lobular_and_Mid_Cingulate',
            'Premotor',
            'Posterior_Opercular',
            'Early_Auditory',
            'Auditory_Association',
            'Insular_and_Frontal_Opercular',
            'Medial_Temporal',
            'Lateral_Temporal',
            'Temporo-Parieto-Occipital_Junction',
            'Superior_Parietal',
            'Inferior_Parietal',
            'Posterior_Cingulate',
            'Anterior_Cingulate_and_Medial_Prefrontal',
            'Orbital_and_Polar_Frontal',
            'Inferior_Frontal',
            'Dorsolateral_Prefrontal']


def plot_correlations(infile_path: Path, outfile='correlations.png'):
    """
    This takes in a "...Model_details.json" file Path, and plot the 180*180 coefficients
    for that specific DTI region. Coefficients were generated during the training of the model,
    and we use that as a metric of whether the correlation of fmri column region * fmri row region
    has a significant impact on the result of the DTI region.

    Note: infile_path needs to be a Path object
    TODO change all path in this util to pathlib.Path?
    """
    correlations = read_coef(infile_path)

    labels = get_column_name(area=180, mapped=True)
    # print(labels)

    # Initialize an empty 180x180 matrix
    matrix_size = 180
    correlation_matrix = np.zeros((matrix_size, matrix_size))
    upper_triangle_indices = np.triu_indices(matrix_size, k=1)
    correlation_matrix[upper_triangle_indices] = correlations
    ret_matrix = correlation_matrix.copy()

    correlation_matrix = correlation_matrix.T
    # np.fill_diagonal(correlation_matrix, 1) #TODO need this?


    #TODO make it optional to plot.
    # Find the minimum and maximum values for better color scaling
    min_val = np.min(correlation_matrix)
    max_val = np.max(correlation_matrix)

    # Plotting only the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(100, 80))
    ax = sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', vmin=min_val, vmax=max_val,
                    xticklabels=labels, yticklabels=labels, square=True, annot=False)
    plt.title('Correlation Matrix: ' + infile_path.name, fontsize=100)
    plt.xticks(rotation=90, fontsize=10)  # Smaller font size for the labels
    plt.yticks(rotation=0, fontsize=10)  # Smaller font size for the labels

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=100)

    plt.savefig(outfile, bbox_inches='tight')
    return ret_matrix


def remove_zero_rows_columns(matrix_df):
    # Identify columns with all zero values
    zero_columns = matrix_df.columns[(matrix_df == 0).all()]
    # Identify rows with all zero values
    zero_rows = matrix_df.index[(matrix_df == 0).all(axis=1)]
    
    # Print out the names of these columns and rows
    print("Columns with all zero values:", zero_columns.tolist())
    print("Rows with all zero values:", zero_rows.tolist())
    
    # Remove the identified rows and columns
    matrix_df = matrix_df.drop(columns=zero_columns)
    matrix_df = matrix_df.drop(index=zero_rows)
    
    # Return the modified DataFrame
    return matrix_df


def test_circlize(input_matrix=None, outfile='circlize_test.png', area = 5):
    '''
    A convenient sanity test when using pyCirclize. 
    Change this function to test  pyCirclizewhen adding new functionalities.
    '''
    row_names = ["a","b","c"]
    col_names = ["a","b","c"]

    matrix_data = [
        [10, 50, 7],
        [50, 9, 10],
        [17, 13, 7],
    ]
    matrix_df = pd.DataFrame(matrix_data, index=row_names, columns=col_names)

    # Initialize Circos from matrix for plotting Chord Diagram
    circos = Circos.initialize_from_matrix(
        matrix_df,
        space=3,
        cmap="tab10",
        label_kws=dict(size=12),
        link_kws=dict(ec="black", lw=0.5), #direction = 1
    )

    circos.savefig(outfile)


def sum_columns_rows(matrix_df, names_list):
    """
    Gets the sum of specified col and row from names_list.
    """
    print("names_list", names_list)

    results = []
    for name in names_list:
        col_sum = matrix_df[name].sum() if name in matrix_df.columns else 0
        row_sum = matrix_df.loc[name].sum() if name in matrix_df.index else 0
        total_sum = col_sum + row_sum
        results.append(total_sum)
    return results


def plot_circlize(input_matrix, outfile='circlize.png', group = 5, aggregate=False, threshold=0.01, plot=True):
    '''
    Plot the circle plot of coefficients of the model for that DTI region. 
    The 180 fMRI regions are categorized to either group=5 or 22 (according to the paper).

    The coefficients are taken the absolute value, then filter out all small coefficients (default <0.01)

    If aggregate=True, then the fMRI regions will be aggregated based on their groups first, then generate
    the plot. The resulting plot will be very simple in this case.
    '''
    col_names = get_column_name(area=group)
    row_names = col_names

    input_matrix = np.abs(input_matrix) #TODO Right now using abs value
    input_matrix[input_matrix < threshold] = 0 # TODO should you do this threshold after aggregation?
    matrix_df = pd.DataFrame(input_matrix, index=row_names, columns=col_names)

    if aggregate:
        # Aggregate the columns and rows
        column_aggregated = matrix_df.groupby(by=matrix_df.columns, axis=1).sum()
        matrix_df = column_aggregated.groupby(by=column_aggregated.index).sum()

    print("after aggregation, dim is: ", matrix_df.shape)

    # remove zero rows and columns:
    print("Outfile:", outfile, "\n")
    matrix_df = remove_zero_rows_columns(matrix_df)
    matrix_df.to_csv("temp_matrix_df.csv", index=True, header=True)

    # colordict = dict(
    #     A="red", B="blue", C="green", ...
    #     'Primary_Visual'="red",
    #     'Early_Visual'="red",
    #     'Dorsal_Stream_Visual'="red",
    #     'Ventral_Stream_Visual'="red",
    #     'MT+_Complex_and_Neighboring_Visual_Areas',
    #     'Somatosensory_and_Motor',
    #     'Paracentral_Lobular_and_Mid_Cingulate',
    #     'Premotor',
    #     'Posterior_Opercular',
    #     'Early_Auditory',
    #     'Auditory_Association',
    #     'Insular_and_Frontal_Opercular',
    #     'Medial_Temporal',
    #     'Lateral_Temporal',
    #     'Temporo-Parieto-Occipital_Junction',
    #     'Superior_Parietal',
    #     'Inferior_Parietal',
    #     'Posterior_Cingulate',
    #     'Anterior_Cingulate_and_Medial_Prefrontal',
    #     'Orbital_and_Polar_Frontal',
    #     'Inferior_Frontal',
    #     'Dorsolateral_Prefrontal'
    #     )

    # Initialize Circos from matrix for plotting Chord Diagram

    if plot:
        circos = Circos.initialize_from_matrix(
            matrix_df,
            space=3,
            cmap="tab10",
            label_kws=dict(size=7),
            link_kws=dict(ec="black", lw=0.5), #direction = 1
        )
        circos.savefig(outfile)

    return sum_columns_rows(matrix_df, get_fMRI_22_name()) # TODO change this hard-coded column.


def heatmap_circlize_folder(in_folder = "ncanda_cv_raw", out_folder="heatmap_circlize_22"):
    '''
    Plot the circle plot of coefficients, for all the models and DTI regions in the in_folder. 
    Save the plots to out_folder.
    '''
    in_folder = Path(in_folder)
    for file in in_folder.iterdir():
        file_prefix = file.name[:-13]
        mat = plot_correlations(file, f"{out_folder}/{file_prefix}_heatmap.png")
        # plot_corr_his(file, f"{out_folder}/{file_prefix}_histogram.png")
        plot_circlize(mat, f"{out_folder}/{file_prefix}_circlize_aggregated_thresh001.png", group=22, aggregate=True, threshold=0.01)










#################################################################################################################
# Creating DTI*DTI and DTI*fMRI heatmaps:


def fmri_heatmap(input_matrix, label_col, label_row, col_groups, row_groups, outfile="DTI_fMRI_heatmap.png"):
    """
    Output a heatmap based on input matrix and row,col labels.
    Generates a clutered and non-clustered heatmap at the same time.
    """
    import matplotlib.ticker as ticker

    # Find the minimum and maximum values for better color scaling
    min_val = np.min(input_matrix[input_matrix > 0]) # Finds the minimum value that's > 0
    max_val = np.max(input_matrix)

    input_matrix = input_matrix.T #TODO For not symmetric matrix?

    # Plotting only the upper triangle of the correlation matrix
    # mask = np.triu(np.ones_like(input_matrix, dtype=bool))

    plt.figure(figsize=(100, 80))
    ax = sns.heatmap(input_matrix, cmap='viridis', vmin=min_val, vmax=max_val,
                     xticklabels=label_row, yticklabels=label_col, square=True, annot=False)
    # Note that tick labels are reversed due to previous transpose of input matrix
    plt.title('DTI measurements by key fMRI features', fontsize=100)
    plt.xticks(rotation=90, fontsize=50)  # Rotate column labels for better visibility
    plt.yticks(rotation=0, fontsize=50)  # Row labels

    # Define group boundaries
    # col_groups = [0, 1, 5, 10, 20, 24, 28]  # Start positions of each group + 1 past the end
    # row_groups = [0, 1, 5, 10, 20, 24, 28]  # Start positions of each group + 1 past the end
    # col_groups = [0, 4, 9, 19, 23, 27]
    # row_groups = [0, 4, 9, 19, 23, 27]

    # Set minor tick locations to group boundaries (excludes the very first and last index to avoid out of bounds)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(row_groups[1:-1]))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(col_groups[1:-1]))

    # Enable minor gridlines at these boundaries
    ax.grid(True, which='minor', color='black', linewidth=5, linestyle='-', axis='both')

    # Disable major grid lines to prevent overlapping lines
    ax.grid(False, which='major')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=100)

    plt.savefig(outfile, bbox_inches='tight')
    cluster_heatmap(input_matrix, label_col, label_row, f"Clustered_{outfile}")


#TODO: clustered heatmap: change style
def cluster_heatmap(matrix_df, row_label, col_label, outfile):
    from sklearn.cluster import AgglomerativeClustering

    data = matrix_df  # Directly use the numpy array

    # Perform hierarchical clustering on rows and columns
    cluster_rows = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='ward')
    cluster_cols = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='ward')

    # Fit the model
    row_indices = cluster_rows.fit_predict(data)
    col_indices = cluster_cols.fit_predict(data.T)  # Transpose to cluster columns

    # Reorder rows and columns based on clustering
    ordered_data = data[np.argsort(row_indices)][:, np.argsort(col_indices)]

    # Reorder row and column labels
    ordered_row_labels = np.array(row_label)[np.argsort(row_indices)]
    ordered_col_labels = np.array(col_label)[np.argsort(col_indices)]

    # Plot the reordered heatmap with labels
    plt.figure(figsize=(100, 80))
    ax = sns.heatmap(ordered_data, cmap='viridis', vmin=data.min(), vmax=data.max(),
                    xticklabels=ordered_col_labels, yticklabels=ordered_row_labels,
                    square=True, annot=False)

    plt.title('DTI measurements by key fMRI features', fontsize=100)
    plt.xticks(rotation=90, fontsize=50)  # Rotate column labels for better visibility
    plt.yticks(rotation=0, fontsize=50)  # Row labels
    plt.savefig(outfile, bbox_inches='tight')


def normalize_rows(matrix):
    '''
    Normalizes each row of a given matrix so that the sum of the elements in each row equals 1
    '''
    # Calculate the sum of each row and reshape to enable broadcasting
    row_sums = matrix.sum(axis=1).reshape(-1, 1)

    # Normalize each row by dividing by the row sum
    normalized_matrix = matrix / row_sums

    return normalized_matrix


def output_fMRI_DTI_heatmap(in_folder = "ncanda_cv_raw", out_folder="heatmap_circlize_22"):
    '''
    Creates a fMRI * DTI heatmap.
    We want to know for each DTI region, after training the model, which fMRI region(categorized from 180 to 22) is predicted 
    to be the most correlated with that DTI region.
    for each DTI region, we find the model predicted coefficients(180*180), then categorize and aggregate the 
    180*180 matrix to 22*22 matrix. Then for each category, add the corresponding row and column (22*2), and that
    will be the value of fmri category-DTI region correlation value.

    A heatmap is created for fMRI*DTI region.
    Another normalized(based on DTI region) heatmap is generated as well.
    '''
    items = get_DTI_col_name()
    # print(items)
    result_matrix = []

    for file_prefix in items:
        print(file_prefix)
        mat = plot_correlations(Path(f"{in_folder}/{file_prefix}_ElasticNetModel_details.json"), f"{out_folder}/{file_prefix}_heatmap.png")
        # plot_corr_his(file, f"{out_folder}/{file_prefix}_histogram.png")
        item_list = plot_circlize(mat, f"{out_folder}/{file_prefix}_circlize_aggregated_thresh001.png", group=22, aggregate=True, threshold=0.01, plot=False)
        print("Length of item_list:",len(item_list))
        result_matrix.append(item_list)
    
    result_matrix = np.vstack(result_matrix)

    row_groups = [0, 1, 5, 10, 20, 24, 28]
    col_groups = [4, 8, 13, 17, 21]
    label_col = get_fMRI_22_name()

    fmri_heatmap(result_matrix, label_col, items, col_groups, row_groups)

    # Normalize so that the sum of fMRI region correlation values of a DTI region is 1
    norm_result_matrix = normalize_rows(result_matrix)
    fmri_heatmap(norm_result_matrix, label_col, items, col_groups, row_groups, outfile="DTI_fMRI_norm_heatmap.png")


def output_heatmap(input_matrix, label_col, label_row, col_groups, row_groups, outfile="DTI_22_22_heatmap.png"):
    """
    Output DTI*DTI heatmaps, including "DTI_22_22_heatmap.png" and "Clustered_DTI_22_22_heatmap.png"
    """
    import matplotlib.ticker as ticker

    # Find the minimum and maximum values for better color scaling
    min_val = np.min(input_matrix[input_matrix > 0]) # Finds the minimum value that's > 0
    max_val = np.max(input_matrix)

    input_matrix = input_matrix.T #TODO For not symmetric matrix?

    # Plotting only the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(input_matrix, dtype=bool))

    plt.figure(figsize=(100, 80))
    ax = sns.heatmap(input_matrix, mask=mask, cmap='viridis', vmin=min_val, vmax=max_val,
                     xticklabels=label_row, yticklabels=label_col, square=True, annot=False)
    # Note that tick labels are reversed due to previous transpose of input matrix
    plt.title('Key features similarity between DTI measurements by coefficient overlap proportion', fontsize=100)
    plt.xticks(rotation=90, fontsize=50)  # Rotate column labels for better visibility
    plt.yticks(rotation=0, fontsize=50)  # Row labels

    # Define group boundaries
    # col_groups = [0, 1, 5, 10, 20, 24, 28]  # Start positions of each group + 1 past the end
    # row_groups = [0, 1, 5, 10, 20, 24, 28]  # Start positions of each group + 1 past the end
    # col_groups = [0, 4, 9, 19, 23, 27]
    # row_groups = [0, 4, 9, 19, 23, 27]

    # Set minor tick locations to group boundaries (excludes the very first and last index to avoid out of bounds)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(row_groups[1:-1]))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(col_groups[1:-1]))

    # Enable minor gridlines at these boundaries
    ax.grid(True, which='minor', color='black', linewidth=2, linestyle='-', axis='both')

    # Disable major grid lines to prevent overlapping lines
    ax.grid(False, which='major')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=100)

    plt.savefig(outfile, bbox_inches='tight')

    cluster_heatmap(input_matrix, label_col, label_row, f"Clustered_{outfile}")


def compare_coeff(infile_A, infile_B):
    """
    Compare the 180*180 model coefficients of two models on different DTI regions (in ncanda_cv_raw/ folder)
    Writes the comparison results to /feature_comparison.
    Will return values from comparison like euclidean distance, cosine similarity, etc.

    TODO remove the hard-coded ncanda_cv_raw. Also for all functions in this whole util file, 
    auto create the folder if folder does not exist.
    """
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity

    infile_A = Path("ncanda_cv_raw/" + infile_A)
    infile_B = Path("ncanda_cv_raw/" + infile_B)
    A_prefix = infile_A.name[:-13]
    B_prefix = infile_B.name[:-13]

    coeff_A = np.array(read_coef(infile_A)) #TODO change read_coef so that it returns np arrays?
    coeff_B = np.array(read_coef(infile_B))

    # Calculate Pearson correlation
    corr, _ = pearsonr(coeff_A, coeff_B)
    scatter_corr(coeff_A, coeff_B, corr) #TODO

    # Calculate cosine similarity
    cos_sim = cosine_similarity([coeff_A], [coeff_B])
    # Euclidean distance
    euclidean_dist = np.linalg.norm(coeff_A - coeff_B)

    # Overlap of non-zero coefficients
    non_zero_A = np.nonzero(coeff_A)[0]
    non_zero_B = np.nonzero(coeff_B)[0]
    # print(non_zero_A)
    # print(non_zero_B)
    # Determine the overlap of non-zero coefficients
    overlap = np.intersect1d(non_zero_A, non_zero_B)
    overlap_proportion = len(overlap) / len(np.union1d(non_zero_A, non_zero_B))

    with open(f"feature_comparison/{A_prefix}_{B_prefix}.txt", "w") as file:
        file.write("Pearson Correlation: " + str(corr))
        file.write("\n")
        file.write("Cosine Similarity: " + str(cos_sim[0][0]))
        file.write("\n")
        file.write("Euclidean Distance: " + str(euclidean_dist))
        file.write("\n")
        file.write(f"Non-zero features overlap proportion: {overlap_proportion:.2f}\n")
        # file.write(f"Non-zero features in model A: {non_zero_A}\n")
        # file.write(f"Non-zero features in model B: {non_zero_B}\n")
        # file.write(f"Overlapping non-zero features: {overlap}\n")
    
    # match term: #TODO
    #     case "Pearson":
    #         return corr
    #     case _:
    #         return euclidean_dist
    return overlap_proportion

    get_column_name(area=5, mapped=True)
    #TODO
        # List respective top 30 - top 30, value, and their 180/22/5 classification.
        # sort by sum of absolute values
        # top-bottom: A/B value, and their 180/22/5 classification.
    for i in overlap:
        correlation_matrix = np.zeros((180, 180))
        upper_triangle_indices = np.triu_indices(180, k=1)
        # upper_triangle_indices[1]
        # correlation_matrix[upper_triangle_indices] = correlations


def batch_compare_coeff():
    """
    run the compare_coeff function on all two-model pairs from directory ncanda_cv_raw/.
    Then generate the coefficient-overlap-proportion heatmap DTI_22_22_heatmap.png 
    """

    items = get_DTI_col_name()

    n = len(items)
    result_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            result_matrix[i][j] = compare_coeff(items[i]+"_ElasticNetModel_details.json", items[j]+"_ElasticNetModel_details.json")
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", result_matrix)


    col_groups = [0, 1, 5, 10, 20, 24, 28]  # Start positions of each group + 1 past the end
    row_groups = [0, 1, 5, 10, 20, 24, 28]
    output_heatmap(result_matrix, items, items, col_groups, row_groups)










#############################################################################################
#Unused/Deprecated util functions are below:
#############################################################################################


def store_var(var, path='data.pkl'):
    """
    Stores a variable to a pickle file.
    
    Usage: store_var(any_variable, "path_for_pickle_file")
    """
    # Data to be stored
    data = {'variable': var}
    # Writing to a Pickle file
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def percentage_infs_nans(arr):
    """
    Checks for the presence of infinite values and NaNs in a numpy array of any dimension
    and calculates the total percentage of these values in the array.
    
    Parameters:
    - arr: A NumPy array to be checked.
    
    Returns:
    - percentage: The percentage of infinite values and NaNs in the array.
    """
    # Calculate the total number of elements in the array
    total_elements = arr.size
    
    # Check for the number of infinite values (both positive and negative) and NaNs
    num_infs = np.isinf(arr).sum()
    num_nans = np.isnan(arr).sum()
    
    # Calculate the total number of invalid elements (infs or NaNs)
    total_invalid = num_infs + num_nans
    
    # Calculate the percentage of invalid elements
    percentage = (total_invalid / total_elements) * 100
    return percentage



from scipy.io import loadmat
import os
def mat_to_csv(mat_file_path, output_dir):
    """
    Converts each numerical array in a .mat file to a CSV file.
    
    Parameters:
    - mat_file_path: Path to the .mat file.
    - output_dir: Directory where the CSV files will be saved.

    Example usage:
    mat_file_path = './data_highres/rsfmri_correlation_matrix/NCANDA_S00033_baseline.mat'  # Update this to the path of your .mat file
    output_dir = 'temp_output'  # The directory where you want to save the CSV files
    mat_to_csv(mat_file_path, output_dir)
    """
    # Load the .mat file
    data = loadmat(mat_file_path)

    print(data)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for key, value in data.items():
        # Skip keys that are part of the .mat file structure but not user data
        if key.startswith('__') or key.endswith('__'):
            continue
        
        # Ensure the value is a 2D numerical array (this skips structs, cell arrays, etc.)
        if isinstance(value, np.ndarray) and value.ndim == 2:
            # Convert the numpy array to a DataFrame
            df = pd.DataFrame(value)
            
            # Construct the CSV file path
            csv_file_path = os.path.join(output_dir, f"{key}.csv")
            
            # Save the DataFrame to CSV
            df.to_csv(csv_file_path, index=False)
            print(f"Saved {key} to {csv_file_path}")