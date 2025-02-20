# plot.py

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def heatmap(attention_scores):
    # Define the colors and their corresponding positions (anchors)
    colors = ["#67749f","#6581b2","#b4d5e5", "white","#fae19b","#e9a94f", "red","#a10318"]  # Color sequence
    anchors = [0.0,0.1, 0.35,0.5,0.55,0.75,0.9, 1.0]  # Position of each color (0 = min, 1 = max)

    # Create a custom colormap with specified anchors
    custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap", list(zip(anchors, colors)))
    plt.figure(figsize=(7, 4))

    plt.imshow(attention_scores, aspect="auto", cmap=custom_cmap)
    cbar = plt.colorbar(label="attention score")
    cbar.outline.set_visible(False) # remove the boundary/frame of the colorbar 
    plt.title("Attention Score Matrix", fontsize=14, fontweight='bold')

    # Set x and y axis labels
    plt.xlabel("Drug Substructures", fontsize=14, fontweight='bold', fontname='Times New Roman')
    plt.ylabel("Genesets", fontsize=14, fontweight='bold', fontname='Times New Roman')

    # Optionally, set the x and y ticks (example with some labels)
    plt.xticks(ticks=np.arange(0, 10, step=1), labels=[f"sub{i}" for i in range(10)])
    # plt.yticks(ticks=np.arange(0, 100, step=10), labels=[f"Sample {i+1}" for i in range(0, 100, 10)])
    plt.yticks([])
    for spine in plt.gca().spines.values(): # remove the boundary/frame of the plot 
        spine.set_visible(False)
    plt.show()

def loss_curve(model_name, train_epoch_loss_list, val_epoch_loss_list, best_epoch, best_val_loss, hyperparameter_folder_path, ylim_top):
    # Create a plot of the loss curve
    fig= plt.figure(figsize=(8, 6))
    # Plot training loss
    plt.plot(range(1, len(train_epoch_loss_list) + 1), train_epoch_loss_list, marker='o', linestyle='-', color='b', label='Training Loss')
    # Plot validation loss
    plt.plot(range(1, len(val_epoch_loss_list) + 1), val_epoch_loss_list, marker='o', linestyle='-', color='r', label='Validation Loss')
    # Mark the best epoch
    plt.plot(best_epoch, best_val_loss , marker='o', markersize=8, linestyle='', color='c', label=f'Best Epoch:{best_epoch}')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curve of '+ model_name, fontsize=16)
    plt.legend()  # Add legend to differentiate between training and validation losses
    plt.grid(True)
    # Set y-axis limit
    plt.ylim(top=ylim_top)
    fig.savefig(os.path.join(hyperparameter_folder_path, 'training_validation_loss_curve.png'))
    return plt
    

# def correlation_density(model_name,train_pearson,val_pearson,test_pearson,train_spearman,val_spearman,test_spearman):
#     #pearson
#     # Create a density plot using seaborn's kdeplot function
#     fig=plt.figure(figsize=(14, 5))
#     plt.subplot(1,2,1)
#     sns.kdeplot(train_pearson, fill=True, color='blue', label='Train',linewidth=1.5)
#     sns.kdeplot(val_pearson, fill=True, color='red', label='Validation',linewidth=1.5)
#     sns.kdeplot(test_pearson, fill=True, color='green', label='Test',linewidth=1.5)
#     # Set the x-axis label to 'Density'
#     plt.xlabel('Pearson\'s Correlation Coefficient Value', fontsize=16)
#     # Set the y-axis label to 'Pearson\'s Correlation Coefficient Value'
#     plt.ylabel('Density', fontsize=16)
#     # Set the title of the plot
#     plt.title(f'Density Plot of PDAC {model_name}', fontsize=16)
#     plt.legend(loc='upper left',fontsize=10) 
#     plt.subplot(1,2,2)

#     #spearman
#     # Create a density plot using seaborn's kdeplot function
#     sns.kdeplot(train_spearman, fill=True, color='blue', label='Train',linewidth=1.5)
#     sns.kdeplot(val_spearman, fill=True, color='red', label='Validation',linewidth=1.5)
#     sns.kdeplot(test_spearman, fill=True, color='green', label='Test',linewidth=1.5)
#     # Set the x-axis label to 'Density'
#     plt.xlabel('Spearman\'s Correlation Coefficient Value', fontsize=16)
#     # Set the y-axis label to 'Pearson\'s Correlation Coefficient Value'
#     plt.ylabel('Density', fontsize=16)
#     # Set the title of the plot
#     plt.title(f'Density Plot of PDAC {model_name}', fontsize=16)
#     plt.legend(loc='upper left',fontsize=10) 
#     plt.show()

def correlation_density(model_name,train_pearson,val_pearson,test_pearson,train_spearman,val_spearman,test_spearman, hyperparameter_folder_path):
    #pearson
    # Create a density plot using seaborn's kdeplot function
    fig=plt.figure(figsize=(14, 5))
    # Set the title of the plot
    
    plt.subplot(1,2,1)
    sns.kdeplot(train_pearson, fill=True, color='blue', label='Train',linewidth=1.5)
    sns.kdeplot(val_pearson, fill=True, color='red', label='Validation',linewidth=1.5)
    sns.kdeplot(test_pearson, fill=True, color='green', label='Test',linewidth=1.5)
    # Set the x-axis label to 'Density'
    plt.xlabel('Pearson\'s Correlation Coefficient Value', fontsize=16)
    # Set the y-axis label to 'Pearson\'s Correlation Coefficient Value'
    plt.ylabel('Density', fontsize=16)
    plt.legend(loc='upper left',fontsize=10) 

    plt.subplot(1,2,2)
    #spearman
    # Create a density plot using seaborn's kdeplot function
    sns.kdeplot(train_spearman, fill=True, color='blue', label='Train',linewidth=1.5)
    sns.kdeplot(val_spearman, fill=True, color='red', label='Validation',linewidth=1.5)
    sns.kdeplot(test_spearman, fill=True, color='green', label='Test',linewidth=1.5)
    # Set the x-axis label to 'Density'
    plt.xlabel('Spearman\'s Correlation Coefficient Value', fontsize=16)
    # Set the y-axis label to 'Pearson\'s Correlation Coefficient Value'
    plt.ylabel('Density', fontsize=16)
    plt.legend(loc='upper left',fontsize=10) 
    fig.suptitle(f'Density Plot of Correlation {model_name}', fontsize=16)
    fig.savefig(f'{hyperparameter_folder_path}/Density_Plot_of_Correlation')
    return plt

# plot Density_Plot_of_AUC_Values of train val test datasets
def Density_Plot_of_AUC_Values(datasets,hyperparameter_folder_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, (targets, outputs, title, color) in enumerate(datasets):
        sns.kdeplot(np.concatenate(targets).tolist(), color=color, label=f'GroundTruthAUC - {title}', ax=axs[i])
        sns.kdeplot(np.concatenate(outputs).tolist(), color=color, linestyle='--', label=f'PredictedAUC - {title}', ax=axs[i])
        axs[i].set_title(f'{title} Set',fontsize=14)
        axs[i].set_xlabel('AUC values',fontsize=14)
        axs[i].legend(loc='upper left',fontsize=10)
    # Set a global title and save the figure
    fig.suptitle('Density Plot of AUC Values for Train, Val, and Test Sets', fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if hyperparameter_folder_path is not None:
        fig.savefig(f'{hyperparameter_folder_path}/Density_Plot_of_AUC_Values')
    return plt




# plot predicted AUC value for every sample by drug label 
'''
sns.set_style('white')
# Get unique labels
unique_labels = sorted(list(set(drug_names_instance)))

# Map each label to a color
label_to_color = {'Cisplatin': '#f59b00',
 'TS-1': 'Green',
 'FLUOROURACIL': 'Turquoise',
 'LEUCOVORIN': 'red',
 'ELOXATIN': 'blue',
 'IRINOTECAN': '#bc08cc',
 'PACLITAXEL': '#7ecc08',
 'GEMCITABINE': 'black'}

# Assign colors based on labels
colors = [label_to_color[label] for label in drug_names_instance]

# Plotting the dot plot
plt.figure(figsize=(8, 6))
plt.scatter(range(len(predict_outputs_list)), predict_outputs_list, c=colors, alpha=0.7)

# Adding a legend
for label in unique_labels:
    plt.scatter([], [], c=label_to_color[label], label=label)

plt.legend(title="Labels",fontsize=11)
plt.title(f'{cancer_type} {gene_num}genes {sample_num}samples',fontsize=15)
plt.xlabel('samples',fontsize=15)
plt.ylabel('Predicted AUC',fontsize=15)
plt.show()
'''

# plot predicted AUC value for every sample by drug label ***one drug one plt***
'''
sns.set_style('white')
unique_labels = list(set(drug_names_instance))

label_to_color = {'Cisplatin': '#f59b00',
 'TS-1': 'Green',
 'FLUOROURACIL': 'Turquoise',
 'LEUCOVORIN': 'red',
 'ELOXATIN': 'blue',
 'IRINOTECAN': '#bc08cc',
 'PACLITAXEL': '#7ecc08',
 'GEMCITABINE': 'black'}

# Calculate the average predicted AUC for each drug
drug_avg = {}
for label in unique_labels:
    indices = [i for i, drug in enumerate(drug_names_instance) if drug == label]
    avg_value = np.mean([predict_outputs_list[i] for i in indices])
    drug_avg[label] = avg_value

# Sort the drugs by average predicted AUC
sorted_labels = sorted(drug_avg, key=drug_avg.get)

# Determine the global y-axis range
all_outputs = np.concatenate([np.array([predict_outputs_list[j] for j in range(len(drug_names_instance)) if drug_names_instance[j] == label]) for label in unique_labels])
y_min, y_max = np.min(all_outputs), np.max(all_outputs)
print('y_min, y_max',y_min, y_max)
# Set up a grid of 2x4 subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Plot each drug in its subplot according to the sorted order
for i, label in enumerate(sorted_labels):
    # Filter the data for the current label
    indices = [j for j, drug in enumerate(drug_names_instance) if drug == label]
    filtered_outputs = [predict_outputs_list[j] for j in indices]
    
    # Plotting the dot plot for the current drug in the respective subplot
    axes[i].scatter(range(len(filtered_outputs)), filtered_outputs, c=label_to_color[label], alpha=0.7)
    axes[i].set_title(f'{label} (Avg: {drug_avg[label]:.3f})', fontsize=20)
    axes[i].set_xlabel('Samples', fontsize=15)
    axes[i].set_ylabel('Predicted AUC', fontsize=15)
    axes[i].set_ylim(y_min, 1)  # Set the y-axis range

# Adjust layout
plt.suptitle(f'{cancer_type} {gene_num} genes {sample_num}samples', fontsize=30)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
'''
# plot PCA for mutation data 
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
sns.set_style('darkgrid')
pca = PCA(n_components=2)
T_package = pca.fit_transform(cancer_type_matched_genenames_TCGA2649_df)

fig=plt.figure(figsize=(14, 5))
plt.subplot(1,2,1)
plt.scatter(x=T_package[:, 0], y=T_package[:, 1], cmap='viridis')
plt.hlines(y=0, xmin=-2, xmax=2, colors='black', linestyle='--')
plt.vlines(x=0, ymin=-2, ymax=2, colors='black', linestyle='--')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% expl. var)",fontsize=15)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% expl. var)",fontsize=15)
plt.title(f'PCA {cancer_type} ori. {gene_num} genes',fontsize=20)
# Adding labels to each point
threshold = 7
mask_outliers = (np.abs(T_package[:, 0]) > threshold) | (np.abs(T_package[:, 1]) > threshold)
sample_names = ccl_names_mut
for i, name in enumerate(sample_names):
    if mask_outliers[i]:
        plt.text(T_package[i, 0], T_package[i, 1], name, fontsize=9, ha='right')

plt.subplot(1,2,2)
threshold=9
mask = (T_package[:, 0] >= -threshold) & (T_package[:, 0] <= threshold) & (T_package[:, 1] >= -threshold) & (T_package[:, 1] <= threshold)
# Apply the mask to filter the data points
T_package_filtered = T_package[mask]

plt.scatter(x=T_package_filtered[:, 0], y=T_package_filtered[:, 1], cmap='viridis')
plt.hlines(y=0, xmin=-threshold, xmax=threshold, colors='black', linestyle='--')
plt.vlines(x=0, ymin=-threshold, ymax=threshold, colors='black', linestyle='--')
plt.xlim(-threshold, threshold)
plt.ylim(-threshold, threshold)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% expl. var)", fontsize=15)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% expl. var)", fontsize=15)
plt.title(f'PCA {cancer_type} ori. {gene_num} genes', fontsize=20)
plt.show()
'''
