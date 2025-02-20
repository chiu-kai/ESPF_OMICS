# plot.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def loss_curve(model_name, loss_values_train, loss_values_validation, best_epoch, best_val_loss, hyperparameter_folder_path, ylim_top):
    # Create a plot of the loss curve
    fig=plt.figure(figsize=(8, 6))
    plt.figure(figsize=(8, 6))
    # Plot training loss
    plt.plot(range(1, len(loss_values_train) + 1), loss_values_train, marker='o', linestyle='-', color='b', label='Training Loss')
    # Plot validation loss
    plt.plot(range(1, len(loss_values_validation) + 1), loss_values_validation, marker='o', linestyle='-', color='r', label='Validation Loss')
    # Mark the best epoch
    plt.plot(best_epoch, best_val_loss , marker='o', markersize=8, linestyle='', color='c', label=f'Best Epoch:{best_epoch}')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curve of '+ model_name, fontsize=16)
    plt.legend()  # Add legend to differentiate between training and validation losses
    plt.grid(True)
    # Set y-axis limit
    plt.ylim(top=ylim_top)
    plt.show()
    fig.savefig(f'{hyperparameter_folder_path}/training validation loss_curve')

def correlation_density(model_name,train_pearson,val_pearson,test_pearson,train_spearman,val_spearman,test_spearman, hyperparameter_folder_path):
    #pearson
    # Create a density plot using seaborn's kdeplot function
    fig=plt.figure(figsize=(14, 5))
    # Set the title of the plot
    plt.title(f'Density Plot of Correlation {model_name}', fontsize=16)
    
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
    plt.show()
    fig.savefig(f'{hyperparameter_folder_path}/Density_Plot_of_Correlation')
    
    
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
