# plot.py

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

def p_to_star(p):
    if p < 0.0001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

def TCGA_predAUDRC_box_plot_twoClass(drug_name,cohort,df,sensitive,resistant,p_val,hyperparameter_folder_path):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['svg.fonttype'] = 'none'  # Use system fonts in SVG
    plt.rcParams['pdf.fonttype'] = 42  # Use Type 42 (TrueType) fonts
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(x='Label', y='predicted AUDRC', data=df, ax=ax, palette = {'0.0': 'red', '1.0': 'blue'})
    ax.set_title(f"{drug_name} in {cohort}", fontsize=14, fontweight="bold")
    # p_text = f"p = {p_val:.4f}"
    p_text = p_to_star(p_val)
    x1, x2 = 0, 1
    y, h = max(df['predicted AUDRC']) + 0.002, 0.002
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((x1+x2) / 2, y+h, p_text, ha='center', va='bottom', fontsize=14)#, color='red'
    # Axis labels # label是0的一定在前面，所以sen=0那就要放前面
    ax.set_xticklabels([    f'resistant (n={len(resistant)})', # \nlabel=0
                            f'sensitive (n={len(sensitive)})'], fontsize=14, fontweight="bold") #\nlabel=1
    # ax.set_xlabel("Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted AUDRC", fontsize=14, fontweight="bold" )
    plt.tight_layout()
    output_file = os.path.join(hyperparameter_folder_path, f'TCGA_{drug_name}_predAUDRC_box_plot_twoClass.png')
    try:
        fig.savefig(output_file)
        os.chmod(output_file, 0o444)
    except Exception as e:
        print(f"Error occurred while saving or setting permissions for {output_file}: {str(e)}")
    return plt
    
def heatmap(attention_scores_Matrix, drug_ID, cell_ID, include_omics ,fontsize_ticks=8, module="", sub_list=None ):
    #sub: sequence of substructure ESPF text, list type
    # Define the colors and their corresponding positions (anchors)
    colors = ["#67749f","#6581b2","#b4d5e5", "white","#fae19b","#e9a94f", "red","#a10318"]  # Color sequence
    anchors = [0.0,0.1, 0.35,0.5,0.55,0.75,0.9, 1.0]  # Position of each color (0 = min, 1 = max)

    # Create a custom colormap with specified anchors
    custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap", list(zip(anchors, colors)))
    plt.figure(figsize=(10, 8))

    plt.imshow(attention_scores_Matrix, aspect="auto", cmap=custom_cmap, vmin=1, vmax=0) # make all color correspond to the specific value
    cbar = plt.colorbar(label="attention score")
    cbar.ax.yaxis.label.set_weight("bold")  # Make label bold
    cbar.outline.set_visible(False) # remove the boundary/frame of the colorbar 

    # Set x and y axis labels
    if module == "AttenScorMat_DrugSelf":
        plt.xlabel("Drug Substructures", fontsize=14, fontweight='bold', fontname='Times New Roman')
        plt.ylabel("Drug Substructures", fontsize=14, fontweight='bold', fontname='Times New Roman')
        
    # Optionally, set the x and y ticks (example with some labels)
    if module == "AttenScorMat_DrugSelf":
        plt.title(f"{drug_ID} SelfAttention Score Matrix", fontsize=14, fontweight='bold')
        if sub_list is not None:
            plt.xticks(fontsize=fontsize_ticks,fontweight="bold",rotation=75, ticks=np.arange(0, attention_scores_Matrix.shape[1], step=1), labels=([f"{i}" for i in sub_list]) ) 
            plt.yticks(fontsize=fontsize_ticks,fontweight="bold",ticks=np.arange(0, attention_scores_Matrix.shape[1], step=1), labels=([f"{i}" for i in sub_list]) ) 
        else:
            plt.xticks(fontsize=fontsize_ticks,fontweight="bold",rotation=75, ticks=np.arange(0, attention_scores_Matrix.shape[1], step=1), labels=[f"sub{i+1}" for i in range(attention_scores_Matrix.shape[1])])
            plt.yticks(fontsize=fontsize_ticks,fontweight="bold",ticks=np.arange(0, attention_scores_Matrix.shape[1], step=1), labels=[f"sub{i+1}" for i in range(attention_scores_Matrix.shape[1])])
            
    if module == "AttenScorMat_DrugCellSelf":
        plt.title(f"{drug_ID} - {cell_ID} SelfAttention Score Matrix", fontsize=14, fontweight='bold')
        if sub_list is not None:
            plt.xticks(fontsize=fontsize_ticks,fontweight="bold",rotation=75, ticks=np.arange(0, attention_scores_Matrix.shape[1], step=1), labels=([f"{i}" for i in sub_list] + include_omics) ) 
            plt.yticks(fontsize=fontsize_ticks,fontweight="bold",ticks=np.arange(0, attention_scores_Matrix.shape[1], step=1), labels=([f"{i}" for i in sub_list] + include_omics) ) 
        else:
            plt.xticks(fontsize=fontsize_ticks,fontweight="bold",rotation=75, ticks=np.arange(0, attention_scores_Matrix.shape[1], step=1), labels=([f"sub{i+1}" for i in range(attention_scores_Matrix.shape[1]-len(include_omics))] + include_omics) ) 
            plt.yticks(fontsize=fontsize_ticks,fontweight="bold",ticks=np.arange(0, attention_scores_Matrix.shape[1], step=1), labels=([f"sub{i+1}" for i in range(attention_scores_Matrix.shape[1]-len(include_omics))] + include_omics) ) 
        
        
    for spine in plt.gca().spines.values(): # remove the boundary/frame of the plot 
        spine.set_visible(False)
    plt.show()

def loss_curve(model_name, train_epoch_loss_list, val_epoch_loss_list, best_epoch, best_val_loss, hyperparameter_folder_path,loss_type="loss_WO_penalty"):
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
    train_max = max(train_epoch_loss_list)
    val_max = max(val_epoch_loss_list)
    train_min = min(train_epoch_loss_list)
    val_min = min(val_epoch_loss_list)
    if (train_min <= 1.5 < train_max) and (val_min <= 1.5 < val_max):
        plt.ylim(top=1.5, bottom=0)
    else:
        plt.ylim(top=None, bottom=None)
    output_file = os.path.join(hyperparameter_folder_path, f'train_valid_{loss_type}_curve.png')
    try:
        fig.savefig(output_file)
        os.chmod(output_file, 0o444)
    except Exception as e:
        print(f"Error occurred while saving or setting permissions for {output_file}: {str(e)}")
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
    output_file = os.path.join(hyperparameter_folder_path, 'Density_Plot_of_Correlation.png')
    try:
        fig.savefig(output_file)
        if platform.system() == "Windows":
            os.system(f'attrib +r "{output_file}"')
        else:
            os.chmod(output_file, 0o444)
    except Exception as e:
        print(f"Error occurred while saving or setting permissions for {output_file}: {str(e)}")
    return plt

# plot Density_Plot_of_AUC_Values of train val test datasets
import matplotlib as mpl
def Density_Plot_of_AUC_Values(datasets,hyperparameter_folder_path=None):
    plt.style.use('default')  # Use default style instead of seaborn
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['figure.dpi'] = 300  # Higher resolution
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['figure.figsize'] = (10, 10)  # Slightly larger figure
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = 1.0
    mpl.rcParams['agg.path.chunksize'] = 10000
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['svg.fonttype'] = 'none'  # Use system fonts in SVG
    plt.rcParams['pdf.fonttype'] = 42  # Use Type 42 (TrueType) fonts
    plt.rcParams["font.family"] = "serif"
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, (targets, outputs, title, color) in enumerate(datasets):
        sns.kdeplot(torch.cat(targets).tolist(), color=color, label=f'GroundTruthAUDRC - {title}', ax=axs[i])
        sns.kdeplot(torch.cat(outputs).tolist(), color=color, linestyle='--', label=f'PredictedAUDRC - {title}', ax=axs[i])
        axs[i].set_title(f'{title} Set',fontsize=14,fontweight="bold")
        axs[i].set_xlabel('AUDRC values',fontsize=14,fontweight="bold")
        axs[i].set_ylabel('Density', fontsize=14,fontweight="bold")
        axs[i].legend(loc='upper left',fontsize=12)
    # Set a global title and save the figure
    fig.suptitle('Density Plot of AUDRC Values for Train, Val, and Test Sets', fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if hyperparameter_folder_path is not None:
        output_file = os.path.join(hyperparameter_folder_path, 'Density_Plot_of_AUC_Values.png')
        try:
            fig.savefig(output_file)
            os.chmod(output_file, 0o444)
            print(f"✅ Set read-only permissions on: {output_file}")
        except Exception as e:
            print(f"⚠️ Failed to set permissions: {e}")
    return plt


def Confusion_Matrix_plot(datasets,hyperparameter_folder_path=None,drug=None):
    plt.rcParams["font.family"] = "serif"
    if len(datasets)==1:
        fig, ax = plt.subplots(figsize=(7, 6))
        cm, title, color = datasets[0]
        sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 16, "weight": "bold"}, 
                    cmap=color, cbar=False, vmin=0, vmax=max(cm.max() * 1.3, 1), linewidths=0,
                    xticklabels=["Predicted  0", "Predicted  1"], yticklabels=["Actual  0", "Actual  1"], ax=ax)
        if drug is not None:
            ax.set_title(f'{drug} {title} samples',fontsize=16, fontweight='bold')
        else:
            ax.set_title(f'{title} samples',fontsize=16, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')
        # Make x and y tick labels bold
        ax.tick_params(axis='x', labelsize=12, labelrotation=0)
        ax.tick_params(axis='y', labelsize=12, labelrotation=0)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        if hyperparameter_folder_path is not None:
            output_file = os.path.join(hyperparameter_folder_path, f'{drug}Confusion_Matrix.png')
            try:
                fig.savefig(output_file)
                os.chmod(output_file, 0o444)
                print(f"✅ Set read-only permissions on: {output_file}")
            except Exception as e:
                print(f"⚠️ Failed to set permissions: {e}")
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, (cm, title, color) in enumerate(datasets):
            sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 16, "weight": "bold"}, 
                        cmap=color, cbar=False, vmin=0, vmax=max(cm.max() * 1.3, 1), linewidths=0,
                        xticklabels=["Predicted  0", "Predicted  1"], yticklabels=["Actual  0", "Actual  1"], ax=axs[i])
            axs[i].set_title(f'{title} Set',fontsize=16, fontweight='bold')
            for spine in axs[i].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor('black')
            # Make x and y tick labels bold
            axs[i].tick_params(axis='x', labelsize=12, labelrotation=0)
            axs[i].tick_params(axis='y', labelsize=12, labelrotation=0)
            for label in axs[i].get_xticklabels() + axs[i].get_yticklabels():
                label.set_fontweight('bold')
        # Set a global title and save the figure
        fig.suptitle('Confusion Matrix for Train, Val, and Test Sets', fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if hyperparameter_folder_path is not None:
            output_file = os.path.join(hyperparameter_folder_path, 'Confusion_Matrix.png')
            try:
                fig.savefig(output_file)
                os.chmod(output_file, 0o444)
                print(f"✅ Set read-only permissions on: {output_file}")
            except Exception as e:
                print(f"⚠️ Failed to set permissions: {e}")
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
