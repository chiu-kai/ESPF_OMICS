import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#pearson 

def DensityPlot(train_val_pearson,test_pearson):
# Create a density plot using seaborn's kdeplot function
    sns.kdeplot(train_val_pearson, shade=True, color='red', label='Train/Validation',edgecolor='black')
    sns.kdeplot(test_pearson, shade=True, linestyle='dashed', color='red', label='Test',edgecolor='black')
    #sns.kdeplot(train_val_pearson_only, shade=True, color='blue', label='Train/Validation',edgecolor='black')
    #sns.kdeplot(test_pearson_only, shade=True, color='blue', linestyle='dashed', label='Test',edgecolor='black')
    plt.xlabel('Pearson\'s Correlation Coefficient Value')
    plt.ylabel('Density')
    plt.title('Density Plot of Mut_DrugModel')
    plt.legend()
    plt.show()