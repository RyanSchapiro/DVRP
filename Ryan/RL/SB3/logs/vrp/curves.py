import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def plot_training(file_paths, save_path=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'R-Model': "#E53BB2",
        'C-Model': "#F50202",
        'RC-Model': "#090600"
    }
    
    for model_name, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        steps = df['Step']
        values = df['Value']
        
        ax.plot(steps, values, 
               color=colors[model_name], 
               linewidth=2.5, 
               alpha=0.8,
               label=model_name)
    
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_facecolor("#FFFFFF")
    
    legend_elements = []
    for model_name, color in colors.items():
        if model_name == 'R-Model':
            label = 'Random Instances (R-Type)'
        elif model_name == 'C-Model':
            label = 'Clustered Instances (C-Type)'
        else:
            label = 'Random-Clustered Instances (RC-Type)'
        
        legend_elements.append(mpatches.Patch(color=color, label=label))
    
    legend = ax.legend(handles=legend_elements,
                   bbox_to_anchor=(0.5, -0.15),
                   loc='upper center',
                   ncol=3,
                   frameon=True,
                   fancybox=True,
                   shadow=True,
                   framealpha=0.9,
                   facecolor='white',
                   edgecolor='gray',
                   fontsize=12,
                   title='Instance Types',
                   title_fontsize=13)
    legend.get_title().set_fontweight('bold')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    
    for spine in ax.spines.values():
        spine.set_edgecolor("#FBF8F8")
        spine.set_linewidth(1.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    plt.show()

if __name__ == "__main__":
    file_paths = {
        'R-Model': 'R_R-model.csv',
        'C-Model': 'C_C-model.csv',
        'RC-Model': 'RC_RC-model.csv'
    }
    
    plot_training(file_paths, save_path='sac_training_curves.png')