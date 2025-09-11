import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

def create_success_rates_plot(save_path=None):
    """
    Create Figure 10: Success Rates of Algorithms at each Threshold
    """
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from Table 5
    algorithms = ['Clarke-Wright', 'ALNS', 'SAC']
    success_90 = [100.0, 100.0, 17.0]
    success_95 = [67.0, 100.0, 2.0]
    success_99 = [11.0, 100.0, 0.0]
    
    # Define colors using professional academic palette
    colors = {
        '90%': "red",  # Dark blue
        '95%': "black",  # Magenta
        '99%': "grey"   # Amber
    }
    
    # Set up bar positions
    x = np.arange(len(algorithms))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, success_90, width, 
                   label='90%', color=colors['90%'], 
                   alpha=0.8, edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x, success_95, width, 
                   label='95%', color=colors['95%'], 
                   alpha=0.8, edgecolor='white', linewidth=0.8)
    bars3 = ax.bar(x + width, success_99, width, 
                   label='99%', color=colors['99%'], 
                   alpha=0.8, edgecolor='white', linewidth=0.8)
    
    # Customize the plot
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=12)
    
    # Format axes
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_facecolor('#FAFAFA')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=10)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color=colors['90%'], label='90%'),
        mpatches.Patch(color=colors['95%'], label='95%'),
        mpatches.Patch(color=colors['99%'], label='99%')
    ]
    
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
                       fontsize=12)
    
    # Format y-axis numbers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    
    # Adjust layout and styling
    plt.tight_layout()
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.2)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    plt.show()

# Usage
if __name__ == "__main__":
    create_success_rates_plot(save_path='success_rates.png')