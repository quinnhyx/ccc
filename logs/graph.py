import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({

    'figure.figsize':   (10, 6),

    'figure.dpi':       100,

    'font.size':        12,

    'axes.titlesize':   13,

    'axes.labelsize':   12,

    'legend.fontsize':  10,

})



# File-GPU mapping

files = {

    '1gpu_ccc_scaling.log': 1,

    '2gpu_ccc_scaling.log': 2,

    '4gpu_ccc_scaling.log': 4,

    '8gpu_ccc_scaling.log': 8,

}



# Load and combine

dfs = []

for file, gpu in files.items():

    df = pd.read_csv(file, sep=r'\s+', engine='python')

    df['GPUS'] = gpu

    dfs.append(df)



df = pd.concat(dfs, ignore_index=True)

df.rename(columns={'TIME(s)': 'TIME'}, inplace=True)



# Plot for each SIZE

for size in sorted(df['SIZE'].unique()):

    subdf = df[df['SIZE'] == size]

    

    fig, ax = plt.subplots()

    sns.lineplot(

        data=subdf,

        x='GPUS', y='TIME', hue='FEATURES',

        marker='o', ax=ax, palette='viridis', alpha=0.9

    )

    

    ax.set_title(f'GPU Scaling Comparison — SIZE = {size}')

    ax.set_xlabel('Number of GPUs')

    ax.set_ylabel('Execution Time (s)')

    ax.set_xticks([1, 2, 4, 8])

    ax.legend(title='Features', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    

    fig.savefig(f'gpuscaling_size_{size}.png', dpi=300)

    plt.close(fig)


