import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# === Plot style settings ===

sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({

    'figure.figsize': (10, 6),

    'figure.dpi': 100,

    'font.size': 12,

    'axes.titlesize': 13,

    'axes.labelsize': 12,

    'legend.fontsize': 10,

})



# === File → GPU count mapping (0 = CPU) ===

files = {

    'cpu_ccc_scaling.log': 0,

    'test_1gpu_ccc_scaling.log': 1,

    'test_2gpu_ccc_scaling.log': 2,

    'test_4gpu_ccc_scaling.log': 4,

    'test_8gpu_ccc_scaling.log': 8,

}



# === Load all logs and label with GPU count ===

dfs = []

for file, gpu in files.items():

    df = pd.read_csv(file, sep=r'\s+', engine='python')

    df['GPUS'] = gpu  # Add GPU count column

    dfs.append(df)



# === Combine all into one DataFrame ===

df = pd.concat(dfs, ignore_index=True)

df.rename(columns={'TIME(s)': 'TIME'}, inplace=True)



# === Sort for clean plotting ===

df = df.sort_values(by=['FEATURES', 'GPUS'])



# === Plot execution time by GPU count, for each SIZE ===

for size in sorted(df['SIZE'].unique()):

    subdf = df[df['SIZE'] == size]



    fig, ax = plt.subplots()

    sns.lineplot(

        data=subdf,

        x='GPUS', y='TIME', hue='FEATURES',

        marker='o', palette='viridis', ax=ax, linewidth=2, alpha=0.9

    )



    ax.set_title(f'CPU vs GPU Scaling — SIZE = {size}')

    ax.set_xlabel('Number of GPUs (0 = CPU)')

    ax.set_ylabel('Execution Time (s)')

    ax.set_xticks([0, 1, 2, 4, 8])

    ax.legend(title='Features', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()



    fig.savefig(f'cpu_gpu_scaling_size_{size}.png', dpi=300)

    plt.close(fig)

