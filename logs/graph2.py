import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# === Plot settings ===

sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({

    'figure.figsize': (10, 6),

    'figure.dpi': 100,

    'font.size': 12,

    'axes.titlesize': 13,

    'axes.labelsize': 12,

    'legend.fontsize': 10,

})



# === File: GPU count mapping ===

files = {

    'cpu_ccc_scaling.log': 0,

    'test_1gpu_ccc_scaling.log': 1,

    'test_2gpu_ccc_scaling.log': 2,

    'test_4gpu_ccc_scaling.log': 4,

    'test_8gpu_ccc_scaling.log': 8,

}



# === Load logs ===

dfs = []

for file, gpu in files.items():

    df = pd.read_csv(file, sep=r'\s+', engine='python')

    df['GPUS'] = gpu

    dfs.append(df)



# === Combine and clean ===

df = pd.concat(dfs, ignore_index=True)

df.rename(columns={'TIME(s)': 'TIME'}, inplace=True)



# 🧼 Handle duplicates: take minimum TIME for same GPUS + SIZE + FEATURES

df = (

    df.groupby(['GPUS', 'SIZE', 'FEATURES'], as_index=False)['TIME']

    .min()

    .sort_values(by=['FEATURES', 'GPUS'])

)



# === Plot per SIZE ===

for size in sorted(df['SIZE'].unique()):

    subdf = df[df['SIZE'] == size]



    fig, ax = plt.subplots()

    sns.lineplot(

        data=subdf,

        x='GPUS', y='TIME', hue='FEATURES',

        marker='o', palette='viridis', ax=ax, linewidth=2, alpha=0.9

    )



    ax.set_title(f'CPU vs GPU Scaling — SIZE = {size}')

    ax.set_xlabel('Processor Count (0 = CPU)')

    ax.set_ylabel('Execution Time (s)')

    ax.set_xticks([0, 1, 2, 4, 8])

    ax.legend(title='Features', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()



    fig.savefig(f'cpu_gpu_scaling_size_{size}.png', dpi=300)

    plt.close(fig)

