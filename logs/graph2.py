import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# === Plot settings ===

sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({

    'figure.figsize':   (10, 6),

    'figure.dpi':       100,

    'font.size':        12,

    'axes.titlesize':   13,

    'axes.labelsize':   12,

    'legend.fontsize':  10,

})



# === File-GPU mapping ===

files = {

    'cpu_ccc_scaling.log': 0,

    '1gpu_ccc_scaling.log': 1,

    '2gpu_ccc_scaling.log': 2,

    '4gpu_ccc_scaling.log': 4,

    '8gpu_ccc_scaling.log': 8,

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



# === Create a human-readable label

df['Label'] = df['GPUS'].astype(str) + ' GPU(s)'



# === Determine label order

labels = df['Label'].unique()

hue_order = sorted(labels, key=lambda x: int(x.split()[0]))



# === Group by SIZE and FEATURES

groups = df.groupby(['SIZE', 'FEATURES'])



for (size, feat), group in groups:

    # Pivot and melt for plotting

    pivot = (

        group.groupby(['Label', 'GPUS'])['TIME']

             .min()

             .unstack('GPUS')

             .reset_index()

             .melt(id_vars='Label', var_name='GPUS', value_name='TIME')

    )



    # Convert GPU column to integer for x-axis

    pivot['GPUS'] = pivot['GPUS'].astype(int)



    # Plot

    fig, ax = plt.subplots()

    sns.lineplot(

        data=pivot,

        x='GPUS', y='TIME', hue='Label',

        marker='o', ax=ax, palette="rocket", hue_order=hue_order,

        alpha=0.85

    )



    ax.set_title(f'Performance Scaling — SIZE={size}, FEATURES={feat}')

    ax.set_xlabel('Number of GPUs (0 = CPU)')

    ax.set_ylabel('Execution Time (s)')

    ax.set_xticks([0, 1, 2, 4, 8])

    ax.legend(title='Config', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    fig.savefig(f'perf_scaling_size{size}_feat{feat}.png', dpi=300)

    plt.close(fig)

