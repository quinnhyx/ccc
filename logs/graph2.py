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



# === Load GPU logs ===

gpu_files = {

    '1gpu_ccc_scaling.log': 1,

    '2gpu_ccc_scaling.log': 2,

    '4gpu_ccc_scaling.log': 4,

    '8gpu_ccc_scaling.log': 8,

}



gpu_dfs = []

for file, gpu_count in gpu_files.items():

    df = pd.read_csv(file, sep=r'\s+', engine='python')

    df['GPUS'] = gpu_count

    df.rename(columns={'TIME(s)': 'TIME'}, inplace=True)

    gpu_dfs.append(df)



gpu_df = pd.concat(gpu_dfs, ignore_index=True)



# === Load CPU log ===

cpu_df = pd.read_csv('cpu_ccc_scaling.log', sep=r'\s+', engine='python')

cpu_df['GPUS'] = 0  # Set to 0 for plotting

cpu_df.rename(columns={'TIME(s)': 'TIME'}, inplace=True)



# === Combine all

df = pd.concat([gpu_df, cpu_df], ignore_index=True)



# === Plot: One plot per SIZE, showing execution time vs GPU/CPU count

for size in sorted(df['SIZE'].unique()):

    subdf = df[df['SIZE'] == size]

    

    fig, ax = plt.subplots()

    sns.lineplot(

        data=subdf,

        x='GPUS', y='TIME', hue='FEATURES',

        marker='o', ax=ax, palette='viridis', linewidth=2

    )



    ax.set_title(f'GPU/CPU Scaling Comparison — SIZE = {size}')

    ax.set_xlabel('Processor Count (0 = CPU)')

    ax.set_ylabel('Execution Time (s)')

    ax.set_xticks([0, 1, 2, 4, 8])

    ax.legend(title='Features', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()



    fig.savefig(f'compare_cpu_gpu_scaling_size_{size}.png', dpi=300)

    plt.close(fig)

