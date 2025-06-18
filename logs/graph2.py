import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# === PLOT SETTINGS ===

sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({

    'figure.figsize':   (10, 6),

    'figure.dpi':       100,

    'font.size':        12,

    'axes.titlesize':   13,

    'axes.labelsize':   12,

    'legend.fontsize':  10,

})



# === LOAD GPU LOGS ===

gpu_files = {

    '1gpu_ccc_scaling.log': '1 GPU',

    '2gpu_ccc_scaling.log': '2 GPUs',

    '4gpu_ccc_scaling.log': '4 GPUs',

    '8gpu_ccc_scaling.log': '8 GPUs',

}



gpu_dfs = []

for file, label in gpu_files.items():

    df = pd.read_csv(file, sep=r'\s+', engine='python')

    df['Label'] = label

    df['GPUS'] = int(label.split()[0])  # Extract numeric GPU count

    df.rename(columns={'TIME(s)': 'TIME'}, inplace=True)

    gpu_dfs.append(df)



gpu_df = pd.concat(gpu_dfs, ignore_index=True)



# === LOAD CPU LOG ===

cpu_df = pd.read_csv('cpu_ccc_scaling.log', sep=r'\s+', engine='python')

cpu_df['Label'] = 'CPU (24 threads)'

cpu_df['GPUS'] = 0  # For x-axis consistency

cpu_df.rename(columns={'TIME(s)': 'TIME'}, inplace=True)



# === COMBINE ALL DATA ===

df = pd.concat([gpu_df, cpu_df], ignore_index=True)



# === PLOT PER SIZE ===

for size in sorted(df['SIZE'].unique()):

    subdf = df[df['SIZE'] == size]

    

    fig, ax = plt.subplots()

    sns.lineplot(

        data=subdf,

        x='GPUS', y='TIME', hue='FEATURES',

        style='Label',  # Differentiate CPU vs GPU

        markers=True, dashes=False,

        marker='o', palette='viridis', ax=ax, alpha=0.9

    )



    ax.set_title(f'Execution Time vs GPU Count — SIZE = {size}')

    ax.set_xlabel('Number of GPUs (0 = CPU)')

    ax.set_ylabel('Execution Time (s)')

    ax.set_xticks([0, 1, 2, 4, 8])

    ax.legend(title='Features', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()



    fig.savefig(f'gpuscaling_with_cpu_size_{size}.png', dpi=300)

    plt.close(fig)

