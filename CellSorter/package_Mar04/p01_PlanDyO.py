import pandas as pd

new_cols = [
    '2024-04-O-s1', '2024-04-O-s4', '2024-04-O-s8',
    '2024-05-O-s1', '2024-05-O-s4', '2024-06-O-s4', 
    '2024-06-O-s8',
    '2024-07-O-s1', '2024-07-O-s4', 
    '2024-08-O-s1', '2024-08-O-s4',
    '2024-11-O-s1', '2024-11-O-s4', '2024-11-O-s8',
    '2024-12-O-s1', 
    '2024-07-H-hm', '2024-07-H-mm', '2024-07-H-og',
    '2024-09-H-hm', '2024-09-H-mm', '2024-09-H-og', 
    '2024-11-H-hm', '2024-11-H-mm', '2024-11-H-og', 
    '2024-12-H-mm', 
    ]

def make_contig_csv(path = "../data/Kraken.upper_group-sample.txt", is_filtered=False):
    # path = "../data/Kraken.upper_group-sample.txt"

    with open(path, "r") as f:
        data = f.read()

    a = data.split("\n")
    b = [v.split("\t") for v in a][:-1]

    cols = [ "Taxonomic_name", "Conventional_name"] + b[0][1:]

    df_upper = pd.DataFrame(
        b[1:],
        columns=cols
    ).set_index("Conventional_name").drop("Taxonomic_name", axis=1).astype("float")# [new_cols]

    if is_filtered:
        df_upper = df_upper[new_cols]
    
    df_upper = df_upper / df_upper.sum(axis=0)
    return df_upper

from scipy.stats import lognorm
def add_noise(df_upper):
    df_noise = lognorm.rvs(s=2, loc=0, scale=1e-5, size=df_upper.shape, random_state=0)
    df_upper_noise = df_upper + df_noise
    return df_upper_noise


