import pandas as pd

new_cols = [
    '2024-05-s1',
    '2024-05-s4',
    '2024-06-s4',
    '2024-06-s8',
    #'2024-07-s1',
    '2024-07-s4',
    '2024-08-s1',
    '2024-08-s4',
    ]

def make_contig_csv():
    path = "../data/upper_tax_group-sample.txt"

    with open(path, "r") as f:
        data = f.read()

    a = data.split("\n")
    b = [v.split("\t") for v in a][:-1]

    cols = [ "Taxonomic_name", "Conventional_name"] + b[0][1:]

    df_upper = pd.DataFrame(
        b[1:],
        columns=cols
    ).set_index("Conventional_name").drop("Taxonomic_name", axis=1)[new_cols].astype("float")
    df_upper = df_upper / df_upper.sum(axis=0)
    return df_upper

from scipy.stats import lognorm
def add_noise(df_upper):
    df_noise = lognorm.rvs(s=2, loc=0, scale=1e-5, size=df_upper.shape, random_state=0)
    df_upper_noise = df_upper + df_noise
    return df_upper_noise


