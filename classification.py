import pandas as pd

df = pd.read_excel("A0 strain.xlsx")

df = df.rename(columns={df.columns[0]: "time"})
df = df.set_index("time")

df_long = (
    df
    .reset_index()
    .melt(
        id_vars="time",
        var_name="sample",
        value_name="OD"
    )
)

# Group letter (N, M, S)
df_long["group_letter"] = df_long["sample"].str.extract(r'^([NMS])')

# Patient number
df_long["patient"] = (
    df_long["sample"]
    .str.extract(r'[NMS](\d+)')
    .astype(int)
)

# Replicate number
df_long["repetition"] = (
    df_long["sample"]
    .str.extract(r'Replicate\s*(\d)')
    .astype(int)
)

group_map = {"N": 0, "M": 1, "S": 2}
df_long["group"] = df_long["group_letter"].map(group_map)

df_cnn = (
    df_long
    .pivot_table(
        index=["patient", "repetition", "group"],
        columns="time",
        values="OD"
    )
    .reset_index()
)

meta_cols = ["patient", "repetition", "group"]

time_cols = sorted(
    [c for c in df_cnn.columns if c not in meta_cols],
    key=lambda x: float(x)
)

df_cnn = df_cnn[meta_cols + time_cols]

print(df_long)