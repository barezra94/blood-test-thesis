import pandas as pd
from datetime import datetime, timedelta

df_uk_53 = pd.read_csv(
    "/Users/shaygafniel/Desktop/Thesis/NewImplementation/blood-test-thesis/docs/data_with_53.csv"
)

df_admission = pd.read_csv(
    "/Users/shaygafniel/Desktop/Thesis/NewImplementation/blood-test-thesis/docs/hesin.txt",
    delimiter="\t",
)
df_diag = pd.read_csv(
    "/Users/shaygafniel/Desktop/Thesis/NewImplementation/blood-test-thesis/docs/hesin_diag.txt",
    delimiter="\t",
)

df_admission = df_admission.drop_duplicates(subset="eid")
df_diag = df_diag.drop_duplicates(subset="eid")

print(df_diag.shape)
print(df_admission.shape)

df_uk_filtered = df_uk_53[["eid", "53-0.0"]]

print(df_uk_filtered)

df_join = pd.merge(df_uk_filtered, df_admission, on="eid", how="left")
print(df_join.shape)
df_join = pd.merge(df_join, df_diag, on="eid", how="left")
print(df_join.shape)

df_final = df_join[["eid", "53-0.0", "admidate"]]
df_final = df_final.dropna()

df_final["admidate"] = df_final["admidate"].apply(
    lambda x: datetime.strptime(x, "%d/%m/%Y")
)
df_final["53-0.0"] = df_final["53-0.0"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d")
)

df_final["diff"] = df_final["53-0.0"] - df_final["admidate"]

df = df_final.loc[df_final["diff"] <= timedelta(days=0)]

df.to_csv(
    "/Users/shaygafniel/Desktop/Thesis/NewImplementation/blood-test-thesis/docs/admissions.csv"
)

print(df)
