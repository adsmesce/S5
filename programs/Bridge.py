import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stats
df = pd.read_csv("lois_continuous.csv")
df_new=df[df["ID"]=="S1"]
t=df_new["Temperature water continuous"]
o=df_new["Oxygen dissolved continuous"]
print(df.dtypes)
print("Mean temp =",stats.mean(t))
print("Median oxygen disolved =",stats.median(o))
t.hist()
plt.title("Lois continuous")
plt.xlabel("Distribution of temperature")
plt.ylabel("Time period of study")
plt.show()

