# analyze_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/test2.csv", sep=',')
print("Data Distribution Analysis:")
print(f"Total samples: {len(df)}")
print(f"Zero cases: {(df['Cases'] == 0).sum()} ({(df['Cases'] == 0).mean():.1%})")
print(f"Cases distribution:")
print(df['Cases'].describe())

# Plot cases distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['Cases'], bins=20, alpha=0.7)
plt.title('Cases Distribution')
plt.xlabel('Cases')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
df.groupby('Puskesmas')['Cases'].mean().plot(kind='bar')
plt.title('Average Cases by Puskesmas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
