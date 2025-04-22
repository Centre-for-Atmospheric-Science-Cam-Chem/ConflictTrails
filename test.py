import pandas as pd
df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
print(df.get(0))