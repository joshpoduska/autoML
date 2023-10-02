



import h2o
import pandas as pd
import sys

print(sys.argv[1])
print(sys.argv[2])

data_dir = sys.argv[1]
target_col = sys.argv[2]
h2o.init()

loaded_data = pd.read_csv(data_dir)


for col in loaded_data.columns:  # Iterate over chosen columns
    loaded_data[col] = pd.to_numeric(loaded_data[col], errors='coerce')

#drop nulls
loaded_data.dropna(inplace=True)

hf = h2o.H2OFrame(loaded_data)

x = hf.columns
y = target_col
x.remove(y)

hf[y] = hf[y].asfactor()

aml = h2o.automl.H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=hf)


lb = aml.leaderboard
lb.head(rows=lb.nrows)

exm = aml.explain(hf)

aml.leader







