import pandas as pd


isu_dis = pd.read_json("08042021up.json")
print(isu_dis[isu_dis.НОМЕР_ПО_ПЛАНУ == None].ДИСЦИПЛИНА.unique())
