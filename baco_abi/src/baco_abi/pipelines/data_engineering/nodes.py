# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from typing import Any, Dict

import pandas as pd
import numpy as np
from scipy.stats import mode
from time import strptime

from sklearn.model_selection import train_test_split

def remove_spaces_lower(col_string):
    col_string = col_string.strip().lower()
    return col_string

def add_date_to_data(data: pd.DataFrame, year_col: str, month_col: str, date_name: str) -> pd.DataFrame:
    
    data_date = data.copy()
    data_date[date_name] = pd.to_datetime(data_date[year_col].astype(str) + data_date[month_col].map('{:02d}'.format), format="%Y%m")
    
    return data_date

def apply_remove_spaces_lower(data: pd.DataFrame, lst_cols: list):
    
    data_temp = data.copy()
    for col in lst_cols:
        data_temp[col] = data_temp[col].apply(remove_spaces_lower)
    
    return data_temp


def apply_remove_spaces_lower(data: pd.DataFrame, lst_cols: list):
    
    data_temp = data.copy()
    for col in lst_cols:
        data_temp[col] = data_temp[col].apply(remove_spaces_lower)
    
    return data_temp

counter = 1
marcas = {("marca_20", "cupo_3", "capacidadenvase_9"): "Prod1", 
        ("marca_16", "cupo_2", "capacidadenvase_10"): "Prod2",
        ("marca_9", "cupo_3", "capacidadenvase_12"): "Prod3",
        ("marca_38", "cupo_2", "capacidadenvase_10"): "ProdInno1",
        ("marca_39", "cupo_2", "capacidadenvase_10"): "ProdInno2"}

counter = 1
def create_key(row, marca_col, cupo_col, capacidad_col):
    global counter
    if (row[marca_col], row[cupo_col], row[capacidad_col]) in marcas:
        return marcas[(row[marca_col], row[cupo_col], row[capacidad_col])]
    else:
        name = "ProdOt{:d}".format(counter)
        counter += 1
        marcas[(row[marca_col], row[cupo_col], row[capacidad_col])] = name
        return name

def add_product_key(data: pd.DataFrame, marca_col, cupo_col, capacidad_col, product_col):

    data_temp = data.copy()
    data_temp[product_col] = data_temp.apply(create_key, marca_col = marca_col, cupo_col = cupo_col, capacidad_col = capacidad_col, axis=1)
    return data_temp

def create_data_full(data: pd.DataFrame, drop_cols_lst):

    data_temp = data.copy()
    data_temp = data_temp.drop(columns = drop_cols_lst)
    return data_temp


def get_product_df(df_full: pd.DataFrame, df_clientes: pd.DataFrame, product, target_dates, train_dates, gen_report=False):
    df_target = df_full[(df_full["Fecha"].isin(target_dates)) & (df_full["Producto"] == product) & (df_full["Volumen"] > 0)][["Cliente", "Producto", "Fecha"]]
    df_target[product] = np.where(df_target["Producto"] == product, 1, 0)
    df_target = df_target.groupby(["Cliente"]).agg("sum")
    df_target[product] = np.where(df_target[product] > 0, 1, 0)
    df_target = df_full[["Cliente"]].drop_duplicates().merge(df_target, on="Cliente", how="left").fillna(0).set_index("Cliente")
    
    print(df_full["Fecha"].max())
    df_data = df_full[df_full["Fecha"].isin(train_dates)]
    df_grouped = df_data.groupby(['Cliente', 'Producto']).agg("mean")
    print(df_data["Fecha"].max())
    print(df_grouped)
    
    df_cont = df_grouped.loc[(slice(None), slice(None)),].unstack()
    df_cont.columns = df_cont.columns.map('_'.join)
    
    df_prod = df_target.join(df_clientes.set_index("Cliente"), how="left").join(df_cont, how="left").fillna(0)
    df_prod = df_prod.drop(columns=["Regional2"])
    df_prod["Nevera"] = pd.Categorical(df_prod["Nevera"])
    
    if gen_report:
        profile = ProfileReport(df_prod, minimal=True)
        profile.to_file(fr"{cwd}/{product}.html")
    
    return df_prod

def get_final_test_df(df_full: pd.DataFrame, df_clientes: pd.DataFrame, product, prod_data):

    columns = list(set(prod_data.columns) - set([product]))

    df_grouped = df_full.groupby(['Cliente', 'Producto']).agg("mean")
    print(df_grouped)
    df_cont = df_grouped.loc[(slice(None), slice(None)),].unstack()
    df_cont.columns = df_cont.columns.map('_'.join)
    df_prod = df_clientes.set_index("Cliente").join(df_cont, how="left").fillna(0)
    df_prod = df_prod.drop(columns=["Regional2"])
    df_prod["Nevera"] = pd.Categorical(df_prod["Nevera"])
    return df_prod.loc[:, columns]

def get_product_df_lst(data_full: pd.DataFrame, data_clientes: pd.DataFrame, lst_products, date):

    lst_df_prods = []
    for product in lst_products:
        df_prod = get_product_df(data_full, data_clientes, product, date, gen_report=False)
        lst_df_prods.append(df_prod)

    return lst_df_prods


def split_data_prod(data_prod: pd.DataFrame, data_test: pd.DataFrame, product):
    data_temp = data_prod.copy()
    data_temp_pd = pd.get_dummies(data_temp)

    data_temp_c = data_temp_pd[data_temp_pd.index.isin(data_test["Cliente"])]

    data_true = data_temp_pd[data_temp_pd[product] == 1]
    data_false = data_temp_pd[data_temp_pd[product] == 0]

    data_false = data_false.sample(n=data_true.shape[0]*2)

    df_resampled = data_true.append(data_false).sample(frac = 1) 
    df_resampled

    X = df_resampled.drop(columns=[product])
    Y = df_resampled[[product]]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return dict(
        train_x=X_train,
        train_y=y_train,
        test_x=X_test,
        test_y=y_test,
        data_c=data_temp_c
    )