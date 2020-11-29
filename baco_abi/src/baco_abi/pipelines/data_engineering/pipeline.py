# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                add_date_to_data,
                ["ventas_data", "params:year_col", "params:month_col", "params:date_name"] ,
                "ventas_date_data"

            ),
            node(
                apply_remove_spaces_lower,
                ["ventas_date_data", "params:list_cols_spaces_lower"],
                "ventas_date_lower"
            ),

            node(
                add_product_key,
                ["ventas_date_lower", "params:marca_col", "params:cupo_col",  "params:capacidad_col", "params:product_key"],
                "ventas_date_lower_prod"
            ),

            node(
                create_data_full,
                ["ventas_date_lower_prod", "params:drop_cols_lst"],
                "ventas_full"
            ),

            node(
                get_product_df,
                ["ventas_full", "clientes_data", "params:product1", "params:dates_inno", "params:dates2_inno"],
                "prod1_data",
                name = "get_product1_df",
            ),

            node(
                get_product_df,
                ["ventas_full", "clientes_data", "params:product2", "params:dates_prod2", "params:dates2_prod2"],
                "prod2_data",
                name = "get_product2_df",
            ),

            node(
                get_product_df,
                ["ventas_full", "clientes_data", "params:product3", "params:dates", "params:dates2"],
                "prod3_data",
                name = "get_product3_df",
            ),

            node(
                get_product_df,
                ["ventas_full", "clientes_data", "params:product_inno1", "params:dates_inno", "params:dates2_inno"],
                "prod_inno1_data",
                name = "get_product_inno1_df",
            ),

            node(
                get_product_df,
                ["ventas_full", "clientes_data", "params:product_inno2", "params:dates_inno", "params:dates2_inno"],
                "prod_inno2_data",
                name = "get_product_inno2_df",
            ),

            node(
                get_final_test_df,
                ["ventas_full", "clientes_data", "params:product1", "prod1_data"],
                "prod1_data_t",
                name = "get_final_test1_df",
            ),

            node(
                get_final_test_df,
                ["ventas_full", "clientes_data", "params:product2", "prod2_data"],
                "prod2_data_t",
                name = "get_final_test2_df",
            ),
            
            node(
                get_final_test_df,
                ["ventas_full", "clientes_data", "params:product3", "prod3_data"],
                "prod3_data_t",
                name = "get_final_test3_df",
            ),

            node(
                get_final_test_df,
                ["ventas_full", "clientes_data", "params:product_inno1", "prod_inno1_data"],
                "prod_inno1_data_t",
                name = "get_final_test_inno1_df",
            ),

            node(
                get_final_test_df,
                ["ventas_full", "clientes_data", "params:product_inno2", "prod_inno2_data"],
                "prod_inno2_data_t",
                name = "get_final_test_inno2_df",
            ),
            #node(
            #    get_product_df_lst,
            #    ["ventas_full", "clientes_data", "params:product_lst", "params:date"],
            #    ["prod1_data", "prod2_data", "prod3_data", "prod_inno1_data", "prod_inno2_data"],
            #    name = "get_product_df_lst",
            #),

            node(
                split_data_prod,
                ["prod1_data", "test_data", "params:product1"],
            dict(
                    train_x="prod1_train_x",
                    train_y="prod1_train_y",
                    test_x="prod1_test_x",
                    test_y="prod1_test_y",
                    data_c="prod1_data_c"),
                name = "split_data_prod1"
            ),

            #node(
            #    split_data_prod,
            #    ["prod1_data", "test_data", "params:product2"],
            #dict(
            #        train_x="prod2_train_x",
            #        train_y="prod2_train_y",
            #        test_x="prod2_test_x",
            #        test_y="prod2_test_y",
            #        data_c="prod2_data_c"),
            #    name = "split_data_prod2"
            #)
        ]
    )
