# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from kedro.pipeline import Pipeline, node

from .nodes import train_rfc, random_search_rfc, train_svmc, train_xgboost


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(train_rfc, ["prod1_train_x", "prod1_train_y", "prod1_test_x", "prod1_test_y", "prod1_data_c", "test_data"], None
            ),

            node(random_search_rfc, ["prod1_train_x", "prod1_train_y", "prod1_test_x", "prod1_test_y"], None
            ),

            node(train_svmc, ["prod1_train_x", "prod1_train_y", "prod1_test_x", "prod1_test_y"], None,
            name = "prod1_train_svmc"
            ),

            node(train_xgboost, ["prod1_train_x", "prod1_train_y", "prod1_test_x", "prod1_test_y", "prod1_data_c"], None,
            name = "prod1_train_xgboost"
            ),
        ]
    )
