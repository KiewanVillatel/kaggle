#!/bin/bash

MLFLOW_TRACKING_URI=http://localhost:5000 mlflow run -e wine_reviews . -P seed=0 -P min_province=10 -P min_designation=50 -P min_variety=50 -P min_region_1=50 -P min_region_2=50 -P min_winery=5 -P min_df=0.0001 -P max_df=0.001
