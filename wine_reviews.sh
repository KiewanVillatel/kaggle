#!/bin/bash

MLFLOW_TRACKING_URI=http://localhost:5000 mlflow run -e wine_reviews . -P min_province=50 -P min_designation=50 -P min_variety=50 -P min_region_1=50 -P min_region_2=50 -P min_winery=0
