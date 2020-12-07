#!/bin/bash

mlflow server --backend-store-uri postgresql://postgres:root@localhost:5432/mlflow --default-artifact-root ./mlruns --host 0.0.0.0
