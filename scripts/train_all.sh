#! /bin/bash
# adisve - 12-04-2024

declare -a arr=("classic" "numba" "dql" "dql_target")

# This really assumes that none of the train types will fail
for train_type in "${arr[@]}"
do
   python -m src.benchmark --train $train_type
done
