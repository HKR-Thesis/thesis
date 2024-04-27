#! /bin/bash
# adisve - 12-04-2024

declare -a on_gpu=("dql" "dql_target")
declare -a on_cpu=("classic" "numba" "dql" "dql_target")

# uses tensorflow with gpu support, expecting tensorflow-2.7.0+nv22.1 for jetson nano
for train_type in "${on_gpu[@]}"
do
   python -m src.benchmark --train $train_type
done

# uses tensorflow with no GPU support
for train_type in "${on_cpu[@]}"
do
   python3.10 -m src.benchmark --train $train_type
done
