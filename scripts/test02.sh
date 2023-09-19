#!/bin/bash

for j in {1..10};
do
    for i in {1..8};
    do
        python test02.py --num_workers $i --pin_memory 1
    done
done


for j in {1..10};
do
    for i in {1..8};
    do
        python test02.py --num_workers $i --pin_memory 0
    done
done

