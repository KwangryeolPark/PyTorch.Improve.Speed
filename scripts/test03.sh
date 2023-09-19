#!/bin/bash

for i in {1..10};
do
    python test03.py --torch_no_grad True
done

for i in {1..10};
do
    python test03.py --torch_no_grad False
done
