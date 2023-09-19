#!/bin/bash

for i in {1..10};
do
    python test01.py --biased biased
done

for i in {1..10};
do
    python test01.py --biased unbiased
done
