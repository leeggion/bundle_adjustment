#!/bin/bash

IMAGES="../test_images/test5"
OUT="../data/test5.txt"

g++ -std=c++17 make_bal.cpp -o make_bal `pkg-config --cflags --libs opencv4`
mkdir vis
rm -rf vis/*
./make_bal "$IMAGES" "$OUT"
