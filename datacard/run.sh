#!/bin/bash

echo "0LEP 3FJ R2"
./combine_cards.sh 0lep_3fj_r2

echo "0LEP 3FJ R3"
./combine_cards.sh 0lep_3fj_r3

echo "0LEP 3FJ R2 + R3"
./combine_cards.sh 0lep_3fj_r2 0lep_3fj_r3

echo "1LEP 1FJ R2"
./combine_cards.sh 1lep_1fj_r2

echo "1LEP 1FJ R3"
./combine_cards.sh 1lep_1fj_r3

echo "1LEP 1FJ R2 + R3"
./combine_cards.sh 1lep_1fj_r2 1lep_1fj_r3

echo "1LEP 2FJ R2"
./combine_cards.sh 1lep_2fj_r2

echo "1LEP 2FJ R3"
./combine_cards.sh 1lep_2fj_r3

echo "1LEP 2FJ R2 + R3"
./combine_cards.sh 1lep_2fj_r2 1lep_2fj_r3

echo "ALL"
./combine_cards.sh 0lep_3fj_r2 0lep_3fj_r3 1lep_1fj_r2 1lep_1fj_r3 1lep_2fj_r2 1lep_2fj_r3