#!/bin/bash

screen -S sim0 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.80
sleep 2
screen -S sim1 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.81
sleep 2
screen -S sim2 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.82
sleep 2
screen -S sim3 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.83
sleep 2
screen -S sim4 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.84
sleep 2
screen -S sim5 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.85
sleep 2
screen -S sim6 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.86
sleep 2
screen -S sim7 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.87
sleep 2
screen -S sim8 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.88
sleep 2
screen -S sim9 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.89
sleep 2
screen -S sim10 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.90
sleep 2
screen -S sim11 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.91
sleep 2
screen -S sim12 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.92
sleep 2
screen -S sim13 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.93
sleep 2
screen -S sim14 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.94
sleep 2
screen -S sim15 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.95
sleep 2
screen -S sim16 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.96
sleep 2
screen -S sim17 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.97
sleep 2
screen -S sim18 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.98
sleep 2
screen -S sim19 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 1 --attack_params=0.99
sleep 2
screen -S sim20 -dm python3 run_sim.py 99 1 ../data/ethpool_miners_27_08_22_07_03.txt 10200 1 7 COST 11 --attack_params=1.0
