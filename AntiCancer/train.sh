#!/bin/bash
program=train
dir=./

start=1
num_random=20
random=0
idx_d=2
idx_c=2
dropout=0.0
device=cuda:0
log=./Log/
chkpt_dir=./Checkpoint/

for ((random=1; random<=num_random; random++))
do 
    name=${program}.random.${random}.num_random.${num_random}.idx_d.${idx_d}.idx_c.${idx_c}.dropout.${dropout}.log.${log}.chkpt_dir.$chkpt_dir
    python3 $program.py --random=$random --num_random=$num_random --idx_d=$idx_d --idx_c=$idx_c --dropout=$dropout --device=$device --log=$log --chkpt_dir=$chkpt_dir
done