TS=$(date +%F_%H-%M-%S)
python -u run_al_staged.py > "loop_${TS}.log" 2> "loop_${TS}.err"