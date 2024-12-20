# State based benchmark
for n in 16 32
do 
  python genesis_gpu_sim.py -e "FrankaMoveBenchmark-v1" \
    -n=$n -o=state --sim-freq=100 --control-freq=50 --save-results benchmark_results/genesis.csv
done