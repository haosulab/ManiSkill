# Benchmark camera modalities
# for obs_mode in rgb depth rgbd
# do
#   for n in 4 16 32 64 128 256 512 1024
#   do
#     python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
#       -n=$n -o=rgb --num-cams=$num_cams --cam-width=128 --cam-height=128
#   done
# done

# Benchmark state FPS
for n in 4 16 32 64 128 256 512 1024 2048 4096 8192 16384
do
  python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
    -n=$n -o=state --save-results
done

# Benchmark number of cameras
for num_cams in {1..6}
do
  for n in 4 16 32 64 128 256 512 1024
  do
    python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
      -n=$n -o=rgb --num-cams=$num_cams --cam-width=128 --cam-height=128 --save-results
  done
done

# Benchmark different number of environments and camera sizes
for obs_mode in rgb rgb+depth
do
  for n in 4 16 32 64 128 256 512 1024
  do
    for cam_size in 80 128 160 224 256 512
    do
      python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
        -n=$n -o=$obs_mode --num-cams=1 --cam-width=$cam_size --cam-height=$cam_size --save-results
    done
  done
done

# Benchmark different number of environments and default maniskill environments
for env_id in "PickCube-v1" "OpenCabinetDrawer-v1"
do
  for n in 4 16 32 64 128 256 512 1024
  do
    python gpu_sim.py -e $env_id \
      -n=$n -o=rgb --num-cams=1 --cam-width=128 --cam-height=128 --sim-freq=100 --control-freq=50 --save-results
  done
done