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
for n in 4 16 32 64 128 256 512 1024 2048 4096 8192
do
  python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
    -n=$n -o=state
done

# Benchmark number of cameras
for num_cams in {1..6}
do
  for n in 4 16 32 64 128 256 512 1024
  do
    python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
      -n=$n -o=rgb --num-cams=$num_cams --cam-width=128 --cam-height=128
  done
done

# Benchmark different number of environments and camera sizes
for n in 4 16 32 64 128 256 512 1024
do
  for cam_size in 80 128 160 224 256 512
  do
    python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
      -n=$n -o=rgb --num-cams=1 --cam-width=$cam_size --cam-height=$cam_size
  done
done