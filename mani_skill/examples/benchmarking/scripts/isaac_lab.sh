# Benchmark state FPS
for n in 4 16 32 64 128 256 512 1024 2048 4096 8192
do
  python isaac_lab_gpu_sim.py \
      --task "Isaac-Cartpole-Direct-Benchmark-v0" \
      --num_envs $n --obs_mode state \
      --headless
done

# Benchmark number of cameras
# isaac lab does not support this

# Benchmark different number of environments and camera sizes
for n in 4 16 32 64 128 256 512 1024 2048 4096
do
  for cam_size in 80 128 160 224 256 512
  do
    python isaac_lab_gpu_sim.py \
      --task "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0" \
      --num_envs $n --obs_mode rgb \
      --num-cams=1 --cam-width=$cam_size --cam-height=$cam_size \
      --enable_cameras --headless
  done
done