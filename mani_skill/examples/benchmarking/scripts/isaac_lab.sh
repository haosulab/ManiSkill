# Benchmark state FPS
for n in 4 16 32 64 128 256 512 1024 2048 4096 8192 16384
do
  isaaclab -p isaac_lab_gpu_sim.py \
      --task "Isaac-Cartpole-Direct-Benchmark-v0" \
      --num-envs $n --obs-mode state \
      --headless --save-results
done

# Benchmark number of cameras
# isaac lab does not support this

# Benchmark different number of environments and camera sizes
for obs_mode in rgb rgb+depth
do
  for n in 4 16 32 64 128 256 512 1024
  do
    for cam_size in 80 128 160 224 256 512
    do
      isaaclab -p isaac_lab_gpu_sim.py \
        --task "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0" \
        --num-envs $n --obs-mode $obs_mode \
        --num-cams=1 --cam-width=$cam_size --cam-height=$cam_size \
        --enable_cameras --headless --save-results
    done
  done
done