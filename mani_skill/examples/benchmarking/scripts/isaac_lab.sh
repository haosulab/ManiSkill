# Benchmark state FPS
for n in 4 16 32 64 128 256 512 1024 2048 4096 8192 16384
do
  python isaac_lab_gpu_sim.py \
      --task "Isaac-Cartpole-Direct-Benchmark-v0" \
      --num-envs $n --obs-mode state \
      --headless --save-results
done

# Benchmark number of cameras
for num_cams in 2 3 4
do
  for n in 4 16 32 64 128 256
  do
    for cam_size in 80 128 160 224 256 512
    do
      python isaac_lab_gpu_sim.py \
        --task "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0" \
        --num-envs $n --obs-mode rgb \
        --num-cams=$num_cams --cam-width=$cam_size --cam-height=$cam_size \
        --enable_cameras --headless --save-results
    done
  done
done

# Benchmark different number of environments and camera sizes
for obs_mode in rgb rgb+depth depth
do
  for cam_size in 80 128 160 224 256 512
  do
    for n in 4 16 32 64 128 256 512 1024
    do
      python isaac_lab_gpu_sim.py \
        --task "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0" \
        --num-envs $n --obs-mode $obs_mode \
        --num-cams=1 --cam-width=$cam_size --cam-height=$cam_size \
        --enable_cameras --headless --save-results
    done
  done
done

# Benchmark high number of environments and small camera sizes
for obs_mode in rgb rgb+depth
do
  for n in 2048 4096
  do
    for cam_size in 80 128
    do
      python isaac_lab_gpu_sim.py \
        --task "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0" \
        --num-envs $n --obs-mode $obs_mode \
        --num-cams=1 --cam-width=$cam_size --cam-height=$cam_size \
        --enable_cameras --headless --save-results
    done
  done
done

# benchmark realistic settings
# droid dataset
for n in 4 16 32 64 128 256
do
  python isaac_lab_gpu_sim.py \
    --task "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0" \
    --num-envs $n --obs-mode depth \
    --num-cams=3 --cam-width=320 --cam-height=180 \
    --enable_cameras --headless --save-results
done

# rt dataset
for n in 4 16 32 64 128
do
  python isaac_lab_gpu_sim.py \
    --task "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0" \
    --num-envs $n --obs-mode depth \
    --num-cams=1 --cam-width=640 --cam-height=480 \
    --enable_cameras --headless --save-results
done

for obs_mode in depth rgb
do
  for n in 4 16 32 64 128
  do
    python isaac_lab_gpu_sim.py \
      --task "Isaac-Franka-Direct-Benchmark-v0" \
      --num-envs $n --obs-mode $obs_mode \
      --num-cams=1 --cam-width=640 --cam-height=480 \
      --enable_cameras --headless --save-results
  done
done

for obs_mode in rgb depth
do
  for n in 4 16 32 64 128 256
  do
    python isaac_lab_gpu_sim.py \
      --task "Isaac-Franka-Direct-Benchmark-v0" \
      --num-envs $n --obs-mode $obs_mode \
      --num-cams=3 --cam-width=320 --cam-height=180 \
      --enable_cameras --headless --save-results
  done
done