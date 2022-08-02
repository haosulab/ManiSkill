FROM haosulab/maniskill2022-challenge:latest

# Install additional python packages you need 
# (note that we don't recommend installing pytorch==1.12.0 due to a known issue; 
#  when a newer version of pytorch comes out, you may choose to install it)
RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch && pip install pytransform3d

# Copy your codes (including user_solution.py) and model weights
COPY ${YOUR_CODES_AND_WEIGHTS} /root/
ENV PYTHONPATH ${YOUR_CODES_AND_WEIGHTS}:$PYTHONPATH
