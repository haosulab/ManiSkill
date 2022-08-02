FROM haosulab/maniskill2022-challenge:latest

# Install additional python packages you need
RUN conda install pytorch cudatoolkit=11.3 -c pytorch && pip install pytransform3d

# Copy your codes (including user_solution.py) and model weights
COPY ${YOUR_CODES_AND_WEIGHTS} /root/
ENV PYTHONPATH ${YOUR_CODES_AND_WEIGHTS}:$PYTHONPATH