FROM haosulab/mani-skill2:latest

# Install additional python packages you need
RUN pip install torch==1.12.1

# Copy your codes (including user_solution.py) and model weights
COPY ./ /root/
ENV PYTHONPATH ./:$PYTHONPATH
