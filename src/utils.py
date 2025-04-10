import os

# Muxi Base Directory 由于不同机器的挂载路径不同，对应的个人目录路径也不同
BASE_DIR = '/mnt/public/data/lh/chy'
# BASE_DIR = '/mnt/data/lh/chy'

# 虽然 os.path.join 是最为标准的路径拼接方式，但是不够直观，可以减少使用
# 如果使用硬编码的路径，对应的路径最后不要添加 /

# BRACE Dataset Metadata Path
META_DIR = f'{BASE_DIR}/BRACE_Eval/BRACE'

# Qwen Pretrained Model Path
MODEL_DIR = f'{BASE_DIR}/models'

# LALM Directory
LALM_DIR = f'{BASE_DIR}/BRACE_Eval/LALM'

# Project Directory
PROJECT_DIR = f'{BASE_DIR}/BRACE_Eval'