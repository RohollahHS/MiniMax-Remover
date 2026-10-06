import os

if 'SCRATCH' in os.environ:
    BASE_DIR = f"{os.getenv('SCRATCH')}/projects/MiniMax-Remover"
else:
    BASE_DIR = '.'