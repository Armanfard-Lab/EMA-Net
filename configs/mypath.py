import os
PROJECT_ROOT_DIR = os.path.abspath(os.curdir)

db_root = '/path/to/dataset/'

db_names = {'nyuv2': 'nyuv2', 'cityscapes': 'cityscapes'}
db_paths = {}
for database, db_pa in db_names.items():
    db_paths[database] = os.path.join(db_root, db_pa)