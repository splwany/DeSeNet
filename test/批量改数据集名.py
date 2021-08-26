# 使用 python ./test/批量改数据名.py --name-name blind --dataset-dir ./datasets/blind/

from pathlib import Path
import argparse


def main(cfg):
    old_names = tuple(cfg.old_names)
    new_name = cfg.new_name
    dataset_dir = Path(cfg.dataset_dir)
    for p in dataset_dir.rglob('**/*.*'):
        if p.name.startswith(old_names):
            print(p)
            replaced_name = p.name.replace(p.name.split('_')[0], new_name)
            p.rename(p.with_name(replaced_name))

    for p in dataset_dir.rglob('**/*'):
        if p.is_dir() and p.name.startswith(old_names):
            print(p)
            replaced_name = p.name.replace(p.name.split('_')[0], new_name)
            p.rename(p.with_name(replaced_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-names', nargs='+', type=str, default=['sidewalk', 'step'], help='原数据集的名字')
    parser.add_argument('--new-name', type=str, required=True, help='新数据集的名字')
    parser.add_argument('--dataset-dir', type=str, required=True, help='要遍历的根目录')
    args = parser.parse_args()
    
    main(args)