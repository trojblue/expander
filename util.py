from typing import List
import os
import csv
from tqdm.auto import tqdm

import os
import csv


def combine_files(base_dir:str, root_dirs:List[str]):
    """

    :param base_dir:  "D:\Andrew\Pictures\==train\BLIP"
    :param root_dirs: [ "aesthetics.1280.cmb.m4", ... ]
    :return: 创建csv文件
    :rtype:
    """

    combined_data = []
    for root_dir in root_dirs:
        caption_dir = os.path.join(base_dir, root_dir)  # caption: base_dir/subdir/*.txt; 每项一个txt
        tag_str_dir = os.path.join(caption_dir, 'txt')  # tag_str: base_dir/subdir/txt/*.txt; 每项一个txt

        for caption_file in tqdm(os.listdir(caption_dir)):
            if caption_file.endswith('.txt'):
                with open(os.path.join(caption_dir, caption_file), 'r') as f:
                    caption = f.read().strip()

                tag_str_file = os.path.join(tag_str_dir, caption_file)
                with open(tag_str_file, 'r') as f:
                    tag_str = f.read().strip()

                combined_data.append((caption, tag_str))

    output_file = os.path.join(base_dir, 'combined.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['caption', 'tag_str'])
        writer.writerows(combined_data)


if __name__ == '__main__':

    base_dir = "D:\Andrew\Pictures\==train\BLIP"
    dir_names = [
        "aesthetics.1280.cmb.m4",
        "px_rank_m_2022-ALL",
        "px_rank_m_2021-ALL",
    ]

    combine_files(base_dir, dir_names)