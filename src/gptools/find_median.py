import os
from typing import List,Dict

"""
Given a folder, organized like so
COIL20
    -> run_1
        -> *.tree
    -> run_2
        -> *.tree
    ...
    -> run_n
        -> *.tree
Find the median complexity tree in each run.
Then find the median complexity tree out of those medians
This one can be said to be representative for discussion purposes
"""

def find_median(path: str) -> str:
    """
    Arguments:
        path: full path to our folder we will crawl
    returns:
        filename: a str with full path to our median .tree filename
    """
    medians: Dict={}
    folder_dict: Dict={}
    for folder in os.listdir(path):
        print(f"crawling {folder}")
        dict_cur: Dict = {}
        files: List[str] = os.listdir(f'{path}/{folder}')
        files=[f for f in files if f.endswith('.tree')]
        for file in files:
            complexity: int = int(file.split('-')[2].split('.')[0])
            dict_cur[file] = int(complexity)
        folder_dict[folder]=dict_cur
    # now find medians in each dict folder
    for d in folder_dict.keys():
        d_cur = folder_dict[d]
        sorted_d: Dict = dict(sorted(d_cur.items(), key=lambda item: item[1]))
        print(sorted_d)
        print("\n"+"~"*40)
        median_key: str = list(sorted_d.keys())[int(len(sorted_d)/2)]
        complexity = int(median_key.split('-')[2].split('.')[0])
        medians[f"{d}/{median_key}"] = complexity
    # now find median of medians
    print(medians)
    sorted_medians: Dict = dict(sorted(medians.items(), key=lambda item: item[1]))
    print(sorted_medians)
    median_key = list(sorted_medians.keys())[int(len(sorted_medians)/2)]
    print(median_key)
    return median_key

if __name__=="__main__":
    find_median("COIL20_pt2")
