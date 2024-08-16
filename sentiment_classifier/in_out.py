from typing import List
import typing

# this function reads a .txt or similar
# it is meant to work with on a document-per-line basis as in the sample data files provided
def read_txt(path: str) -> List[str]:
    with open(path, 'r') as f:
        text = f.read()
        documents = text.split('\n')
        return documents

# generates path for the output in the same directory as the input
# the output is named output.txt
def out_path(path: str) -> str:
    path = path.split('/')
    path[-1] = 'output.txt'
    output_path = '/'.join(path)
    return output_path 

# generates the output file
# prints a message when finished -> 'Success!'
def write_txt(out_path: str, prediction_labels: List[str]):
    with open(out_path, 'w') as f:
        for label in prediction_labels:
            f.write(label + '\n')
        print('Success!')
