import sys 
import os
from glob import glob
import natsort 
import pickle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse 
import pandas as pd

from animation import * 
from matplotlib.animation import FuncAnimation, PillowWriter 

parser = argparse.ArgumentParser()
# data path
parser.add_argument('--labels_path', type= str, required= True, help= 'a path to labels, ex) ./heteroNRI/test/labels')
parser.add_argument('--preds_path', type= str, required= True, help= 'a path to preictions, ex) ./heteroNRI/test/predictions')
parser.add_argument('--graphs_path', type= str, required= True, help= 'a path to graphs ex) ./heteroNRI/test/graphs')
parser.add_argument('--enb_name', type= str, required= True, help= 'the name of eNB')
parser.add_argument('--cache', type= str, help= 'a cache file containing max and min of columns')
# output path
parser.add_argument('--animation_path', type= str, default= './',
                    help= 'a path to save the animations')

args = parser.parse_args()
print(args) 

print("loading the cache file")
try: 
    with open(args.cache, 'rb') as f: 
        cache = pickle.load(f) 
except: 
    cache = None
    print("loading a cache file failed...")
    print("max and min values will be caculated...")

if not os.path.exists(args.animation_path): 
    print("Making a path to save the model...")
    os.makedirs(args.animation_path, exist_ok= True)
else: 
    print(f"The path {args.animation_path} already exists, skip making the path...")

def main(args): 
    graphs_files = glob(os.path.join(args.graphs_path, f'{args.enb_name}/*.csv')) # csv files
    print(f'the number of graphs in the path: {args.graphs_path} = {len(graphs_files)}')
    graphs_files = natsort.natsorted(graphs_files,reverse=False) # sort by the number of files

    labels_file = os.path.join(args.labels_path, f'labels_{args.enb_name}.csv')
    preds_file = os.path.join(args.preds_path, f'predictions_{args.enb_name}.csv')

    labels = pd.read_csv(labels_file)
    preds = pd.read_csv(preds_file)
    graphs = [pd.read_csv(g, index_col= 0) for g in graphs_files]

    ani = SubplotAnimation(enb_num= args.enb_name, preds= preds, labels= labels, graphs = graphs, num_obs= len(graphs_files), cache= cache)
    print(f'saving the animation in path {args.animation_path}....')

    ani.save(os.path.join(args.animation_path, f'test_toy_{args.enb_name}.gif'), dpi=300, writer=PillowWriter(fps=3.), savefig_kwargs={'facecolor':'white'})

if __name__ == '__main__':
    main(args)