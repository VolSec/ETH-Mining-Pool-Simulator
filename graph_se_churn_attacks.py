import sys 
import os
from volsec_tools.graphing.generic_graph_util import graph_cdf, graph_x_vs_y
import matplotlib.pyplot as plt

def main(argv):
    if len(argv) < 2:
        print("usage: sim_output_file(s)")
        sys.exit(1)

    for i, f in enumerate(argv):
        if i != 0:
            honest_miners, mal_miners, all_miners = process_file(f)
    
    graph_stuff(honest_miners, mal_miners, all_miners)

def graph_stuff(honest_miners, mal_miners, all_miners):
    target = None
    reg_tups = []
    mal_m = mal_miners[0]
    target = mal_m.target
    target_tup = None
    low = None
    high = None
    mal_tup = (mal_m.mid, mal_m.avg_block_cost)
 
    for m in honest_miners:
        if high is None or m.avg_block_cost > high: 
            high = m.avg_block_cost
        if  low is None or m.avg_block_cost < low:
            low = m.avg_block_cost
        if m.mid != target:
            if m.avg_block_cost > 0:
                reg_tups.append((m.mid, m.avg_block_cost))
        else:
            target_tup = ((m.mid, m.avg_block_cost))
    
    plt.scatter([tup[0] for tup in reg_tups], [tup[1] for tup in reg_tups])
    if mal_tup[1] > 0.0:
        plt.scatter(mal_tup[0], mal_tup[1], c ='r')
    if target_tup[1] > 0.0:
        plt.scatter(target_tup[0], target_tup[1], c='k')
    
    print(low, high)
    #axes = plt.gca()
    #axes.set_ylim([ymin,ymax])
    plt.show()

def process_file(f):
    TYPE_LOC = 0
    ID_LOC = 1
    HR_LOC = 2
    SHR_LOC = 3
    BW_LOC = 4
    ABC_LOC = 5
    TGT_LOC = 6
    RAT_LOC = 7

    honest_miners = []
    mal_miners = []
    all_miners = []

    with open(f, 'r') as ftr:
        for line in ftr:
            tokens = line.strip().split(',')
            mt = tokens[TYPE_LOC]
            mid = int(tokens[ID_LOC])
            hr = float(tokens[HR_LOC])
            bw = int(tokens[BW_LOC])
            abc = float(tokens[ABC_LOC])
            if mt == 'M':
                tgt_id = int(tokens[TGT_LOC])
                ratio = float(tokens[RAT_LOC])
                m = Miner(mt, mid, hr, bw, abc, tgt_id, ratio)
                mal_miners.append(m)
                all_miners.append(m)
            else:
                m = Miner(mt, mid, hr, bw, abc)
                honest_miners.append(m)
                all_miners.append(m)

    return (honest_miners, mal_miners, all_miners)
    
class Miner:
    def __init__(self, mt, mid, hash_rate, blocks_won, avg_block_cost,
                 target=None, ratio=None):
        self.mt = mt
        self.mid = mid
        self.hr = hash_rate
        self.bw = blocks_won
        self.avg_block_cost = avg_block_cost
        self.target = target
        self.ratio = ratio
    

if __name__ == "__main__":
    main(sys.argv)
