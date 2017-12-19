'''
Mining Pool for solo block crypto pool attack
'''

import math
import random
import logging
import datetime
import requests
from tabulate import tabulate
import numpy as np
import volsec_tools
from aminingpoolsimulator.miner import HonestMiner, CostMaliciousMiner, SpiteMaliciousMiner, ScorchedEarthMiner

class MiningPool:
    '''
    Mining Pool class that holds all of the miners associated with the pool,
    and runs the "rounds" that the miners all contribute to, as well as
    determines when the next block is found
    '''

    def __init__(self, num_honest_miners, num_malicious_miners,
                 hash_rates_file, round_length, num_blocks_to_sim,
                 shares_per_second=1, attack_type='COST', seed=None, attack_params=None, file_name=None):
        self.logger = logging.getLogger(__name__)
        self.num_honest_miners = num_honest_miners
        self.num_malicious_miners = num_malicious_miners
        self.num_blocks_to_sim = num_blocks_to_sim
        self.round_length = round_length
        self.shares_per_second = shares_per_second / 1000
        self.attack_type = attack_type
        self.cost_history = []
        self.diff_history = []
        self.attack_params = attack_params
        self.file_name = file_name

        self.seed = seed

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.load_hash_rates(hash_rates_file)
        self.get_network_statistics()
        self.create_miners()
        self.calculate_total_hash_rate()
        self.determine_avg_block_time()
        self.determine_rounds_with_found_blocks()


    def get_network_statistics(self):
        '''
        Grab network statistics from the pool api, so we can find out
        the difficutly of the network, the network hashrate, and the avg block
        time
        '''

        network_stats = requests.get('http://api.ethpool.org/networkStats').json()
        if network_stats['status'] != 'OK':
            print("network status for ethpool is down, need manual input for stats")
            self.network_hash_rate = input("Input network hashrate (h/s): ")
            self.network_difficulty = input("Input network difficulty: ")
            self.network_block_time = input("Input network average block time: ")
        else:
            self.network_hash_rate = network_stats['data']['hashrate']
            self.network_difficulty = network_stats['data']['difficulty']
            self.network_block_time = network_stats['data']['blockTime']

    def determine_avg_block_time(self):
        '''
        Determine the avg time it will take for our pool to mine a block,
        given the network hash rate, our pool hashrate, and the average
        block time of the network
        '''


        pool_hash_rate = 0
        for miner in self.all_miners:
            pool_hash_rate += miner.hash_rate

        self.logger.info('pool hash rate:     {} H/s'.format(pool_hash_rate))
        self.logger.info('network difficulty: {} H'.format(self.network_difficulty))

        self.avg_block_time = float(self.network_hash_rate / pool_hash_rate)
        self.avg_block_time = self.avg_block_time * self.network_block_time
        self.logger.info("avg block time for the pool: {0} seconds".format(self.avg_block_time))



    def hash_rate_to_share_rate(self, miner_hash_rate):
        #difficulty = math.floor(math.log(miner_hash_rate / self.shares_per_second) / math.log(2))
        #return difficulty * miner_hash_rate / math.pow(2, difficulty)
        return miner_hash_rate


    def get_shares_for_miner(self, miner_hash_rate, num_samples_needed):
        #self.logger.debug('hash rate: {} H/s'.format(miner_hash_rate))
        difficulty = math.floor(math.log(miner_hash_rate / self.shares_per_second) / math.log(2))
        #self.logger.debug('difficulty: {}'.format(difficulty))
        actual_probability = 1.0 / math.pow(2, difficulty)
        #self.logger.debug('P(share): {}'.format(actual_probability))

        # Returns a numpy 1-D array of length num_samples_needed
        found_shares_per_round = np.random.binomial(miner_hash_rate*self.round_length, actual_probability,
                                                    num_samples_needed)


        # We then need to multiply each value by the calculated difficulty
        shares_per_round = found_shares_per_round * math.pow(2,difficulty)

        #self.logger.debug('expected shares/s:     {}'.format(miner_hash_rate))
        #self.logger.debug('expected shares/round: {}'.format(miner_hash_rate/self.round_length))
        #self.logger.debug('effective share rate:  {} S/round'.format(sum(shares_per_round) / len(shares_per_round)))

        return shares_per_round


    def create_miners(self):
        '''
        Create the miners, keeping a global list, an honest list, and
        a  malicious list
        '''

        self.honest_miners = []
        self.malicious_miners = []
        self.all_miners = []
        # Populate honest miners
        for i in range(0, self.num_honest_miners):
            m = HonestMiner(i, random.choice(self.hash_rates),
                            self.round_length, self)
            self.all_miners.append(m)
            self.honest_miners.append(m)

        # Populate malicious miners
        for i in range(0, self.num_malicious_miners):
            if self.attack_type == 'COST':
                m = CostMaliciousMiner(i + self.num_honest_miners,
                                random.choice(self.hash_rates),
                                self.round_length, self.attack_params,
                                self)
            elif self.attack_type == 'SPITE':
                m = SpiteMaliciousMiner(i + self.num_honest_miners,
                                random.choice(self.hash_rates),
                                self.round_length, self.attack_params,
                                self)
            elif self.attack_type == 'SCORCHED_EARTH':
                m = ScorchedEarthMiner(i + self.num_honest_miners,
                                random.choice(self.hash_rates),
                                self.round_length, self.attack_params,
                                self)
            self.all_miners.append(m)
            self.malicious_miners.append(m)

    def load_hash_rates(self, hash_rates_file):
        '''
        Load in distribution of hashrates for our pool from file
        '''

        self.hash_rates = []
        with open(hash_rates_file, 'r') as ftr:
            for line in ftr:
                self.hash_rates.append(float(line.strip()))

    def calculate_total_hash_rate(self):
        total_hash_rate = 0

        #for hr in self.hash_rates:
            #total_hash_rate += hr
        for miner in self.all_miners:
            total_hash_rate += miner.hash_rate
        self.total_hash_rate = total_hash_rate


    def determine_rounds_with_found_blocks(self):
        '''
        So this takes the average amount of time our pool takes to find a block,
        the length of each round selected by the user, and the total number of
        blocks we want to simulate to build a list of rounds that we will
        simulate finding a block in ahead of time
        '''
        self.block_rounds = []
        p = float(self.round_length * self.total_hash_rate / self.network_difficulty)
        samples_in_seconds = np.random.geometric(p, size=self.num_blocks_to_sim)
        sample_sum = 0
        for sample in samples_in_seconds:
            sample_sum += sample
            self.block_rounds.append(sample_sum)

        mean = sample_sum / self.num_blocks_to_sim
        self.logger.info('pool hash rate:     {} H/s'.format(self.total_hash_rate))
        self.logger.info('network difficulty: {} H'.format(self.network_difficulty))
        expect = self.network_difficulty / self.total_hash_rate

        self.logger.debug('actual mean time to block: {} rounds ({} seconds)'.format(mean, mean*self.round_length))
        self.logger.debug('expect mean time to block: {} rounds ({} seconds)'.format(expect, expect*self.round_length))

    def find_all_blocks(self):
        '''
        Find all of the blocks that we are simulating, by running in time slices
        chosen by the user, held in round_length
        '''

        for miner in self.all_miners:
            miner.pre_simulation()

        self.num_blocks_found = 0
        round_num = 0
        block_rounds_index = 0
        last_round = self.block_rounds[-1]
        while self.num_blocks_found != self.num_blocks_to_sim:
            if round_num == self.block_rounds[block_rounds_index]:
                block_rounds_index += 1
                self.num_blocks_found += 1
                if self.num_blocks_found % 500 == 0:
                    self.logger.info("NUM BLOCKS FOUND: {0}".format(self.num_blocks_found))
                self.run_round(round_num, True)
            else:
                self.run_round(round_num, False)
            round_num += 1

        for miner in self.all_miners:
            miner.end_simulation()

        n_miners = len(self.all_miners)
        skip = n_miners*2
        shown = len(self.cost_history) - skip

        #volsec_tools.graphing.generic_graph_util.graph_cdf(self.cost_history[skip:], 'Distribution of block costs with {} miners over {} rounds'.format(n_miners, shown), 'Shares paid for block', 'Proportion of blocks', 'block_cost_cdf', False)

        #self.logger.info("Cost History: {}".format(self.cost_history))
        self.logger.debug("Number of rounds ran: {0}".format(round_num))


    def run_round(self, round_number, block_round):
        '''
        Run a single round, which lasts for time length round_length.
        All we have to do is update each miner for the round, and if we found
        a block update the top miner appropriately.
        '''

        for miner in self.all_miners:
            miner.update_for_round(round_number)

        if block_round:
            sorted_miners = sorted(self.all_miners,
                                   key=lambda miner: miner.shares,
                                   reverse=True)
            winner = sorted_miners[0]
            second = sorted_miners[1]
            cost = second.shares
            diff = winner.shares - second.shares

            self.cost_history.append(cost)
            self.diff_history.append(diff)

            winner.shares -= cost
            winner.blocks_won += 1

            for miner in self.all_miners:
                if self.num_blocks_found > 1000: 
                    miner.update_block_found(winner, cost, diff, round_number, 
                                             True)
                else:
                    miner.update_block_found(winner, cost, diff, round_number, 
                                             False)

    def debug_print_scoreboard(self):
        all_data = []
        for m in self.honest_miners:
            all_data.append([m.Id, m.shares, m.hash_rate, m.blocks_won, 'Hon'])
        for m in self.malicious_miners:
            all_data.append([m.Id, m.shares, m.hash_rate, m.blocks_won, 'Mal'])
        self.logger.info(tabulate(all_data, headers=["Id", "Shares", "HashRate", "Blocks Won", "Type"]))

    def write_final_stats(self):
        '''
        Write simulation statistics to file
        '''

        if self.file_name is None:
            file_name = 'mining_sim_stats_{0}_{1}_{2}_{3}.txt'.format(self.num_honest_miners,
                                                                  self.num_malicious_miners,
                                                                  self.num_blocks_to_sim,
                                                                  datetime.datetime.now().isoformat())
        else:
            file_name = self.file_name

        with open(file_name, 'w') as ftw:
            sorted_honest_miners = sorted(self.honest_miners, 
                                          key=lambda miner: miner.hash_rate)
            for miner in sorted_honest_miners:
                if len(miner.cost_history) == 0:
                    avg_block_cost = 0
                else:
                    avg_block_cost = sum(miner.cost_history) / float(len(miner.cost_history)) 
                miner_output = "H,{0},{1},{2},{3},{4}\n".format(miner.Id,
                                                            miner.hash_rate,
                                                            miner.shares,
                                                            miner.blocks_won,
                                                            avg_block_cost)
                ftw.write(miner_output)
                miner.print_self()
            sorted_malicious_miners = sorted(self.malicious_miners, 
                                             key=lambda miner: miner.hash_rate)
            for miner in sorted_malicious_miners:
                if len(miner.cost_history) == 0:
                    avg_block_cost = 0
                else:
                    avg_block_cost = sum(miner.cost_history) / float(len(miner.cost_history)) 
                miner_output = "M,{0},{1},{2},{3},{4},".format(miner.Id,
                                                            miner.hash_rate,
                                                            miner.shares,
                                                            miner.blocks_won,
                                                            avg_block_cost)
                if self.attack_type == "SPITE":
                    miner_output += "{0},{1},{2},{3}\n".format(miner.target_miner.Id, 
                                                    float(miner.hash_rate / miner.target_miner.hash_rate),miner.saved_shares,miner.lost_shares)
                else:
                    miner_output += "{0},{1}\n".format(miner.target_miner.Id, 
                                                    float(miner.hash_rate / miner.target_miner.hash_rate))
                ftw.write(miner_output)
                miner.print_self()
