'''
Miner class for simulating the solo-block crypto miner attack
'''

import numpy as np
import math
import sys
import random
import logging

SHARE_BATCH_SIZE = 10000

class HonestMiner:
    def __init__(self, Id, hash_rate, round_length, pool):
        self.logger = logging.getLogger(__name__)
        self.Id = Id
        self.hash_rate = hash_rate
        self.round_length = round_length
        self.pool = pool
        self.shares = 0
        self.blocks_won = 0
        self.shares_per_round = None
        self.share_samples = []
        self.cost_history = []
        self.share_sample_index = 0

    def update_for_round(self, round_num):
        if round_num % SHARE_BATCH_SIZE == 0 and round_num != 0:
            self.shares_per_round = self.pool.get_shares_for_miner(self.hash_rate, SHARE_BATCH_SIZE)
        self.shares += self.shares_per_round[round_num % SHARE_BATCH_SIZE]

    def update_block_found(self, winner, cost, diff, round_num, record_result):
        if winner is self and record_result is True:
            self.cost_history.append(cost)

    def end_simulation(self):
        pass

    def pre_simulation(self):
        self.shares_per_round = self.pool.get_shares_for_miner(self.hash_rate, SHARE_BATCH_SIZE)

    def sample_shares(self, n=1):
        if n > (len(self.share_samples) - self.share_sample_index):
            self.share_samples = self.pool.get_shares_for_miner(self.hash_rate, max(SHARE_BATCH_SIZE, n))
            self.share_sample_index = 0
        
        samples = self.share_samples[self.share_sample_index:self.share_sample_index+n]
        self.share_sample_index += n
        return samples
            
    def print_self(self):
        to_print = "Honest Miner: {0}, hash_rate: {1}, ".format(self.Id,
                                                                self.hash_rate)
        to_print += "shares: {0}, blocks won: {1}".format(self.shares,
                                                          self.blocks_won)
        self.logger.debug(to_print)


class CostMaliciousMiner:
    def __init__(self, Id, hash_rate, round_length, attack_params, pool):
        self.logger = logging.getLogger(__name__)
        self.Id = Id
        self.pool = pool
        # override hash rate with median hash rate
        median_hash = sorted(self.pool.hash_rates)[len(self.pool.hash_rates)//2]
        self.hash_rate = median_hash

        self.round_length = round_length
        self.other_miners = []
        self.shares = 0
        self.blocks_won = 0
        self.time_since_last_block = 0
        self.shares_per_round = None
        self.predicted_shares_per_round = None
        self.trying = False
        self.pool = pool
        self.cost_history = []
        self.win_history = []
        self.wasted_shares = 0
        self.total_shares = 0
        self.target_proportion = 0.9
        if attack_params is not None and attack_params != '':
            self.target_proportion = float(attack_params)

    def pre_simulation(self):
        pac = self.pool.network_difficulty
        self.target_shares = self.target_proportion * pac
        self.logger.debug('targeting shares: {}'.format(self.target_shares))

    def get_expected_shares(self, hash_rate, t):
        return hash_rate * t

    def update_block_found(self, winner, cost, diff, round_num, record_result):
        if winner is not self:
            self.cost_history.append(cost)
        else:
            self.win_history.append(cost)

    def end_simulation(self):
        avg_cost = sum(self.cost_history) / float(len(self.cost_history))

        self.logger.info('average block cost for others: {}'.format(avg_cost))

        if len(self.win_history) > 0:
            avg_my_cost = sum(self.win_history) / float(len(self.win_history))
            self.logger.info('average block cost for me: {}'.format(avg_my_cost))
        else:
            self.logger.info('I won 0 blocks :(')


        percent_wasted = self.wasted_shares / self.total_shares * 100
        self.logger.info('shares wasted: {} ({}% of shares mined)'.format(self.wasted_shares, percent_wasted))


    def print_self(self):
        to_print = "Malicious Miner: {0}, hash_rate: {1}, ".format(self.Id,
                                                                   self.hash_rate)
        to_print += "shares: {0}, blocks won: {1}".format(self.shares,
                                                          self.blocks_won)
        self.logger.debug(to_print)

    def update_for_round(self, round_num):
        if round_num % SHARE_BATCH_SIZE == 0:
            self.shares_per_round = self.pool.get_shares_for_miner(self.hash_rate, SHARE_BATCH_SIZE)

        new_shares = self.shares_per_round[round_num % SHARE_BATCH_SIZE]
        self.total_shares += new_shares

        if self.shares < self.target_shares:
            self.shares += new_shares
        else:
            self.wasted_shares += new_shares

class SpiteMaliciousMiner:
    def __init__(self, Id, hash_rate, round_length, attack_params, pool):
        self.logger = logging.getLogger(__name__)
        self.Id = Id
        self.pool = pool
        self.hash_rate = hash_rate
        self.round_length = round_length
        self.shares = 0
        self.blocks_won = 0
        self.rounds_since_last_block = 0
        self.avg_rounds_between_blocks = 0
        self.miner_avg_sprs = []
        self.target_costs = []
        self.cost_history = []
        self.good_attacks = []
        self.active_saved_shares = 0
        self.attack_params = attack_params
        self.lost_shares = 0
        self.last_winner = None
        self.attack_account = self
        self.offload_account = None
        self.saved_shares = 0
        self.run_honest = False
        self.push_attack_account = False

        run_type, dar, dor, dampening = self.attack_params.split(',')
        self.desired_attack_rank = int(dar)
        self.desired_offload_rank = int(dor)
        self.dampening = float(dampening)
        
        if run_type == 'HONEST': 
            self.run_honest = True

    def pre_simulation(self):
        if self.attack_params == 'TOP' or 'HONEST':
            self.set_highest_hash_rate() 
        self.avg_rounds_between_blocks = self.determine_avg_block_rate()
        self.shares_per_round = self.pool.get_shares_for_miner(self.hash_rate, 
                                                               SHARE_BATCH_SIZE)
        for miner in self.pool.all_miners:
            self.miner_avg_sprs.append((sum(miner.shares_per_round) / float(len(miner.shares_per_round))))

    def set_highest_hash_rate(self):
        hh = self.hash_rate
        nh = max(miner.hash_rate for miner in self.pool.all_miners)
        for miner in self.pool.all_miners:
            if miner.hash_rate ==  nh:
                    miner.hash_rate = hh
        self.hash_rate = nh
         
        sorted_miners = sorted(self.pool.all_miners,
                               key = lambda miner: miner.hash_rate,
                               reverse=True)
        self.target_miner = sorted_miners[3]
        self.target_difficulty = math.floor(math.log(self.target_miner.hash_rate / self.pool.shares_per_second) / math.log(2))
        self.target_share_value = math.pow(2,self.target_difficulty)
        
        # Take out the smallest guy in the pool for our second account
        self.offload_account = sorted_miners[len(sorted_miners) - 1]
        self.other_account = sorted_miners[len(sorted_miners) - 1]
        self.other_account.hash_rate = .01
        
        self.logger.info("My hashrate: {0}, t_hashrate: {1}".format(self.hash_rate,
                                                                    self.target_miner.hash_rate))
        self.logger.info("My Id: {0}, Target Id: {1}".format(self.Id, self.target_miner.Id))
        self.logger.info("My target's single share value is: {0}".format(self.target_share_value))
                                
    def determine_avg_block_rate(self):
        p = self.pool.round_length * self.pool.total_hash_rate
        p = float(p / self.pool.network_difficulty)
        samples = np.random.geometric(p, 20000) 
        
        return math.floor(float(sum(samples) / len(samples)))

    def update_for_round(self, round_num):
        if round_num % SHARE_BATCH_SIZE == 0 and round_num != 0:
            self.shares_per_round = self.pool.get_shares_for_miner(self.hash_rate, 
                                                                   SHARE_BATCH_SIZE)
        
        # Keep track of block times
        self.rounds_since_last_block += 1
        shares_this_round = self.shares_per_round[round_num % SHARE_BATCH_SIZE]
        rounds_till_next_block = self.avg_rounds_between_blocks - self.rounds_since_last_block
        if rounds_till_next_block < 1:
            rounds_till_next_block = 1
        
        # If we're running honestly just act like a regular miner
        if self.run_honest:
            self.shares += shares_this_round
            return
 
        self.active_saved_shares += shares_this_round 
        # Nothing to do
        if self.active_saved_shares == 0:
            return
    
        # Determine state of leaderboards
        scoreboard = sorted(self.pool.all_miners,
                            key = lambda miner : miner.shares,
                            reverse = True)
        attack_account = self.attack_account
        offload_account = self.offload_account 
        attack_rank = scoreboard.index(attack_account)
        target_rank = scoreboard.index(self.target_miner)

        #self.print_attack_state(scoreboard)
    
        # Push my attack account over
        if self.push_attack_account:
            attack_account.shares += self.active_saved_shares
            self.active_saved_shares = 0
            return

        # If I'm not in the top n, try to get there
        if attack_rank > self.desired_attack_rank: 
            self.drive_up_leaderboard(scoreboard) 
        else:
            # If I am in the top n but my target is not, wait for him
            max_target_shares = self.target_miner.shares+(self.target_share_value * rounds_till_next_block)
            if max_target_shares*self.dampening < attack_account.shares:
                self.churn_for_target(scoreboard)
            # If we are both in the top n, and the target is ahead, try to jump gap
            elif (attack_rank - target_rank) > 0: 
                self.attack_target(scoreboard)
                # Try to offload shares
                self.offload_shares(scoreboard, rounds_till_next_block) 
            # If we are both in the top 10, and I am ahead, save shares
            elif (attack_rank - target_rank) < 0:        
                # Try to offload shares
                self.offload_shares(scoreboard, rounds_till_next_block) 
            else:
                print("shit")

    def drive_up_leaderboard(self, scoreboard):
        '''
        We're outside the spot we want to sit on the leaderboard, so we need
        to drive our attack account up
        '''

        diff = scoreboard[self.desired_attack_rank].shares - self.attack_account.shares - 1 
        if diff < self.active_saved_shares:
            self.attack_account.shares += diff
            self.active_saved_shares -= diff
        else:
            self.attack_account.shares += self.active_saved_shares
            self.active_saved_shares = 0

    def churn_for_target(self, scoreboard):
        '''
        We're inside our range we want to sit on the leaderboard, but 
        our target is lower, so we need to churn near the top

        I THINK WE SHOULDNT DO THIS BASED ON RANK BUt BASED ON IF THEY CAN CATCH
        UP TO US, TO KEEP MOMENTUM 
        '''

        offload_rank = scoreboard.index(self.offload_account)
        if offload_rank > self.desired_offload_rank:
            offload_diff = scoreboard[self.desired_offload_rank].shares - self.offload_account.shares - 1
            if offload_diff < self.active_saved_shares:
                self.offload_account.shares += offload_diff
                self.active_saved_shares -= offload_diff
                if self.pool.num_blocks_found > 1000:
                    self.saved_shares += self.active_saved_shares
            else:
                self.offload_account.shares += self.active_saved_shares
                self.active_saved_shares += 0
        else:
            if self.pool.num_blocks_found > 1000:
                self.saved_shares += self.active_saved_shares
            self.active_saved_shares = 0

    def attack_target(self, scoreboard):
        '''
        Target jumped ahead of us, and we can catch up.. time to jump 
        1 below him
        '''

        gap = self.target_miner.shares - self.attack_account.shares - 1
        used_shares = min(gap, self.active_saved_shares)
        self.attack_account.shares += used_shares
        self.active_saved_shares -= used_shares

    def offload_shares(self, scoreboard, rounds_till_next_block):
        '''
        Determines if, and how many, shares to offload to other accounts
        '''
        
        max_target_shares = (self.target_share_value * rounds_till_next_block)
        sto = self.active_saved_shares - max_target_shares
        # We think can safely offload shares
        if sto > 0:
            # Do we offload to other account, or save them for another pool?
            offload_rank = scoreboard.index(self.offload_account)
            if offload_rank > self.desired_offload_rank:
                off_diff = scoreboard[self.desired_offload_rank].shares - self.offload_account.shares - 1
                # Can position offload account with leftover shares
                if off_diff < sto:
                    self.offload_account.shares += sto - off_diff
                    sto -= off_diff
                    if self.pool.num_blocks_found > 1000:
                        self.saved_shares += sto
                # Drive up offload account
                else:
                    self.offload_account.shares += sto
            # In a good spot with offload account, keep it there
            else:
                if self.pool.num_blocks_found > 1000:
                    self.saved_shares += sto
                self.active_saved_shares -= sto

    def update_block_found(self, winner, cost, diff, round_num, record_result):
        '''
        Just reset our active shares,
        and bask in the glory of winning
        '''    
        scoreboard = sorted(self.pool.all_miners,
                            key = lambda miner : miner.shares,
                            reverse = True)
        
        self.logger.info("Block found! Winner: {0}".format(winner.Id))
        
        if winner is self.target_miner:
            self.logger.info("  Target miner won at a diff of: {0}".format(diff))
            if record_result:
                self.target_costs.append(cost)
            attack_rank = scoreboard.index(self.attack_account)
            if attack_rank < 3:
                self.push_attack_account = True
                print("    Pushing attack account")
        
        if winner is self.attack_account:
            print("    Attack account won a block")
            self.push_attack_account = False
            if record_result:
                self.cost_history.append(cost)

            attack_rank = scoreboard.index(self.attack_account)
            off_rank = scoreboard.index(self.offload_account) 
            if off_rank < attack_rank: 
                self.attack_account, self.offload_account = self.offload_account, self.attack_account 
        
        self.print_attack_state(scoreboard)
        if record_result:
            self.lost_shares += self.active_saved_shares
            self.active_saved_shares = 0
            self.rounds_since_last_block = 0
    
    def print_attack_state(self, scoreboard):
        target_rank = scoreboard.index(self.target_miner)
        attack_rank = scoreboard.index(self.attack_account)
        off_rank = scoreboard.index(self.offload_account) 
        print("  Current State: ")
        print("  Attack Acc: {0}, offload acc: {1}, target acc: {2}".format(attack_rank,
                                                                            off_rank,
                                                                            target_rank)) 
    def end_simulation(self):
        self.pool.debug_print_scoreboard()
        pass
    
    def print_self(self):
        to_print = "Malicious Miner: {0}, hash_rate: {1}, ".format(self.Id,
                                                                   self.hash_rate)
        to_print += "shares: {0}, blocks won: {1}".format(self.shares,
                                                          self.blocks_won)
        self.logger.debug(to_print)

class ScorchedEarthMiner:
    def __init__(self, Id, hash_rate, round_length, attack_params, pool):
        self.logger = logging.getLogger(__name__)
        self.Id = Id
        self.hash_rate = hash_rate
        self.round_length = round_length
        self.pool = pool
        self.shares = 0
        self.blocks_won = 0
        self.shares_per_round = None
        self.share_samples = []
        self.cost_history = []
        self.target_costs = []
        self.share_sample_index = 0
        self.active_saved_shares = 0
        self.target_miner = None
        self.my_hash_rate_index = int(attack_params.split(',')[0])
        self.target_hash_rate_index = int(attack_params.split(',')[1])

    def pre_simulation(self):
        self.shares_per_round = self.pool.get_shares_for_miner(self.hash_rate, SHARE_BATCH_SIZE)
        self.target_miner = random.choice(self.pool.honest_miners)
        miners_by_hash_rate = sorted(self.pool.all_miners,
                                    key = lambda miner : miner.hash_rate,
                                    reverse = True)
        switch_miner = miners_by_hash_rate[self.my_hash_rate_index]
        self.hash_rate, switch_miner.hash_rate = switch_miner.hash_rate, self.hash_rate
        self.target_miner = miners_by_hash_rate[self.target_hash_rate_index]
        miners_by_hash_rate = sorted(self.pool.all_miners,
                                    key = lambda miner : miner.hash_rate,
                                    reverse = True)
        print("My Id: {0}, I have the {1}th highest hashrate in the pool".format(self.Id, 
                                                                                 miners_by_hash_rate.index(self) + 1))
        print("Target Id: {0}, Has the {1}th highest hashrate in the pool".format(self.target_miner.Id,
                                                                                  miners_by_hash_rate.index(self.target_miner) + 1))
        
    def update_for_round(self, round_num):
        if round_num % SHARE_BATCH_SIZE == 0 and round_num != 0:
            self.shares_per_round = self.pool.get_shares_for_miner(self.hash_rate, SHARE_BATCH_SIZE)
        self.active_saved_shares += self.shares_per_round[round_num % SHARE_BATCH_SIZE]
        
        scoreboard = sorted(self.pool.all_miners,
                            key = lambda miner : miner.shares,
                            reverse = True)
        
        if scoreboard.index(self.target_miner) == 0:
            diff = scoreboard[0].shares - scoreboard[1].shares - 1
            if diff <= self.active_saved_shares:
                scoreboard[1].shares += diff
                self.active_saved_shares -= diff

    def update_block_found(self, winner, cost, diff, round_num, record_result):
        print("Block found! Winner: {0}".format(winner.Id))
        self.active_saved_shares = 0 
        if winner == self.target_miner and record_result is True:
            self.target_costs.append(cost)
            print("Target miner won with diff of: {0}".format(diff))

        if winner is self and record_result is True:
            self.cost_history.append(cost)

    def end_simulation(self):
        pass
 
    def print_self(self):
        to_print = "Honest Miner: {0}, hash_rate: {1}, ".format(self.Id,
                                                                self.hash_rate)
        to_print += "shares: {0}, blocks won: {1}".format(self.shares,
                                                          self.blocks_won)
        self.logger.debug(to_print)



# Utility functions that may be useful in the future

def predict_target_victory(self, ms_tups, next_block_time, target): 
    num_blocks_till_victory = 0
    ms_tups = self.predict_future_block(ms_tups, next_block_time)
    while True:
        if ms_tups[0][0] is target:
            break
        ms_tups[0] = (ms_tups[0][0], ms_tups[0][1] - ms_tups[1][1])
        ms_tups = self.predict_future_block(ms_tups, 
                                            self.avg_rounds_between_blocks)
        num_blocks_till_victory += 1
    
    return num_blocks_till_victory

def predict_future_block(self, ms_tuples, next_block_time):
    for index, tup in enumerate(ms_tuples):
        xps = self.miner_avg_sprs[tup[0].Id] * next_block_time
        if tup[0] is self:
            xps += tup[0].active_saved_shares
        ms_tuples[index] = (tup[0], tup[1] + xps)
     
    sorted_miner_guess_tuples = sorted(ms_tuples, 
                                       key=lambda tup: tup[1],
                                       reverse=True)

    return sorted_miner_guess_tuples 
