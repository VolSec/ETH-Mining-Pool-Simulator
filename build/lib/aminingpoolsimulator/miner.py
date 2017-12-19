'''
Miner class for simulating the solo-block crypto miner attack
'''

import numpy as np
import math
import sys
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
        self.share_sample_index = 0

    def update_for_round(self, round_num):
        if round_num % SHARE_BATCH_SIZE == 0 and round_num != 0:
            self.shares_per_round = self.pool.get_shares_for_miner(self.hash_rate, SHARE_BATCH_SIZE)
        self.shares += self.shares_per_round[round_num % SHARE_BATCH_SIZE]

    def update_block_found(self, winner, cost, diff, round_num):
        pass

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

    def update_block_found(self, winner, cost, diff, round_num):
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
        self.saved_shares = 0
        self.rounds_since_last_block = 0
        self.avg_rounds_between_blocks = 0
        self.miner_avg_sprs = []
        self.target_costs = []
        self.my_costs = []
        self.good_attacks = []
        self.active_saved_shares = 0
        self.attack_params = attack_params
        self.lost_shares = 0
        self.run_honest = False 
        if attack_params == 'HONEST':
            self.logger.info("RUNNING HONESTLY")
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
        self.target_miner = sorted_miners[1]
        self.other_account = sorted_miners[len(sorted_miners) - 1]
        self.other_account.hash_rate = .01
        
        self.logger.info("My hashrate: {0}, t_hashrate: {1}".format(self.hash_rate,
                                                                    self.target_miner.hash_rate))
        self.logger.info("My Id: {0}, Target Id: {1}".format(self.Id, self.target_miner.Id))

                                
    def determine_avg_block_rate(self):
        p = self.pool.round_length * self.pool.total_hash_rate
        p = float(p / self.pool.network_difficulty)
        samples = np.random.geometric(p, 20000) 
        
        return math.floor(float(sum(samples) / len(samples)))
 
    def update_for_round(self, round_num):
        if round_num % SHARE_BATCH_SIZE == 0 and round_num != 0:
            self.shares_per_round = self.pool.get_shares_for_miner(self.hash_rate, 
                                                                   SHARE_BATCH_SIZE)

        self.rounds_since_last_block += 1
        shares_this_round = self.shares_per_round[round_num % SHARE_BATCH_SIZE]
        
        self.active_saved_shares += shares_this_round
        
        # Nothing to do
        if self.active_saved_shares == 0:
            return

        # If we're running honestly just act like a regular miner
        if self.run_honest:
            self.shares += shares_this_round
            return 
         
        # Guess when the next block will actually come
        next_block_time = self.avg_rounds_between_blocks - self.rounds_since_last_block
        if next_block_time < 1:
            next_block_time = 1
        
        # get current miners sorted by shares
        miners_right_now = sorted(self.pool.all_miners, 
                                  key=lambda miner: miner.shares,
                                  reverse=True)
    
        current_winner = miners_right_now[0]
        current_second = miners_right_now[1]
        current_gap = current_winner.shares - current_second.shares
        target_gap = self.target_miner.shares - self.shares
        my_place = miners_right_now.index(self)
        target_place = miners_right_now.index(self.target_miner)
        
        self.logger.debug("  CP: {0}, CS: {1}, TP: {2}, TS: {3}".format(my_place,
                                                                     self.shares,
                                                                     target_place,
                                                                     self.target_miner.shares))
        
        # Behind and we can catch up (this round)
        if target_gap <= self.active_saved_shares and target_gap > 0:
            self.logger.debug("    I can catch up, gonna sit 1 behind") 
            self.shares += target_gap - 1
            self.active_saved_shares -= target_gap - 1

            potential_shares = self.miner_avg_sprs[self.target_miner.Id] * (next_block_time + 2)

            self.logger.debug("       Predicted target miner shares at next block {0}".format(potential_shares))

            if potential_shares < self.active_saved_shares:
                self.logger.debug("        Offloading excess shares")

                definitely_excess = self.active_saved_shares - potential_shares
                self.other_account.shares += definitely_excess
                self.active_saved_shares -= definitely_excess

        # Behind and can't catch up (this round)
        elif target_gap > self.active_saved_shares:

            mg_tups = []
            for miner in self.pool.all_miners:
                mg_tups.append((miner, miner.shares))
            
            my_victory = self.predict_target_victory(mg_tups, 
                                                     next_block_time,
                                                     self)
            target_victory = self.predict_target_victory(mg_tups,
                                                         next_block_time,
                                                         self.target_miner)
            # I will win first (predicted), try to close the gap
            if my_victory < target_victory:
                self.logger.debug("    Need to close the gap, adding all shares")
                self.shares += self.active_saved_shares
                self.active_saved_shares = 0
            # I won't win first (predicted), cut my losses
            else:
                self.other_account.shares += self.active_saved_shares
                self.active_saved_shares = 0
                self.logger.debug("    Should offload shares wait till he wins")
        # We're tied
        elif target_gap == 0:
            self.logger.debug("    Fuck me we're tied, giving 1 share, saving rest") 
            self.target_miner.shares += 1
            self.active_saved_shares -= 1
        # we're ahead
        elif target_gap < 0:
            # Presumably, the target recently won, so we should now try to follow him to
            # the bottom by winning a block ourselves
            if self.shares + self.active_saved_shares > current_winner.shares:
                self.shares += self.active_saved_shares
                self.active_saved_shares = 0
            # We guessed wrong and are now far away from our target; wait for him to
            # catch up to us
            else:
                self.other_account.shares += self.active_saved_shares
                self.active_saved_shares = 0
        else:
            if current_winner is self.target_miner and current_gap <= self.active_saved_shares:
                if next_block_time <= 3:
                    current_second.shares += current_gap - 1
                    self.active_saved_shares -= current_gap - 1
            #print("what did we miss")
            print("  CP: {1}, CS: {1}, TP: {2}, TS: {3}".format(my_place,
                                                                self.shares,
                                                                target_place,
                                                                self.target_miner.shares))

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
        '''
        Predict the next block ahead by sampling the other miners shares,
        to determine our next move
        '''
        for index, tup in enumerate(ms_tuples):
            xps = self.miner_avg_sprs[tup[0].Id] * next_block_time
            ms_tuples[index] = (tup[0], tup[1] + xps)
         
        sorted_miner_guess_tuples = sorted(ms_tuples, 
                                           key=lambda tup: tup[1],
                                           reverse=True)

        return sorted_miner_guess_tuples 

    def update_block_found(self, winner, cost, diff, round_num):
        '''
        Just reset our timing mechanism for simulating ahead,
        and bask in the glory of winning
        '''
        if winner is self.target_miner: 
            second = sorted(self.pool.all_miners,
                            key = lambda miner: miner.shares,
                            reverse=True)[0]
            if second is self:
                self.spend = True
            self.target_costs.append(cost)
            print("Target miner won at a diff of: {0}".format(diff))
            if diff == 1:
                self.good_attacks.append(cost)

        self.spend = False
        if winner is self:
            self.my_costs.append(cost)
        print("Block found! Winner: {0}".format(winner.Id))
        
        self.rounds_since_last_block = 0
        self.logger.info("lost {0} shares this block".format(self.active_saved_shares))
        self.lost_shares += self.active_saved_shares
        self.active_saved_shares = 0
        self.last_winner = winner 
 
    def end_simulation(self):
        self.pool.debug_print_scoreboard()
        self.logger.info("Saved: {0} shares for other pools".format(self.saved_shares))
        avg_cost = sum(self.pool.cost_history) / float(len(self.pool.cost_history))
        my_avg_cost = sum(self.my_costs) / float(len(self.my_costs))
        target_avg_cost = sum(self.target_costs) / float(len(self.target_costs))
        if not self.run_honest:
            good_costs = sum(self.good_attacks[len(self.good_attacks) // 2:]) / (float(len(self.good_attacks) / 2)) 
            self.logger.info('average block cost for target diff == 1: {0}'.format(good_costs))
        
        self.logger.info('Target was: {0}'.format(self.target_miner.Id))
        self.logger.info('average block cost for pool: {}'.format(avg_cost))
        self.logger.info('average block cost for me: {0}'.format(my_avg_cost))
        self.logger.info('average block cost for target: {0}'.format(target_avg_cost))
        self.logger.info('num blocks saved: {0}'.format(self.saved_shares / avg_cost))

    def sample_shares(self, n=1):
        '''
        Get a sample of shares from the distribution given a miners hashrate
        '''

        if n > (len(self.share_samples) - self.share_sample_index):
            self.share_samples = self.pool.get_shares_for_miner(self.hash_rate, 
                                                                max(SHARE_BATCH_SIZE, 
                                                                    n))
 
        samples = self.share_samples[self.share_sample_index:self.share_sample_index+n]
        self.share_sample_index += n
        
        return samples
    
    def print_self(self):
        to_print = "Malicious Miner: {0}, hash_rate: {1}, ".format(self.Id,
                                                                   self.hash_rate)
        to_print += "shares: {0}, blocks won: {1}".format(self.shares,
                                                          self.blocks_won)
        self.logger.debug(to_print)
