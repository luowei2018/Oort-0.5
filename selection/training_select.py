import logging
import math
from statistics import mean
from collections import OrderedDict
import numpy as np2

def create_training_selector(args):
    return _training_selector(args)

class _training_selector(object):
    def __init__(self, args, seed=518):
        np2.random.seed(seed)

        self.nodes = OrderedDict()
        self.args = args
        self.round = 0

        self.explore_factor = args.exploration_factor
        self.decay_factor = args.exploration_decay
        self.explore_min = args.exploration_min
        self.alpha = args.exploration_alpha
        self.threshold = args.round_threshold
        self.prefer_duration = float('inf')
        self.sample_window = args.sample_window

        self.explore = []
        self.exploit = []
        self.exploitHistory = []
        self.unexplored = set()

        self.successful_clients = set()
        self.blacklist = None

    def update_client(self, id, feedbacks):
        if id not in self.nodes:
            self.nodes[id] = {
                'reward': feedbacks['reward'],
                'duration': feedbacks['duration'],
                'time_stamp': self.round,
                'count': 0,
                'status': True
            }
            self.unexplored.add(id)
        else:
            self.nodes[id]['reward'] = feedbacks['reward']
            self.nodes[id]['duration'] = feedbacks['duration']
            self.nodes[id]['time_stamp'] = feedbacks['time_stamp']
            self.nodes[id]['count'] += 1
            self.nodes[id]['status'] = feedbacks['status']

            self.successful_clients.add(id)
            self.unexplored.discard(id)


    def cal_avg_reward(self, clients):
        count, rewards = 0, 0
        for client in clients:
            if client in self.successful_clients:
                count += 1
                rewards += self.nodes[client]['reward']

        if count == 0:
            return 0
        return rewards/count

    def cal_avg_rewards(self):

        explorationRes = self.cal_avg_reward(self.explore)
        exploitationRes = self.cal_avg_reward(self.exploit)

        return explorationRes, exploitationRes

    def pacer(self):
        step = self.args.pacer_step
        delta = self.args.pacer_delta
        training_round = self.round
        avgExplorationRewards, avgExploitationRewards = self.cal_avg_rewards()

        self.exploitHistory.append(avgExploitationRewards)
        self.successful_clients = set()

        if int(training_round % step) == 0 and training_round >= 2 * step:
            curr_round_reward = sum(self.exploitHistory[-step:])
            last_round_reward = sum(self.exploitHistory[-2 * step: -step])

            reward_change = abs(curr_round_reward - last_round_reward)
            # change too sharp
            if reward_change >= 6 * last_round_reward: # change 5 to 8
                self.threshold = max(delta, self.threshold - delta)
            # change too flat
            elif reward_change <= 0.15 * last_round_reward:
                self.threshold = min(100., self.threshold + delta)

            logging.debug("Training selector (round {}): last_round_reward {}, curr_round_reward {}"
                .format(training_round, last_round_reward, curr_round_reward))

        logging.info("Training selector: avgExploitationRewards {}, avgExplorationRewards {}".
                        format(avgExploitationRewards, avgExplorationRewards))

    def set_blacklist(self):
        roundsNum = self.args.blacklist_rounds
        if roundsNum == -1:
            bl = []

        sorted_ids = sorted(list(self.nodes), reverse=True,
                            key=lambda k:self.nodes[k]['count'])

        for id in sorted_ids:
            if self.nodes[id]['count'] <= roundsNum:
                break
            bl.append(id)
        self.blacklist = set(bl)

        # Exceed blacklist threshold
        max_len = int(self.args.blacklist_max_len)
        bl_threshold = max_len * len(self.nodes)
        if len(bl) > bl_threshold:
            self.blacklist = set(bl[:bl_threshold ])


    def select_client(self, num_clients, avail_clients=None):
        if avail_clients != None:
            clients = avail_clients
        else:
            clients = set([x for x in self.nodes.keys() if self.nodes[x]['status']] == True)

        return self.getTopK(num_clients, self.round+1, clients)

    def update_duration(self, clientId, duration):
        if clientId in self.nodes:
            self.nodes[clientId]['duration'] = duration

    def mean_reward(self):
        totalRewards = [self.nodes[x]['reward'] for x in list(self.nodes.keys()) if int(x) not in self.blacklist]
        if len(totalRewards) > 0:
            return mean(totalRewards)

        return 0

    def get_clients(self):
        return self.nodes

    def normalize(self, data, clip_bound=0.96, threshold=1e-4):
        data.sort()
        _min = float(data[0])
        _max = float(data[-1])
        _avg = mean(data)
        _range = max(_max - _min, threshold)
        _clip_val = float(data[min(len(data)-1, int(len(data)*clip_bound))])

        return _max, _min, _range, _avg, _clip_val

    def getTopK(self, numOfSamples, cur_time, feasible_clients):
        self.round = cur_time
        self.set_blacklist()
        self.pacer()
        orderedKeys = []
        if feasible_clients != None:
            orderedKeys = [x for x in list(feasible_clients) if x not in self.blacklist] # Does feasible_clients stores clientId (int)?

        # normalize the score of all arms: Avg + Confidence
        scores, numExploited, exploreLen = {}, 0, 0
        moving_reward, staleness, allloss = [], [], {}

        for clientId in orderedKeys:
            if self.nodes[clientId]['reward'] <= 0:
                continue
            moving_reward.append(self.nodes[clientId]['reward'])
            staleness.append(cur_time - self.nodes[clientId]['time_stamp'])

        max_reward, min_reward, range_reward, avg_reward, clip_value = self.normalize(moving_reward, self.args.clip_bound)
        max_staleness, min_staleness, range_staleness, avg_staleness, _ = self.normalize(staleness, threshold=1)

        if self.threshold < 100.:
            sortedDuration = sorted([self.nodes[clientId]['duration'] for clientId in list(self.nodes.keys())])
            self.prefer_duration = sortedDuration[min(int(len(sortedDuration) * self.threshold/100.), len(sortedDuration)-1)]
        else:
            self.prefer_duration = float('inf')

        for key in orderedKeys:
            # we have played this arm before
            if self.nodes[key]['count'] > 0:
                clientDuration = self.nodes[key]['duration']
                creward = min(self.nodes[key]['reward'], clip_value)
                numExploited += 1

                sc = (creward - min_reward)/range_reward \
                    + math.sqrt(0.1*math.log(cur_time)/self.nodes[key]['time_stamp']) # temporal uncertainty

                if clientDuration > self.prefer_duration:
                    sc *= ((self.prefer_duration/max(1e-4, clientDuration)) ** self.args.round_penalty)

                if self.nodes[key]['time_stamp']==cur_time:
                    allloss[key] = sc

                scores[key] = abs(sc)

        clientLakes = list(scores.keys())
        self.explore_factor = max(self.explore_factor*self.decay_factor, self.explore_min)
        explorationLen = int(numOfSamples*self.explore_factor)

        # exploitation
        exploitLen = min(numOfSamples-explorationLen, len(clientLakes))

        # take the top-k, and then sample by probability, take 95% of the cut-off loss
        sortedClientUtil = sorted(scores, key=scores.get, reverse=True)

        # take cut-off utility
        cut_off_util = scores[sortedClientUtil[exploitLen]] * self.args.cut_off_util

        tempPickedClients = []
        for clientId in sortedClientUtil:
            # we want at least 10 times of clients for augmentation
            if scores[clientId] < cut_off_util and len(tempPickedClients) > 15.*exploitLen: # change 10 to 15
                break
            tempPickedClients.append(clientId)

        augment_factor = len(tempPickedClients)

        totalSc = max(1e-4, float(sum([scores[key] for key in tempPickedClients])))
        self.exploit = list(np2.random.choice(tempPickedClients, exploitLen, p=[scores[key]/totalSc for key in tempPickedClients], replace=False))

        pickedClients = []

        # exploration
        _unexplored = [x for x in list(self.unexplored) if int(x) in feasible_clients]
        if len(_unexplored) > 0:
            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = self.nodes[cl]['reward']
                clientDuration = self.nodes[cl]['duration']

                if clientDuration > self.prefer_duration:
                    init_reward[cl] *= ((float(self.prefer_duration)/max(1e-4, clientDuration)) ** self.args.round_penalty)

            # prioritize w/ some rewards (i.e., size)
            exploreLen = min(len(_unexplored), numOfSamples-len(self.exploit))
            pickedUnexploredClients = sorted(init_reward, key=init_reward.get, reverse=True)[:min(int(self.sample_window*exploreLen), len(init_reward))]

            unexploredSc = float(sum([init_reward[key] for key in pickedUnexploredClients]))

            pickedUnexplored = list(np2.random.choice(pickedUnexploredClients, exploreLen,
                            p=[init_reward[key]/max(1e-4, unexploredSc) for key in pickedUnexploredClients], replace=False))

            self.explore = pickedUnexplored


        pickedClients = self.explore + self.exploit
        top_k_score = []
        for i in range(min(3, len(pickedClients))):
            clientId = pickedClients[i]
            _score = (self.nodes[clientId]['reward'] - min_reward)/range_reward
            _staleness = self.alpha*((cur_time-self.nodes[clientId]['time_stamp']) - min_staleness)/float(range_staleness) #math.sqrt(0.1*math.log(cur_time)/max(1e-4, self.nodes[clientId]['time_stamp']))
            top_k_score.append((self.nodes[clientId], [_score, _staleness]))

        logging.info("At round {}, UCB exploited {}, augment_factor {}, exploreLen {}, un-explored {}, exploration {}, round_threshold {}, sampled score is {}"
            .format(cur_time, numExploited, augment_factor/max(1e-4, exploitLen), exploreLen, len(self.unexplored), self.explore_factor, self.threshold, top_k_score))
        logging.info("At time {}, all rewards are {}".format(cur_time, allloss))

        return pickedClients

    def getTopK_PF(self, numOfSamples, cur_time, feasible_clients):
        self.round = cur_time
        self.set_blacklist()
        self.pacer()
        orderedKeys = []
        if feasible_clients != None:
            orderedKeys = [x for x in list(feasible_clients) if x not in self.blacklist] # Does feasible_clients stores clientId (int)?

        # normalize the score of all arms: Avg + Confidence
        scores, numExploited, exploreLen = {}, 0, 0
        moving_reward, staleness, allloss = [], [], {}

        for clientId in orderedKeys:
            if self.nodes[clientId]['reward'] <= 0:
                continue
            moving_reward.append(self.nodes[clientId]['reward'])
            staleness.append(cur_time - self.nodes[clientId]['time_stamp'])

        max_reward, min_reward, range_reward, avg_reward, clip_value = self.normalize(moving_reward, self.args.clip_bound)
        max_staleness, min_staleness, range_staleness, avg_staleness, _ = self.normalize(staleness, threshold=1)

        if self.threshold < 100.:
            sortedDuration = sorted([self.nodes[clientId]['duration'] for clientId in list(self.nodes.keys())])
            self.prefer_duration = sortedDuration[min(int(len(sortedDuration) * self.threshold/100.), len(sortedDuration)-1)]
        else:
            self.prefer_duration = float('inf')

        for key in orderedKeys:
            # we have played this arm before
            if self.nodes[key]['count'] > 0:
                clientDuration = self.nodes[key]['duration']
                creward = min(self.nodes[key]['reward'], clip_value)
                numExploited += 1

                sc = (creward - min_reward)/range_reward \
                    + math.sqrt(0.1*math.log(cur_time)/self.nodes[key]['time_stamp']) # temporal uncertainty

                if clientDuration > self.prefer_duration:
                    sc *= ((self.prefer_duration/max(1e-4, clientDuration)) ** self.args.round_penalty)

                if self.nodes[key]['time_stamp']==cur_time:
                    allloss[key] = sc

                scores[key] = abs(sc)

        clientLakes = list(scores.keys())
        self.explore_factor = max(self.explore_factor*self.decay_factor, self.explore_min)
        explorationLen = int(numOfSamples*self.explore_factor)

        # exploitation
        exploitLen = min(numOfSamples-explorationLen, len(clientLakes))

        # take the top-k, and then sample by probability, take 95% of the cut-off loss
        sortedClientUtil = sorted(scores, key=scores.get, reverse=True)

        # take cut-off utility
        cut_off_util = scores[sortedClientUtil[exploitLen]] * self.args.cut_off_util

        tempPickedClients = []
        for clientId in sortedClientUtil:
            # we want at least 10 times of clients for augmentation
            if scores[clientId] < cut_off_util and len(tempPickedClients) > 15.*exploitLen: # change 10 to 15
                break
            tempPickedClients.append(clientId)

        augment_factor = len(tempPickedClients)

        totalSc = max(1e-4, float(sum([scores[key] for key in tempPickedClients])))
        self.exploit = list(np2.random.choice(tempPickedClients, exploitLen, p=[scores[key]/totalSc for key in tempPickedClients], replace=False))

        pickedClients = []

        # exploration
        _unexplored = [x for x in list(self.unexplored) if int(x) in feasible_clients]
        if len(_unexplored) > 0:
            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = self.nodes[cl]['reward']
                clientDuration = self.nodes[cl]['duration']

                if clientDuration > self.prefer_duration:
                    init_reward[cl] *= ((float(self.prefer_duration)/max(1e-4, clientDuration)) ** self.args.round_penalty)

            # prioritize w/ some rewards (i.e., size)
            exploreLen = min(len(_unexplored), numOfSamples-len(self.exploit))
            pickedUnexploredClients = sorted(init_reward, key=init_reward.get, reverse=True)[:min(int(self.sample_window*exploreLen), len(init_reward))]

            unexploredSc = float(sum([init_reward[key] for key in pickedUnexploredClients]))

            pickedUnexplored = list(np2.random.choice(pickedUnexploredClients, exploreLen,
                            p=[init_reward[key]/max(1e-4, unexploredSc) for key in pickedUnexploredClients], replace=False))

            self.explore = pickedUnexplored

        # pareto front
        tmp_list = []
        for client in orderedKeys:
            client_dict = dict.fromkeys(scores[client])
            client_dict[scores[client]] = client
            pareto_tuple = (scores[client], self.nodes[client]['reward'])
            logging.info("pareto_tuple: {}", pareto_tuple)
            tmp_list.append(pareto_tuple)
            logging.info("tmp_list: {}", tmp_list)

        pareto_list = []
        while len(pareto_list) < numOfSamples:
            for score, reward in enumerate(tmp_list):
                is_pareto[score] = np.all(np.any(tmp_list[:score] < reward, axis=1)) and np.all(np.any(tmp_list[score+1:] < reward, axis=1))
                if is_pareto[score]:
                    iter_list.append(client_dict[score])
                    tmp_list.remove((score, reward))
            if (len(pareto_list) + len(iter_list)) <= numOfSamples:
                pareto_list.append(iter_list)
            else:
                break

        remain_picked = getTopK(numOfSamples - len(pareto_list), self.training_round, _unexplored)

        #pareto: 
        #ickedClients = pareto_list + remain_picked
        #original:
        pickedClients = self.explore + self.exploit
        top_k_score = []
        for i in range(min(3, len(pickedClients))):
            clientId = pickedClients[i]
            _score = (self.nodes[clientId]['reward'] - min_reward)/range_reward
            _staleness = self.alpha*((cur_time-self.nodes[clientId]['time_stamp']) - min_staleness)/float(range_staleness) #math.sqrt(0.1*math.log(cur_time)/max(1e-4, self.nodes[clientId]['time_stamp']))
            top_k_score.append((self.nodes[clientId], [_score, _staleness]))

        logging.info("At round {}, UCB exploited {}, augment_factor {}, exploreLen {}, un-explored {}, exploration {}, round_threshold {}, sampled score is {}"
            .format(cur_time, numExploited, augment_factor/max(1e-4, exploitLen), exploreLen, len(self.unexplored), self.explore_factor, self.threshold, top_k_score))
        logging.info("At time {}, all rewards are {}".format(cur_time, allloss))

        return pickedClients