#!/usr/bin/env python3

import numpy
import random
from matplotlib import pyplot
import matplotlib.ticker as mtick



class KArmedBandit:
    def __init__(self, K, initial_sigma = 1.0, random_walk_sigma = 0.0):
        self.K = K
        self.random_walk_sigma = random_walk_sigma
        self._q = numpy.random.normal(0.0, initial_sigma, K)
        self._optimal_k = self._q.argmax()
        self._q_hist = [[] for i in range(self.K)]


    def play(self, players, num_steps):
        for t in range(num_steps):
            q_adjust = numpy.random.normal(0.0, self.random_walk_sigma, self.K)
            self._q += q_adjust
            for i in range(self.K):
                self._q_hist[i].append(self._q[i])
            self._optimal_k = self._q.argmax()
            for player in players:
                player.play_one(t)


    def do_action(self, k):
        return numpy.random.normal(self._q[k], 1.0)


    def show(self):
        fig = pyplot.figure()
        pyplot.title("evolution of real q values per arm")
        for k in range(self.K):
            pyplot.plot(self._q_hist[k])



class Player:
    def __init__(self, testbed, explore_pct):
        self.testbed = testbed
        self.K = testbed.K
        self.explore_pct = explore_pct
        self.Q = numpy.zeros(self.K)
        self.Q_hist = [[] for i in range(self.K)]
        self.r_hist = []
        self.optimal_choice = []
        self.debug = False


    def play_one(self, t):
        k = self.select_action()
        r = self.testbed.do_action(k)
        self.update(k, r)
        if self.debug:
            print("{} : r:{:6.3f} => Q[{}]={:6.3f}".format(k, r, k, self.Q[k]))
        self.r_hist.append(r)
        self.optimal_choice.append((k == self.testbed._optimal_k) * 1.0)


    def select_action(self):
        possible_actions = list(range(self.K))
        if self.should_move_greedy():
            possible_actions = self.get_best_actions()
        return self.select_random_action(possible_actions)


    def should_move_greedy(self):
        randval = random.random()
        return (randval > self.explore_pct)


    def get_best_actions(self):
        maxval = numpy.max(self.Q)
        return list(numpy.argwhere(self.Q == maxval).flatten())


    def select_random_action(self, possible_actions):
        i = random.randint(0, len(possible_actions) - 1)
        return possible_actions[i]


    def update(self, k, r):
        self.update_action_value(k, r)
        for i in range(self.K):
            self.Q_hist[i].append(self.Q[i])


    def update_action_value(self, k, r):
        raise Exception("update_action_value() not implemented in base player")


    def show(self):
        fig = pyplot.figure()
        pyplot.title("evolution of Q estimates per arm")
        for k in range(self.K):
            self.Q_hist[k].append(self.testbed._q[k])
            pyplot.plot(self.Q_hist[k], marker='o', markevery=[len(self.Q_hist[k])-1])



class SampleAveragePlayer(Player):
    def __init__(self, testbed, explore_pct):
        super().__init__(testbed, explore_pct)
        self.N = numpy.zeros(self.K)


    def update_action_value(self, k, r):
        self.N[k] += 1
        self.Q[k] += 1.0/self.N[k] * (r - self.Q[k])



class ConstantStepSizePlayer(Player):
    def __init__(self, testbed, explore_pct, alpha):
        super().__init__(testbed, explore_pct)
        self.alpha = alpha


    def update_action_value(self, k, r):
        self.Q[k] += self.alpha * (r - self.Q[k])



class UnbiasedConstantStepSizePlayer(Player):
    def __init__(self, testbed, explore_pct, alpha):
        super().__init__(testbed, explore_pct)
        self.alpha = alpha
        self.trace_of_one = numpy.zeros(self.K)


    def update_action_value(self, k, r):
        self.trace_of_one[k] += self.alpha * (1 - self.trace_of_one[k])
        beta = self.alpha / self.trace_of_one[k]
        self.Q[k] += beta * (r - self.Q[k])



def main():
    num_arms = 10
    num_runs = 2000
    num_steps = 10000

    sigma_initial = 0.0
    sigma_random_walk = 0.01

    explore_pct = 0.1
    fixed_alpha = 0.1

    player_labels = []
    player_labels.append("sample average")
    player_labels.append("fixed alpha = {}".format(fixed_alpha))
    player_labels.append("unbiased fixed alpha = {}".format(fixed_alpha))
    num_players = len(player_labels)

    avg_rewards = numpy.zeros((num_players, num_steps))
    avg_optimal = numpy.zeros((num_players, num_steps))
    for b in range(num_runs):
        testbed = KArmedBandit(num_arms, sigma_initial, sigma_random_walk)
        player1 = SampleAveragePlayer(testbed, explore_pct)
        player2 = ConstantStepSizePlayer(testbed, explore_pct, fixed_alpha)
        player3 = UnbiasedConstantStepSizePlayer(testbed, explore_pct, fixed_alpha)
        players = [player1, player2, player3]
        testbed.play(players, num_steps)
        for p, player in enumerate(players):
            avg_rewards[p] += numpy.asarray(player.r_hist)
            avg_optimal[p] += numpy.asarray(player.optimal_choice)
        print("Run {:5d}".format(b), end = '\r')
    print()

    # show details of last run
    testbed.show()
    for player in players:
        player.show()
    pyplot.show()

    # show averages over all runs
    pyplot.figure()
    avg_rewards /= num_runs
    avg_optimal /= num_runs
    ax1 = pyplot.subplot(2, 1, 1)
    pyplot.title("average reward")
    pyplot.plot(avg_rewards.T)
    pyplot.legend(player_labels)
    ax2 = pyplot.subplot(2, 1, 2)
    pyplot.title("% optimal action")
    pyplot.plot(avg_optimal.T)
    pyplot.legend(player_labels)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    pyplot.show()



if __name__ == "__main__":
    main()
