#!/usr/bin/env python3

import numpy
import random
from matplotlib import pyplot
import matplotlib.ticker as mtick



class KArmedBandit:
    def __init__(self, K, random_walk_sigma = 0.0):
        self.K = K
        self.random_walk_sigma = random_walk_sigma
        self._q = numpy.random.normal(0.0, 1.0, K)
        self._optimal_k = self._q.argmax()
        self._q_hist = [[] for i in range(self.K)]


    def play(self, k):
        q_adjust = numpy.random.normal(0.0, self.random_walk_sigma, self.K)
        self._q += q_adjust
        for i in range(self.K):
            self._q_hist[i].append(self._q[i])
        self._optimal_k = self._q.argmax()
        return numpy.random.normal(self._q[k], 1.0)


    def show(self):
        fig = pyplot.figure()
        for k in range(self.K):
            pyplot.plot(self._q_hist[k])
        pyplot.show()



class Player:
    def __init__(self, testbed, explore_pct):
        self.testbed = testbed
        self.K = testbed.K
        self.explore_pct = explore_pct
        self.Q = numpy.zeros(self.K)
        self.score = 0
        self.Q_hist = [[] for i in range(self.K)]
        self.avg_score_hist = []
        self.optimal_choice = []
        self.debug = False


    def play(self, num):
        for t in range(num):
            k = self.select_action()
            r = self.testbed.play(k)
            self.update(k, r)
            self.score += r
            avg_score = self.score / (t + 1)
            if self.debug:
                print("{} : r:{:6.3f} => Q[{}]={:6.3f}, avg score:{:6.3f}".format(k, r, k, self.Q[k], avg_score))
            self.avg_score_hist.append(avg_score)
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
        for k in range(self.K):
            self.Q_hist[k].append(self.testbed._q[k])
            pyplot.plot(self.Q_hist[k], marker='o', markevery=[len(self.Q_hist[k])-1])
        pyplot.plot(self.avg_score_hist)
        pyplot.show()



class SampleAveragePlayer(Player):
    def __init__(self, testbed, explore_pct):
        super().__init__(testbed, explore_pct)
        self.N = numpy.zeros(self.K)


    def update_action_value(self, k, r):
        self.N[k] += 1
        self.Q[k] += 1.0/self.N[k] * (r - self.Q[k])



def main():
    num_arms = 10
    num_runs = 200
    num_steps = 10000

    avg_scores = numpy.zeros(num_steps)
    avg_optimal = numpy.zeros(num_steps)
    for b in range(num_runs):
        testbed = KArmedBandit(num_arms, 0.01)
        player = SampleAveragePlayer(testbed, 0.1)
        player.play(num_steps)
        avg_scores += numpy.asarray(player.avg_score_hist)
        avg_optimal += numpy.asarray(player.optimal_choice)
        print("Run {:5d}".format(b), end = '\r')
    print()
    player.show()
    testbed.show()

    avg_scores /= num_runs
    avg_optimal /= num_runs
    ax1 = pyplot.subplot(2, 1, 1)
    pyplot.title("average reward")
    pyplot.plot(avg_scores)
    ax2 = pyplot.subplot(2, 1, 2)
    pyplot.title("% optimal action")
    pyplot.plot(avg_optimal)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    pyplot.show()



if __name__ == "__main__":
    main()
