import numpy
import random
from matplotlib import pyplot



class KArmedBandit:
    def __init__(self, K):
        self.K = K
        self._q = numpy.random.normal(0.0, 1.0, K)


    def play(self, k):
        return numpy.random.normal(self._q[k], 1.0)



class SampleAveragePlayer:
    def __init__(self, testbed, explore_pct):
        self.testbed = testbed
        self.K = testbed.K
        self.explore_pct = explore_pct
        self.Q = numpy.zeros(self.K)
        self.N = numpy.zeros(self.K)
        self.score = 0
        self.Q_hist = [[] for i in range(self.K)]
        self.avg_score_hist = []


    def play(self, num):
        for t in range(num):
            k = self.select_action()
            r = self.testbed.play(k)
            self.update(k, r)
            self.score += r
            avg_score = self.score / (t + 1)
            print("{} : r:{:6.3f} => Q[{}]={:6.3f}, avg score:{:6.3f}".format(k, r, k, self.Q[k], avg_score))
            self.avg_score_hist.append(avg_score)


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
        self.N[k] += 1
        self.Q[k] += 1.0/self.N[k] * (r - self.Q[k])
        for i in range(self.K):
            self.Q_hist[i].append(self.Q[i])


    def show(self):
        fig = pyplot.figure()
        for k in range(self.K):
            self.Q_hist[k].append(self.testbed._q[k])
            pyplot.plot(self.Q_hist[k], marker='o', markevery=[len(self.Q_hist[k])-1])
        pyplot.plot(self.avg_score_hist)
        pyplot.show()



def main():
    testbed = KArmedBandit(10)
    player = SampleAveragePlayer(testbed, 0.1)
    player.play(1000)
    player.show()



if __name__ == "__main__":
    main()
