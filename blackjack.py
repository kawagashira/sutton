#!/usr/bin/env python
#
#                                   blackjack.py
#
#   Exercise 5.1: Blackjack

import numpy as np
import random


class Env:

    def __init__(self):

        self.deck = [1,2,3,4,5,6,7,8,9,10,10,10,10]

    def first_two_cards(self, player, dealer):
        """
Returns:    state = (player's sum, dealer's showing card, player's usable ace)
        """

        player.draw_card(self.hit())
        player.draw_card(self.hit())
        dealer.draw_card(self.hit())
        dealer.draw_card(self.hit())
        self.showing_card = dealer.cards[0]
        self.state = (player.sum, self.showing_card, player.usable_ace)
        return self.state

    def generate_experience(self, player, dealer):

        episode = []
        terminated = 0
        self.first_two_cards(player, dealer)
        reward, terminated = self.update(player, dealer)
        """
        print ('player %8s %2d' % (player.cards, player.sum),  \
            'dealer %8s %2d' % (dealer.cards, dealer.sum),          \
            self.state)
        """
        i = 4
        ### PLAYER"S TURN ###
        while not terminated:
            i += 1
            old_state = self.state
            #print ('old_state', old_state, 'sum', player.sum)
            action = player.player_action()
            card = 0
            if action:          # HITS
                card = self.hit()
                player.draw_card(card)
            reward, terminated = self.update(player, dealer)
            print (i, 'Player', old_state, 'act:%d card:%2d' % (action, card), 'R %+d' % reward, 'sum', player.sum, player.status)
            episode.append([old_state, action, reward])
            if terminated:
                break
            if action == 0:     # STICKS
                break
        #print ('Player status', player.status, dealer.status, 'terminated', terminated)

        ### DEALER'S TURN ###
        #while dealer.status == 'playing':
        while not terminated:
            i += 1
            old_state = self.state
            action = dealer.dealer_action()
            card = 0
            if action:          # HITS
                card = self.hit()
                dealer.draw_card(card)
            reward, terminated = self.update(player, dealer) 
            print (i, 'Dealer', old_state, 'act:%d card:%2d' % (action, card), 'R %+d' % reward, 'sum', dealer.sum, dealer.status)
            if action == 0:     # When dealer sticks
                break

        ### COMPARE ###
        if not terminated:
            final_reward = np.sign(player.sum - dealer.sum)
            episode[-1][-1] = final_reward
        if len(episode) != 0:
            print ('### player:%2d' % player.sum, 'dealer:%2d' % dealer.sum, 'R:%+d' % episode[-1][-1])
        return episode

    def hit(self):

        return self.deck[random.randint(0, 12)]

    def update(self, player, dealer):

        """
Returns reward & terminated
        """
        self.state = [player.sum, self.showing_card, player.usable_ace]
        if player.status == 'natural':
            return +1, 1
        elif dealer.status == 'natural':
            return -1, 1
        elif player.status == 'bust':
            return -1, 1
        elif dealer.status == 'bust':
            return +1, 1
        else:
            return 0, 0


class Agent:

    def __init__(self):

        self.clean_cards()
        self.value = 2 * (np.random.rand(10,10,2) - 0.5)

    def __getitem__(self, state):

        # [player'sum, dealer's showing card, player's uable ace]
        return self.value[state[0]-12, state[1]-1, int(state[2])]

    def clean_cards(self):
        
        self.cards  = []
        self.sum    = 0
        self.usable_ace = False

    def get_policy(self, s):
        """
s = (players_sum, dealers_showing_card, player_usable_ace)
        """
        #return int(self[s] >=0)
        return int(random.random() > ((self[s] + 1.0)/2))

    def player_action(self):

        return (self.sum <= 19)     # player sticks when sum is 20 or 21

    def dealer_action(self):

        return (self.sum <= 16)     # dealer sticks the sum of 17 or greater

    def draw_card(self, x):

        self.cards.append(x)
        self.usable_ace = False
        if x == 1:
            ace_sum       = self.sum + 1
            eleven_sum    = self.sum + 11
            if ace_sum == 21:
                self.usable_ace = True
                self.sum = ace_sum
                self.status = 'natural'
            elif eleven_sum == 21:
                self.sum = eleven_sum
                self.usable_ace = False
                self.status = 'natural'
            elif eleven_sum <=20:    # Ace=11
                self.usable_ace = False
                self.sum = eleven_sum
                self.status = 'playing'
            elif ace_sum <=20:       # Ace=1
                self.usable_ace = True
                self.sum = ace_sum
                self.status = 'playing'
            else:
                self.sum = ace_sum
                self.status = 'bust'
        else:       # execpt an ace
            self.sum += x
            if self.sum == 21:
                self.status = 'natural'
            elif self.sum <= 20:
                self.status = 'playing'
            else:
                self.status = 'bust'
        return self.status

    def show_value(self):

        print ('USABLE ACE')
        print (np.flipud(self.value[:, :, 1]))
        print ('NO USABLE ACE')
        print (np.flipud(self.value[:, :, 0]))


class Returns:

    def __init__(self):

        self.data = {}

    def __getitem__(self, s):

        s = tuple(s)
        if not s in self.data.keys():
            self.data[s] = []
        return self.data[s]

    def __str__(self):

        return 'Returns'


def plot_surface(agent):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # x, y, z成分のデータの作成
    _x = np.arange(1, 11)
    _y = np.arange(12, 22)
    _xx, _yy = np.meshgrid(_x, _y)
    #x, y = _xx.ravel(), _yy.ravel()
    surf = ax1.plot_surface(_xx, _yy, agent.value[:,:,1])
    surf = ax2.plot_surface(_xx, _yy, agent.value[:,:,0])
    ax1.set_title('Usable ace')
    ax1.set_xlabel('Dealer showing')
    ax1.set_ylabel('Player sum')
    ax2.set_title('No usable ace')
    ax2.set_xlabel('Dealer showing')
    ax2.set_ylabel('Player sum')
    plt.savefig('bj.png')
    plt.show()


GAMMA = 1.0
def main():

    from statistics import mean
    import copy
    returns = Returns()
    env = Env()
    player = Agent()
    dealer = Agent()
    epi_nr = 0
    #for e in range(500000):
    for e in range(100000):
    #for e in range(10000):
        epi_nr += 1
        episode = env.generate_experience(player, dealer)
        G = 0
        state_list = []
        player.clean_cards()
        dealer.clean_cards()
        if epi_nr % 1000 == 0:
            print (epi_nr, episode)
        for i in range(-1, -(len(episode)+1), -1):
            #print (i, episode[i])
            state, action, reward = episode[i]
            if state[0] <= 11:      # when player sum <= 11
                continue
            G = GAMMA * G + reward 
            state_list.append(tuple(state))
            if not state in state_list:
                returns[state].append(G)
                #old_value = copy.copy(player[state])
                player.value[state[0]-12, state[1]-1, int(state[2])] =  \
                    mean(returns[state])
                """
                print ('orig %.5f %.5f' %   \
                    (player.value[state[0]-12, state[1]-1, int(state[2])],     \
                    player[state], tuple(state), player.cards)
                print ('old: %+.4f new: %+.4f diff %+.6f' %     \
                    (old_value, player[state], player[state] - old_value),  \
                    tuple(state))#, returns[state])
                """
            #print (returns.data)
    player.show_value()
    plot_surface(player)     # usable ace


main()

