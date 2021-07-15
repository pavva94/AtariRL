# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import gym

from DQN import DQNAgent
from A2C import ACAgent
from utils import image_preprocess_observations, create_video, self_play, plot_bar, plot_graph, plot_mult_bar, plot_multi_graph


def main(args):
    print("You are entering the RL Project by Alessandro Pavesi")
    model_sel = 0
    while model_sel not in [1, 2]:
        model_sel = int(input("Firstly select the agent yoyu want: 1: DQN, 2: A2C"))

    print("\nWhat you want to do?")
    print("1. train one agent")
    print("2. create a video for a trained agent")
    print("3. create graphs for a trained agent")
    menu_sel = int(input())

    train_agent = False
    create_video_agent = False
    create_graph_agent = False
    # play parameters
    num_of_episodes = 50000
    timesteps_per_episode = 1500
    cont_e = 0

    if menu_sel == 1:
        print("For training the agent insert the numbero of episode:")
        num_of_episodes = int(input())
        train_agent = True
    elif menu_sel == 2:
        print("Creating a video for the best agent")
        create_video_agent = True
    elif menu_sel == 3:
        print("Creating the graphs for the best agent")
        create_graph_agent = True


    environment = gym.make("Assault-v0")

    #print('Number of states: {}'.format(environment.observation_space))
    #print('Number of actions: {}'.format(environment.action_space))
    #print(environment.unwrapped.get_action_meanings())

    if model_sel == 1:
        # DQNAgent
        if train_agent:
            dqn_agent = DQNAgent(num_of_episodes, greedy=False)
            #training dqn
            total_reward_dqn, total_action_dqn = dqn_agent.play(environment, num_of_episodes, timesteps_per_episode, train=True)

        if create_video_agent:
            # Creating Video
            best_agent = DQNAgent(1, True)
            dqn_filename = "Assault_best_weights"
            best_agent.load_model(dqn_filename)
            create_video(environment, best_agent, "DQN_video")

        if create_graph_agent:

            # Evaluation
            num_of_evaluation_step = 10
            timesteps_per_episode = 1500

            dqn_agent = DQNAgent(num_of_evaluation_step, greedy=True)
            dqn_model_file = "Assault_best_weights"
            dqn_agent.load_model(dqn_model_file)

            total_reward_dqn, total_action_dqn = dqn_agent.play(environment, num_of_evaluation_step, timesteps_per_episode, train=False)
            actions_count_dqn = [total_action_dqn.count(0), total_action_dqn.count(1), total_action_dqn.count(2), total_action_dqn.count(3), total_action_dqn.count(4), total_action_dqn.count(5), total_action_dqn.count(6)]
            plot_bar(actions_count_dqn, len(actions_count_dqn), "ActionsBarDQN"+str(num_of_evaluation_step))
            plot_graph(total_reward_dqn, filename="RewardGraphDQN"+str(num_of_evaluation_step))



    else:
        if train_agent:
            # A2CAgent
            ac_agent = ACAgent()
            # training a2c
            total_reward_a2c, total_action_a2c = ac_agent.play(environment, num_of_episodes, timesteps_per_episode, train=True)

        if create_video_agent:
            # Video Creation
            ac_loaded_model = ACAgent()
            ac_filename = "AC_best_weights"
            ac_loaded_model.load_model(ac_filename)
            create_video(environment, ac_loaded_model, "AC_video")

        if create_graph_agent:
            # Evaluation
            environment.reset()

            ac_loaded_model = ACAgent()
            ac_filename = "AC_best_weights"
            ac_loaded_model.load_model(ac_filename)

            total_reward_a2c, total_action_a2c = ac_loaded_model.play(environment, num_of_evaluation_step, timesteps_per_episode, train=False)
            actions_count_a2c = [total_action_a2c.count(0), total_action_a2c.count(1), total_action_a2c.count(2), total_action_a2c.count(3), total_action_a2c.count(4), total_action_a2c.count(5), total_action_a2c.count(6)]
            plot_bar(actions_count_a2c, len(actions_count_a2c), "ActionsBarA2C750"+str(num_of_evaluation_step))
            plot_graph(total_reward_a2c, filename="RewardGraphA2C750"+str(num_of_evaluation_step))


    #plot_multi_graph(total_reward_dqn, total_reward_a2c, filename="RewardGraphCompare")
    #plot_mult_bar(actions_count_dqn, actions_count_a2c, len(actions_count_dqn), filename="ActionBarCompare")

if __name__ == '__main__':
    main(None)
