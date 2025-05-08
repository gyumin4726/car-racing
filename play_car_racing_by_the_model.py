import argparse
import gymnasium as gym
from collections import deque
from CarRacingDQNAgent_torch import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.pth` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes

    env = gym.make('CarRacing-v3', render_mode='human')
    agent = CarRacingDQNAgent(epsilon=0) # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(train_model)

    for e in range(play_episodes):
        init_state = env.reset()
        if isinstance(init_state, tuple):
            init_state = init_state[0]
        init_state = process_state_image(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            if isinstance(action, tuple):
                action = np.array(action)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            total_reward += reward
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e+1, play_episodes, time_frame_counter, float(total_reward)))
                break
            time_frame_counter += 1
