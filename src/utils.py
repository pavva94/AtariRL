import imageio
import IPython
import base64
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output

# This function can resize to any shape, but was built to resize to 84x84
def image_preprocess_observations(frame, shape=(200, 160)):
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[10:, :160]  # crop image
    frame = frame.reshape((*shape))

    return frame

def create_video(env, model, video_filename = 'imageio'):
  def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
      <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


  num_episodes = 2
  video_filename = video_filename + ".mp4"
  with imageio.get_writer(video_filename, fps=60) as video:
    for _ in range(num_episodes):
        observation = environment.reset()

        state = image_preprocess_observations(observation)
        #take the firsts 4 step as init
        states = [state, state, state, state]

        terminated = False
        video.append_data(observation)
        while not terminated:
            agent_states = np.expand_dims(np.array(tf.cast(states, tf.float32)).reshape(model.state_size), axis=0)
            env_action = model.act(agent_states)

            # Take action
            observation, _, terminated, _ = environment.step(env_action)
            next_state = image_preprocess_observations(observation)

            next_states = states[1:]
            next_states.append(next_state)
            states = next_states
            video.append_data(observation)

  embed_mp4(video_filename)

def play_games(environment, agent):
    print(agent)
    total_reward = 0
    for i in range(4):
        observation = environment.reset()
        observation, _, _, _ = environment.step(1)

        state = image_preprocess_observations(observation)
        #take the firsts 4 step as init
        states = [state, state, state, state]

        terminated = False
        print("Start game {}".format(i))
        while not terminated:
            agent_states = np.expand_dims(np.array(tf.cast(states, tf.float32)).reshape(agent.state_size), axis=0)
            env_action = agent.act(agent_states)

            # Take action
            #print(env_action)
            observation, reward, terminated, _ = environment.step(env_action)
            next_state = image_preprocess_observations(observation)

            next_states = states[1:]
            next_states.append(next_state)
            states = next_states

            total_reward += reward
    return total_reward

def self_play(best_agent_reward, current_agent, environment):
    #best_agent_reward = play_games(environment, best_agent)
    current_agent_reward = play_games(environment, current_agent)
    print("Best reward {}".format(best_agent_reward))
    if current_agent_reward > best_agent_reward:
        print("Better model found! Saving best_model")
        current_agent.save_model(0, "Break_best_model")
        return current_agent_reward
    return current_agent_reward


def plot_bar(data, n, filename="Bar"):
    plt.figure()
    plt.bar(np.arange(n), data, align='center', alpha=0.5)
    plt.xlabel('Actions')
    plt.ylabel('Probabilities')
    plt.xticks(np.arange(7), ['NOOP', 'FIRE', "UP", 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'])
    plt.title('Histogram of Actions choosen')
    plt.grid(True)
    plt.savefig(filename + ".png")
    plt.close()


def plot_graph(data, filename="Graph"):
    plt.figure()
    plt.plot(data)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.xticks(np.arange(len(data)))
    plt.title('Episode Rewards')
    plt.grid(True)
    plt.savefig(filename + ".png")
    plt.close()


def plot_mult_bar(data1, data2, n, filename="MultiBar"):
    plt.figure()
    w = 0.3
    plt.bar(np.arange(n), data1, width=w, align='center', alpha=0.5)
    plt.bar(np.arange(n), data2, width=w, align='center', alpha=0.5)
    plt.xlabel('Actions')
    plt.ylabel('Probabilities')
    plt.xticks(np.arange(7), ['NOOP', 'FIRE', "UP", 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'])
    plt.title('Histogram of Actions choosen')
    plt.grid(True)
    plt.savefig(filename + ".png")
    plt.close()


def plot_multi_graph(data1, data2, filename="MultiGraph"):
    plt.figure()
    plt.plot(data1)
    plt.plot(data2)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.xticks(np.arange(len(data1)))
    plt.title('Episode Rewards')
    plt.grid(True)
    plt.savefig(filename + ".png")
    plt.close()
