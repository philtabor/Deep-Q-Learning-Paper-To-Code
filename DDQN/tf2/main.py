import numpy as np
from agent import Agent
from utils import plot_learning_curve, make_env, manage_memory
from gym import wrappers

if __name__ == '__main__':
    manage_memory()
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    record_agent = False
    n_games = 250
    agent = Agent(gamma=0.99, epsilon=1, lr=0.0001,
                  input_dims=(env.observation_space.shape),
                  n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                  batch_size=32, replace=1000, eps_dec=1e-5,
                  chkpt_dir='models/', algo='DQNAgent',
                  env_name='PongNoFrameskip-v4')
    if load_checkpoint:
        agent.load_models()
        agent.epsilon = agent.eps_min

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
        + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a
    # mkdir video
    if record_agent:
        env = wrappers.Monitor(env, "video",
                               video_callable=lambda episode_id: True,
                               force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode {} score {:.1f} avg score {:.1f} '
              'best score {:.1f} epsilon {:.2f} steps {}'.
              format(i, score, avg_score, best_score, agent.epsilon,
                     n_steps))

        if score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
