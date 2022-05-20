import gym
import argparse
from datetime import datetime


from stable_baselines3.dqn.dqn import DQN
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
import traci


if __name__ == '__main__':

    # add by me
    experiment_time = str(datetime.now()).split('.')[0]

    env = SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                    route_file='nets/2way-single-intersection/single-intersection-vhvh-type-test.rou.xml',
                                    out_csv_name='outputs/{}_DQN_2way'.format(experiment_time),
                                    single_agent=True,
                                    use_gui=True,
                                    num_seconds=100000,
                                    max_depart_delay=0)

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.01,
        learning_starts=0,
        train_freq=1,
        target_update_interval=100,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1
    )
    #model.learn(total_timesteps=100000)


    #changed by me
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    prs.add_argument("-pretrain", action="store_true", default=False, help="Do you want to use pretained model?\n")

    args = prs.parse_args()

    if args.pretrain:
        model = DQN.load("outputs/last_saved_dqn_2way")
        model.set_env(env)
        model.learn(total_timesteps=100000)
    else:
        model.learn(total_timesteps=100000)


    save_model = input('Do you want to save model (Y/N) ?')
    if save_model == 'Y':
        model.save("outputs/last_saved_dqn_2way")

    env.close()


