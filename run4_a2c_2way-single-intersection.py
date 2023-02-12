import os
import sys
from datetime import datetime 

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
from sumo_rl.util.gen_route import write_route_file
import traci

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C


if __name__ == '__main__':

    write_route_file('nets/2way-single-intersection/single-intersection-gen.rou.xml', 400000, 100000)
    experiment_time = str(datetime.now()).split('.')[0]
    # multiprocess environment
    n_cpu = 1
    env = SubprocVecEnv([lambda: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                        #route_file='nets/2way-single-intersection/single-intersection-gen.rou.xml',
                                        route_file='nets/2way-single-intersection/single-intersection-vhvh-type-test.rou.xml',
                                        out_csv_name='outputs/a2c_2way-single-intersection_{}'.format(experiment_time),
                                        single_agent=True,
                                        use_gui=True,
                                        num_seconds=100000,
                                        min_green=10,
                                        max_depart_delay=0) for _ in range(n_cpu)])

    model = A2C(MlpPolicy, env, verbose=0, learning_rate=0.001, lr_schedule='constant')
    model.learn(total_timesteps=100000)
