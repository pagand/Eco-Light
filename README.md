<img src="outputs/logo.png" align="right" width="30%"/>



# EcoLight

EcoLight is a ecosystem friendly RL appraoch for traffic control signal. The code is based on [SUMO-RL](https://github.com/LucasAlegre/sumo-rl)

More information: https://upaspro.com/ecolight/

SUMO-RL provides a simple interface to instantiate Reinforcement Learning environments with [SUMO](https://github.com/eclipse/sumo) for Traffic Signal Control. 

The main class [SumoEnvironment](/sumo_rl/environment/env.py) inherits [MultiAgentEnv](https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py) from [RLlib](https://github.com/ray-project/ray/tree/master/python/ray/rllib).  
If instantiated with parameter 'single-agent=True', it behaves like a regular [Gym Env](https://github.com/openai/gym/blob/master/gym/core.py) from [OpenAI](https://github.com/openai).  
[TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/master/environment/traffic_signal.py) is responsible for retrieving information and actuating on traffic lights using [TraCI](https://sumo.dlr.de/wiki/TraCI) API.

Goals of this repository:
- Provide a simple interface to work with Reinforcement Learning for Traffic Signal Control using SUMO
- Support Multiagent RL
- Compatibility with gym.Env and popular RL libraries such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [RLlib](https://docs.ray.io/en/master/rllib.html)
- Easy customisation: state and reward definitions are easily modifiable

## Install

### Install SUMO latest version:

```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc 
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

### Install SUMO-RL

Stable release version is available through pip
```
pip install sumo-rl
```

Alternatively you can install using the latest (unreleased) version
```
git clone https://github.com/LucasAlegre/sumo-rl
cd sumo-rl
pip install -e .
```
### Run EcoLight
```
git clone https://github.com/pagand/ecolight
cd ecolight
python3 run[...].py
```



## Examples

Check [experiments](/experiments) to see how to instantiate a SumoEnvironment and use it with your RL algorithm.

### [Q-learning](/sumo-rl/agents/ql_agent.py) in a one-way single intersection:
```
python3 experiments/ql_single-intersection.py 
```

### [RLlib A3C](https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/a3c) multiagent in a 4x4 grid:
```
python3 experiments/a3c_4x4grid.py
```

### [stable-baselines3 DQN](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py) in a 2-way single intersection:
```
python3 experiments/dqn_2way-single-intersection.py
```

### Plotting results:
```
python3 outputs/plot.py -f outputs/2way-single-intersection/a3c 
```
![alt text](outputs/result.png)




### How to cite:

  @article{agandecolight,
  title={EcoLight: Reward Shaping in Deep Reinforcement Learning for Ergonomic Traffic Signal Control},
  author={Agand, Pedram and Iskrov, Alexey},
  booktitle={NeurIPS 2021 Workshop on Tackling Climate Change with Machine Learning},
  year={2021}
}
