# Eco-Light white paper


### Stable Baseline:

Stable Baselines is a set of improved implementations of reinforcement learning algorithms based on OpenAI. 

Some algorithms (like PPO2) are capable of running in GPU.

The network architecture is Tensorflow.


### Stable Baseline3:

Stable Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch.

Stable-Baselines3 supports PyTorch 1.4+ and python 3.6+


```
1. import gym
2. 
3. from stable_baselines3 import PPO
4. 
5. env = gym.make("CartPole-v1")
6. 
7. model = PPO("MlpPolicy", env, verbose=1)
8. model.learn(total_timesteps=10000)
9. 
10. obs = env.reset()
11. for i in range(1000):
12.     action, _states = model.predict(obs, deterministic=True)
13.     obs, reward, done, info = env.step(action)
14.     env.render()
15.     if done:
16.       obs = env.reset()
17. env.close()
```

Get the current emission: this is the api from Traci:

sum([ traci.lane.getCO2Emission(lane)  **for **lane **in **self.lanes])


### The total stopped over time:

Returns the total number of halting vehicles for the last time step on the given edge. A speed of less than 0.1 m/s is considered a halt.

sum([traci.lane.getLastStepHaltingNumber(lane) **for **lane **in **self.lanes])

The total wait time over time:

traci.vehicle.getAccumulatedWaitingTime(veh)

The total travel time:

Edge value retrieval

Returns the current travel time (length/mean speed).

sum([traci.lane.getTraveltime (lane) **for **lane **in **self.lanes])

State: a vector containing the traffic light phases with density and number of queue for each lane

E.g. in  a 2 way intersection with one traffic light, the state vector has 9 elements. 

For phases it uses one-hot encoded

phase_id = [1 **if **self.phase//2 == i **else **0 **for **i **in **range(self.num_green_phases)]

Density is the number of vehicle is a specific lane divided by the total number of vehicle that can fit in that lane:

traci.lane.getLastStepVehicleNumber(lane)

Returns the number of vehicles that were on the named induction loop within the last simulation step [#];

Lane queue is the ratio of number of cars that are halted to the total number of cars that can stop in the lane

traci.lane.getLastStepHaltingNumber(lane)

Return number of vehicle with speed less than 0.1. 


### The problem with travel time:

traci.lane.getTraveltime

Compute the travel time using the following relation:

length/ave_speed

The problem is that, if a lane has no moving vehicle, there would be a large value for travel time.

Solution: we redefine the travel time:

Travel_time = ( <span style="text-decoration:underline;">L</span> * N_t)/(sum [ <span style="text-decoration:underline;">V</span>_{lt}* N_{lt}])

<span style="text-decoration:underline;">L</span>: average trip length

traci.lane.getLength(lane)

N_t: total num of cars in last step

<span style="text-decoration:underline;">V</span>_{lt}: average velocity of lane l at last step

traci.lane.getLastStepMeanSpeed(lane)

N_{lt}: total num of cars of lane l in last step

traci.lane.getLastStepVehicleNumber(lane)

The speed is considered to be 50 km/h or 13.9 m/s. The average trip length is 300 m. Hence, the fastest travel time is 21.58 Sec.


### Co2 emission:

You can get the current emitted co2 from each vehicle directly via:

[ traci.vehicle.getCO2Emission(veh)  **for **veh **in **self.env.vehicles])


### Q-table:

Policy is epsilon-greedy and the network structure is Q-table.

Network structure:

Alpha learning rate:

alpha=0.1

Gamma discount factor: 

gamma=0.99

epsilon=0.05

min_epsilon=0.005

decay=1.0

Time of simulation:

seconds=100000


### Reward: queue average reward

It tries to minimize the number of stopped vehicle:

new_average = np.mean(self.get_stopped_vehicles_num())

reward = - new_average

In order to make the reward normalize, it also consider the previous queue, and try to maximize the difference:

reward = self.last_measure - new_average

The stopped vehicle are obtained from traci:

traci.lane.getLastStepHaltingNumber(lane)


### New reward:

- (sum(self.get_stopped_vehicles_num()))**2

Considering the waiting time as the reward:

ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0

reward = self.last_measure - ts_wait


#### Queue length t1: 

reward = - (sum(self.get_stopped_vehicles_num()))**2


#### Queue length t2: 

new_average = np.mean(self.get_stopped_vehicles_num())

reward = self.last_measure - new_average


#### Co2 pressure t1:

reward=abs(sum(self.get_lanes_emission())-sum(self.get_out_lanes_emission()))


#### Co2 pressure t2:

new_average = abs(sum(self.get_lanes_emission())-sum(self.get_out_lanes_emission()))

reward = self.last_measure - new_average

self.last_measure = new_average


#### Co2 difference:

new_co2 = self.get_total_emission()/vehicle_base_co2

reward = self.last_measure - new_co2

self.last_measure = new_co2


### Available approaches to change the emission class:


#### 1- using the attributes in vType


```
<routes>
    <vType id="type1" length="5" maxSpeed="70" emissionClass="HBEFA3/Bus" carFollowModel="Krauss" accel="2.6" decel="4.5" sigma="0.5"/>
</routes>
```


Generally we have:


```
emissionClass="<model>/<class>"
```


For the model [HBEFA v3.1-based](https://sumo.dlr.de/docs/Models/Emissions/HBEFA3-based.html) we have:


<table>
  <tr>
   <td>Bus
   </td>
   <td>average urban bus (all fuel types)
   </td>
  </tr>
  <tr>
   <td>Coach
   </td>
   <td>average long distance bus (all fuel types)
   </td>
  </tr>
  <tr>
   <td>HDV
   </td>
   <td>average heavy duty vehicle (all fuel types)
   </td>
  </tr>
  <tr>
   <td>HDV_D_EU4
   </td>
   <td>diesel driven heavy duty vehicle Euro norm 4
   </td>
  </tr>
  <tr>
   <td>zero
   </td>
   <td>zero emission vehicle
   </td>
  </tr>
  <tr>
   <td>LDV
   </td>
   <td>average light duty vehicles (all fuel types)
   </td>
  </tr>
  <tr>
   <td>PC
   </td>
   <td>average passenger car (all fuel types)
   </td>
  </tr>
  <tr>
   <td>PC_G_EU4
   </td>
   <td>gasoline driven passenger car Euro norm 4 (default)
   </td>
  </tr>
</table>


Flow is used when we want repeated vehicle (same as sumo-rl)

It is possible to define repeated vehicle emissions ("flow"s), which have the same parameters as the vehicle or trip definitions except for the departure time. The id of the created vehicles is "flowId.runningNumber" and they are distributed either equally or randomly in the given interval. 


<table>
  <tr>
   <td>begin/end
   </td>
   <td>first/last vehicle departure time
   </td>
  </tr>
  <tr>
   <td>vehsPerHour
   </td>
   <td>number of vehicles per hour, equally spaced (not together with period or probability)
   </td>
  </tr>
  <tr>
   <td>period
   </td>
   <td>insert equally spaced vehicles at that period (not together with vehsPerHour or probability)
   </td>
  </tr>
  <tr>
   <td>probability
   </td>
   <td>probability for emitting a vehicle each second (not together with vehsPerHour or period)
   </td>
  </tr>
  <tr>
   <td>number
   </td>
   <td>total number of vehicles, equally spaced
   </td>
  </tr>
</table>



#### The second approach by using tracI:

Changing vehicle type state

Run-time execution

traci.vehicle.setEmissionClass(vehid,”HBEFA3/Bus”)

Given the vehicles that are launched to the simulation, we can set them to be in a different class using this code:

I defined a method in the class, which go through each id of vehicles and set their emission class:



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")


The problem with this is that, traci is applied to the simulation after each simulation step. Hereby, during the time, the vehicle is launched already but the traci is able to change the class only after a certain time, when the step time arrives.

So when we set the emission class to zero for all the vehicles, the total emitted co2 is not exactly zero, it’s something around 3000 which is not what we want.

I have add four different type of vehicle and currently integrating two of passenger car and truck (HDV) into concurrent flow. In the simulation, sometimes there would be a collision between 2 vehicles, but it omits one of them

&lt;vType id="electric" emissionClass="zero" color="1,0,0" accel="0.8" decel="4.5"/>  &lt;!--HBEFA3/Bus-->

&lt;vType id="bus" emissionClass="HBEFA3/Bus" color="0,1,0" accel="0.4" decel="2.2"/>  &lt;!--Bus-->

&lt;vType id="truck" emissionClass="HBEFA3/HDV" color="0,0,1" accel="0.3" decel="2"/>  &lt;!--heavy duty-->

&lt;vType id="car"  color="1,1,1" accel="0.8" decel="4.5" />  &lt;!--HBEFA3/Bus-->

&lt;flow id="flow_nsc" route="route_ns" type="car" begin="0" end="100000" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>

&lt;flow id="flow_wec" route="route_we" type="car" begin="0" end="100000" probability="0.4" departSpeed="max" departPos="base" departLane="best"/>

&lt;flow id="flow_nst" route="route_ns" type="truck" begin="0" end="100000" probability="0.02" departSpeed="max" departPos="base" departLane="best"/>

&lt;flow id="flow_wet" route="route_we" type="truck" begin="0" end="100000" probability="0.05" departSpeed="max" departPos="base" departLane="best"/>


### normalized lane emission 

[ max(0,min(1,(traci.lane.getCO2Emission(lane)-self.vehicle_base_co2) / vehicle_base_max/

                  max(1,traci.lane.getLastStepVehicleNumber(lane)))) **for **lane **in **self.lanes]


### Co2 pressure t1:

We compute in/out pressure given the number of vehicles. But we finally consider the weighted version according to their emission base norm.

in_pressure = [traci.lane.getLastStepVehicleNumber(lane) **for **lane **in **self.lanes]

in_weighted_pressure = [a *(b+1) **for **a, b **in **zip(in_pressure, self.get_lanes_emission_norm())]

out_pressure = [traci.lane.getLastStepVehicleNumber(lane) **for **lane **in **self.out_lanes]

out_weighted_pressure = [a * (b + 1) **for **a, b **in **zip(out_pressure, self.get_out_lanes_emission_norm())]

We finally return the difference of their sum as reward:

**Reward = **abs(sum(in_weighted_pressure) - sum(out_weighted_pressure))


### Co2 pressure t2:

In this case, we compute the emission for in/out lane, but instead of minimizing it directly, we do it via the difference to the previous measurement

new_average = abs(sum(self.get_lanes_emission())-sum(self.get_out_lanes_emission()))

reward = self.last_measure - new_average


### Co2 pressure t3:

In here we only have the translation of pressure for emission, hence it is computed as follows:

abs(sum(traci.lane.getCO2Emission(lane)/self.vehicle_base_co2 **for **lane **in **self.lanes) -

          sum(traci.lane.getLastStepVehicleNumber(lane)/self.vehicle_base_co2 **for **lane **in **self.out_lanes))


### CO2 reward:

The reward in here is purely based on the co2 emission in a difference manner:

new_co2 = self.get_total_emission()/self.vehicle_base_co2

We also keep track of the previous measurement, and consider the difference as the reward:

reward = self.last_measure - new_co2


### Weighted queue:

weighted_queue = [a * (b+1)  **for **a, b **in **zip(queue, self.get_lanes_emission_norm())]

We consider the queue length of the vehicle in halting situation which is multiplied to the normalized lane emission.

Finally the reward is as follows:

**Reward =  **- (sum(weighted_queue))**2


### New scenario: 

Duration of the simulation is 11000 sec with two flows and some constant vehicle randomly.
```

&lt;flow id="flow_nsc1" route="route_ns" type="car" begin="9918" end="11000" period="14" departSpeed="max" departPos="base" departLane="best"/>

&lt;flow id="flow_wec1" route="route_we" type="car" begin="9999" end="11000" period="13" departSpeed="max" departPos="base" departLane="best"/>
```

Compute the vehicle lane weight based on the type:

In here we consider an additional value (count) for truck or any other vehicle other than passenger. Then compute the normalized value for each lane
```
weights = []
for lane in self.lanes:
   count = 0
   veh_list = traci.lane.getLastStepVehicleIDs(lane)
   for veh in veh_list:
       if traci.vehicle.getEmissionClass(veh) == "HBEFA3/HDV":
           count += 10
   weights.append((len(veh_list)+count)/max(1,len(veh_list)))
return weights
```

### Weighted queue t2: 

Is considering the non-normalized weight based on the co2 emission as follows:
```

weight = [ (traci.lane.getCO2Emission(lane) / self.vehicle_base_co2/

                  max(1,traci.lane.getLastStepVehicleNumber(lane))) **for **lane **in **self.lanes]

weighted_queue = [a * b  **for **a, b **in **zip(queue, self.get_lane_weight())]
```


### Weighted queue t3: 

Compute the queue length given the weight according to the type of vehicle:
```

weighted_queue = [a * b  **for **a, b **in **zip(queue, self.get_lane_weight())]
```


### Weighted waiting time:

The weighted waiting time is basically same as weighting time, however, it has the corresponding weights added to it. Count +=10
```

weighted_wait = [a * b  **for **a, b **in **zip(self.get_waiting_time_per_lane(), self.get_lane_weight())]

ts_wait = sum(weighted_wait) / 100.0

reward = self.last_measure - ts_wait

self.last_measure = ts_wait
```


### Add a flow of bus:
```
flow id="flow_nst" route="route_ns" type="bus" begin="0" end="11000" period="20" departSpeed="max" departPos="base" departLane="best"/>
```

We use stable baseline3 for the DRL approach

I add the following code to run from the pretrained model

And also at the end, we ask if it wants to save the model:

```
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

```


### Compute the weighted pressure

 according to lane weight for in/outgoing lanes:
```
in_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]
in_weighted_pressure = [a *b for a, b in zip(in_pressure, self.get_lane_weight(self.lanes))]
out_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes]
out_weighted_pressure = [a * b  for a, b in zip(out_pressure, self.get_lane_weight(self.out_lanes))]
return abs(sum(in_weighted_pressure) - sum(out_weighted_pressure))
```

### Weighted Queue length with sarsa:
```
weight = [ (traci.lane.getCO2Emission(lane) / self.vehicle_base_co2/ max(1,traci.lane.getLastStepVehicleNumber(lane)))
```
