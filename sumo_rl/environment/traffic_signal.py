import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gym import spaces


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = 0
        self.last_measure = 0.0
        self.last_reward = None
        self.phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0].phases
        self.num_green_phases = len(self.phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order only in lane
        self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes)) # only out lane
        self.vehicle_base_co2 = 2500 # add by me
        self.HDV_count = 2
        self.LDV_count = 0.05
        self.Bus_count = 1

        """
        Default observation space is a vector R^(#greenPhases + 2 * #lanes)
        s = [current phase one-hot encoded, density for each lane, queue for each lane]
        You can change this by modifing self.observation_space and the method _compute_observations()

        Action space is which green phase is going to be open for the next delta_time seconds
        """
        # change by me
        #self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 2*len(self.lanes)), high=np.ones(self.num_green_phases + 2*len(self.lanes)))
        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 3*len(self.lanes)), high=np.ones(self.num_green_phases + 3*len(self.lanes)))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),                       # Green Phase
            #spaces.Discrete(self.max_green//self.delta_time),            # Elapsed time of phase
            *(spaces.Discrete(10) for _ in range(2*len(self.lanes)))      # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

        logic = traci.trafficlight.Logic("new-program"+self.id, 0, 0, phases=self.phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step
    
    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            traci.trafficlight.setPhase(self.id, self.green_phase)
            self.is_yellow = False

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases] 
        """
        new_phase *= 2
        if self.phase == new_phase or self.time_since_last_phase_change < self.min_green + self.yellow_time:
            self.green_phase = self.phase
            traci.trafficlight.setPhase(self.id, self.green_phase)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            self.green_phase = new_phase
            traci.trafficlight.setPhase(self.id, self.phase + 1)  # turns yellow
            self.next_action_time = self.env.sim_step + self.delta_time + self.yellow_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    
    def compute_observation(self):
        phase_id = [1 if self.phase//2 == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        #elapsed = self.traffic_signals[ts].time_on_phase / self.max_green
        density = self.get_lanes_density()
        ##co2 = self.get_lanes_emission()
        queue = self.get_lanes_queue()
        # add by me
        type = self.get_lanes_emission_norm(self.lanes)  # average type is between 0 (PC) - 1 (LDV) for each lane
        observation = np.array(phase_id + density + queue + type)
        ##observation = np.array(phase_id + co2 + queue)
        return observation
            
    def compute_reward(self):
        #self.last_reward = self._waiting_time_reward()
        #self.last_reward = self._queue_reward()
        #self.last_reward = self._queue_average_reward()
        #self.last_reward = -self.get_pressure()
        #self.last_reward = -self.pressure_co2_reward()
        #self.last_reward = self.pressure_co2_reward2()
        #self.last_reward = -self.pressure_co2_reward3()
        #self.last_reward = -self.pressure_co2_reward4()
        self.last_reward = self.co2_queue_reward()
        #self.last_reward = self.co2_reward()
        #self.last_reward = self._weighted_waiting_time_reward()
        return self.last_reward


    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return - (sum(self.get_stopped_vehicles_num()))**2

    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        #ts_wait = sum(self.get_waiting_time())
        # change by me
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0/ts_wait
        return reward

    def _waiting_time_reward3(self):
        #ts_wait = sum(self.get_waiting_time())
        # change by me
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_pressure(self):
        return abs(sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]



    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]
    
    def get_total_queued(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

    ## Added By me
    def change_veh_class(self):
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                traci.vehicle.setEmissionClass(veh, "HBEFA3/PC_G_EU4")  # HBEFA3/HDV_D_EU4
                a = traci.vehicle.getCO2Emission(veh)


    def get_lane_weight(self,lanes):
        weights = []
        for lane in lanes:
            count = 0
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                if traci.vehicle.getEmissionClass(veh) == "HBEFA3/HDV":
                    count += self.HDV_count
                elif traci.vehicle.getEmissionClass(veh) == "HBEFA3/LDV":
                    count += self.LDV_count
                elif traci.vehicle.getEmissionClass(veh) == "HBEFA3/Bus":
                    count += self.Bus_count
            weights.append((len(veh_list)+count)/max(1,len(veh_list)))
        return weights


    def _weighted_waiting_time_reward(self):
        weighted_wait = [a * b  for a, b in zip(self.get_waiting_time_per_lane(), self.get_lane_weight(self.lanes))]
        ts_wait = sum(weighted_wait) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward


    def get_stopped_vehicles_num(self):
        return [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes]

    def get_lanes_emission_norm(self,lanes):
        vehicle_base_max = self.vehicle_base_co2 +500  # base value
        return [ max(0,min(1,(traci.lane.getCO2Emission(lane)-self.vehicle_base_co2) / vehicle_base_max/
                           max(1,traci.lane.getLastStepVehicleNumber(lane)))) for lane in lanes]


    def get_lanes_emission(self):
        return [ traci.lane.getCO2Emission(lane) / self.vehicle_base_co2 for
                lane in self.lanes]

    def get_out_lanes_emission(self):
        return [ traci.lane.getCO2Emission(lane) / self.vehicle_base_co2 for
                lane in self.out_lanes]

    def pressure_co2_reward(self):
        # return abs(sum(self.get_lanes_emission())-sum(self.get_out_lanes_emission()))
        in_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]
        in_weighted_pressure = [a *(b+1) for a, b in zip(in_pressure, self.get_lanes_emission_norm(self.lanes))]
        out_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes]
        out_weighted_pressure = [a * (b + 1) for a, b in zip(out_pressure, self.get_lanes_emission_norm(self.out_lanes))]
        return abs(sum(in_weighted_pressure) - sum(out_weighted_pressure))


    def pressure_co2_reward2(self):
        new_average = abs(sum(self.get_lanes_emission())-sum(self.get_out_lanes_emission()))
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def pressure_co2_reward3(self):
        return abs(sum(traci.lane.getCO2Emission(lane)/self.vehicle_base_co2 for lane in self.lanes) -
                   sum(traci.lane.getLastStepVehicleNumber(lane)/self.vehicle_base_co2 for lane in self.out_lanes))

    def pressure_co2_reward4(self):
        in_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]
        in_weighted_pressure = [a *b for a, b in zip(in_pressure, self.get_lane_weight(self.lanes))]
        out_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes]
        out_weighted_pressure = [a * b  for a, b in zip(out_pressure, self.get_lane_weight(self.out_lanes))]
        return abs(sum(in_weighted_pressure) - sum(out_weighted_pressure))

    def co2_reward(self):
        alpha = 0 # weighting
        new_co2 = self.get_total_emission()/self.vehicle_base_co2
        reward = self.last_measure - new_co2
        self.last_measure = new_co2
        return reward - alpha*self.get_pressure()


    def co2_queue_reward(self):
        queue = self.get_stopped_vehicles_num()
        weight = [ (traci.lane.getCO2Emission(lane) / self.vehicle_base_co2/
                           max(1,traci.lane.getLastStepVehicleNumber(lane))) for lane in self.lanes]
        #weighted_queue = [a * (b+1)  for a, b in zip(queue, self.get_lanes_emission_norm(self.lanes))]
        weighted_queue = [a * b  for a, b in zip(queue, weight)]
        #weighted_queue = [a * b  for a, b in zip(queue, self.get_lane_weight(self.lanes))]
        return - (sum(weighted_queue))**2

    def get_total_emission(self):
        return sum([traci.lane.getCO2Emission(lane) for lane in self.lanes])

    def get_mean_emission(self):
        a = sum([traci.lane.getCO2Emission(lane) for lane in self.lanes])/1000
        n_t = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes])
        return a/n_t if n_t else 0

    def get_total_travel_time(self):
        # return sum([traci.lane.getTraveltime (lane) for lane in self.lanes])
        v_wa = sum([traci.lane.getLastStepMeanSpeed(lane)*traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes])
        n_t = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes])
        l_t = [traci.lane.getLength(lane) for lane in self.lanes][0] # only the ingoing lane
        return n_t*l_t/v_wa if v_wa>1 else 0