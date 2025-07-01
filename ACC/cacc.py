import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ACCEnv():
    def __init__(self):
        self.min_action = -3.0  # minimum acceleration 
        self.max_action = 3.0   # maximum acceleration
        self.min_speed = 0.0
        self.max_speed = 32.0
        self.min_rel_speed = -8.0
        self.max_rel_speed = 8.0
        self.min_spacing = 5.0    #space between two vehicles
        self.max_spacing = 200.0

        self.low_state = np.array(
            [self.min_speed, self.min_spacing, self.min_rel_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_speed, self.max_spacing, self.max_rel_speed], dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        self.delta_t = 0.1
        self.is_stall = 0
        self.is_collision = 0
        self.ttc_threshold = 4.0
        self.penalty = 100.0   

        self.w_jerk = 1.0
        self.w_ttc = 1.0
        self.w_timehwy = 1.0
        self.collision_count = 0

        self.w_ss = 0.1
        self.ss_gamma = 0.8
        

    def step(self, action: np.ndarray):
        lv_speed = self.lv_speed[self.timestep]
        self.timestep += 1

        # calculate next_state
        # next_speed means the speed of ego vehicle in the next state
        # next_spacing means the distance between ego and lead vehicle in next state
        next_speed = self.state[0] + action[0]*self.delta_t   #state[0] = speed of ego vehicle current state 
        next_rel_speed = lv_speed - next_speed
        next_spacing = self.state[1] + next_rel_speed*self.delta_t  #state[1] = spacing

        if next_speed <= 0:
            next_speed = 0.00001
            self.is_stall = 1
        else:
            self.is_stall = 0

        if next_spacing < self.min_spacing:
            self.is_collision = 1
            print('collision!')
            self.collision_count += 1

        # calculate reward
        if next_rel_speed < 0:
            ttc = -next_spacing/next_rel_speed
        else:
            ttc = 10000

        timehwy = next_spacing/next_speed
        jerk = (action[0] - self.last_action) / self.delta_t
        self.last_action = action[0]

        if ttc >= 0 and ttc <= self.ttc_threshold:
            r_ttc = np.log(ttc/self.ttc_threshold) 
        else:
            r_ttc = 0.0
        
        # mu = 0.5  
        # sigma = 1.0

        mu = 0.422618
        sigma = 0.43659

        desired_headway = np.exp(mu-sigma**2)
        if timehwy <= 0:
            r_timehwy = -1.0
        else:
            r_timehwy = (np.exp(-(np.log(timehwy) - mu) ** 2 / (2 * sigma ** 2)) / (timehwy * sigma * np.sqrt(2 * np.pi)))

        r_jerk = -(jerk ** 2)/3600.0

        if timehwy < desired_headway and next_speed > lv_speed:
            r_ss = -(next_speed-self.ss_gamma*lv_speed)**2
        else:
            r_ss = 0.0
        
        # calculate the reward
        reward = self.w_jerk*r_jerk + self.w_ttc*r_ttc + self.w_timehwy*r_timehwy - self.penalty * self.is_collision - self.penalty * self.is_stall + self.w_ss*r_ss

        # check terminal state
        if self.timestep == self.lv_speed.shape[0] or self.is_collision == 1:
            done = True
        else:
            done = False

        self.state = np.array([next_speed, next_spacing, next_rel_speed], dtype=np.float32) # next state

        return self.state, reward, done, False, {}

    def reset(self, data):
        self.timestep = 0
        self.lv_speed = data[:, 3] # spacing, ego vehicle speed, relative speed, lead vehicle speed
        self.state = np.array([data[0, 1], data[0, 0], data[0, 2]], dtype=np.float32)  # spacing, cur_speed, rel_speed (lead-ego)
        self.is_stall = 0   #depends on next_speed
        self.is_collision = 0   #depends on next_spacing
        self.last_action = 0.0
        self.collision_count = 0

        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        raise NotImplementedError
    






class CACCEnv():
    def __init__(self, prediction_method):
        self.min_action = -3.0  # minimum acceleration 
        self.max_action = 3.0   # maximum acceleration
        self.min_speed = 0.0
        self.max_speed = 32.0
        self.min_rel_speed = -8.0
        self.max_rel_speed = 8.0
        self.min_spacing = 5.0    #space between two vehicles
        self.max_spacing = 200.0
        self.prediction_horizon = 30 + 1
        self.prediction_method = prediction_method # 'constant', 'idm', or 'data'
        self.prediction_rate = 1 # how often we predict in terms of frames (1 means every frame)
        self.delta_t = 0.1
        self.gamma = 0.95
       

        self.low_state = np.array(
            [self.min_speed, self.min_spacing, *[self.min_rel_speed]*int(1+(self.prediction_horizon-1)/self.prediction_rate)], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_speed, self.max_spacing, *[self.max_rel_speed]*int(1+(self.prediction_horizon-1)/self.prediction_rate)], dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        self.is_stall = 0
        self.is_collision = 0
        self.ttc_threshold = 4.0
        self.penalty = 100.0   

        self.w_jerk = 2.0
        self.w_ttc = 1.0
        self.w_timehwy = 1.0
        self.collision_count = 0

        self.w_ss = 0.1
        self.ss_gamma = 0.8
        self.ss_window = 30

        

    def step(self, action: np.ndarray):
        if self.prediction_method == 'constant':
            lv_speed = self.lv_speed[self.timestep] * np.ones(self.prediction_horizon)
        elif self.prediction_method == 'data':
            lv_speed = self.lv_speed[np.minimum(self.timestep + np.arange(0, self.prediction_horizon), self.lv_speed.shape[0]-1)] 
        elif self.prediction_method == 'idm':
            raise NotImplementedError
        
        self.timestep += 1
        next_speed = np.zeros(self.prediction_horizon)
        next_spacing = np.zeros(self.prediction_horizon)
        next_rel_speed = np.zeros(self.prediction_horizon)
        #next_acc = np.zeros(self.prediction_horizon)

        next_speed[0] = self.state[0] + action[0]*self.delta_t   #state[0] = speed of ego vehicle current state 
        next_rel_speed[0] = lv_speed[0] - next_speed[0]
        next_spacing[0] = self.state[1] + next_rel_speed[0]*self.delta_t  #state[1] = spacing

        for t in range(1, self.prediction_horizon):
            next_speed[t] = next_speed[t-1] + action[0]*self.delta_t   #state[0] = speed of ego vehicle current state 
            next_rel_speed[t] = lv_speed[t] - next_speed[t] # assume LV with constant speed
            next_spacing[t] = next_spacing[t-1] + next_rel_speed[t]*self.delta_t  #state[1] = spacing
        
        if np.any(next_speed <= 0):
            next_speed = 0.00001 * np.ones(self.prediction_horizon)
            self.is_stall = 1
        else:
            self.is_stall = 0

        if np.any(next_spacing < self.min_spacing):
            self.is_collision = 1
            print('collision!')
            self.collision_count += 1

        # calculate reward
        ttc = np.zeros(self.prediction_horizon)
        for t in range(self.prediction_horizon):
            if next_rel_speed[t] < 0:
                ttc[t] = -next_spacing[t]/next_rel_speed[t]
            else:
                ttc[t] = 10000

        timehwy = next_spacing/next_speed
        jerk = (action[0] - self.last_action) / self.delta_t
        self.last_action = action[0]

        ttc_min = min(ttc)
        if ttc_min >= 0 and ttc_min <= self.ttc_threshold:
            r_ttc = np.log(ttc_min/self.ttc_threshold) 
        else:
            r_ttc = 0.0

        mu = 0.422618  
        sigma = 0.43659

        # mu = 0.5  
        # sigma = 1.0
        desired_headway = np.exp(mu - sigma**2)

        # timehwy_mean = np.mean(timehwy)
        timehwy_mean = 0.0
        for n in range(len(timehwy)):
            timehwy_mean = timehwy_mean + self.gamma**n * timehwy[n]

        if timehwy_mean <= 0:
            r_timehwy = -1.0
        else:
            r_timehwy = (np.exp(-(np.log(timehwy_mean) - mu) ** 2 / (2 * sigma ** 2)) / (timehwy_mean * sigma * np.sqrt(2 * np.pi))) 

        r_jerk = -(jerk ** 2)/3600.0 
        
        if timehwy[0] < desired_headway and next_speed[0] > lv_speed[0]:
            # r_ss = -(next_speed[0]-self.ss_gamma*lv_speed[0])**2
            r_ss = -np.max(next_speed[0]-self.ss_gamma*self.lv_speed[max(0, self.timestep-self.ss_window):self.timestep+1])**2
        else:
            r_ss = 0.0
        
        # calculate the reward
        reward = self.w_jerk*r_jerk + self.w_ttc*r_ttc + self.w_timehwy*r_timehwy + self.w_ss*r_ss - self.penalty * self.is_collision - self.penalty * self.is_stall 

        # check terminal state
        if self.timestep == self.lv_speed.shape[0] or self.is_collision == 1:
            done = True
        else:
            done = False
     
        self.state = np.array([next_speed[0], next_spacing[0], *next_rel_speed[0::self.prediction_rate]], dtype=np.float32) # next state

        return self.state, reward, done, False, {}

    def reset(self, data):
        self.timestep = 0
        self.lv_speed = data[:, 3] # spacing, ego vehicle speed, relative speed, lead vehicle speed
        self.state = np.array([data[0, 1], data[0, 0], *[data[0, 2]]*int(1+(self.prediction_horizon-1)/self.prediction_rate)], dtype=np.float32)  # spacing, cur_speed, rel_speed
        self.is_stall = 0   #depends on next_speed
        self.is_collision = 0   #depends on next_spacing
        self.last_action = 0.0
        self.collision_count = 0
        return np.array(self.state, dtype=np.float32), {}

        
    def render(self):
        raise NotImplementedError
    


class ICACCEnv():
    def __init__(self, prediction_method):
        self.min_action = -3.0  # minimum acceleration 
        self.max_action = 3.0   # maximum acceleration
        self.min_speed = 0.0
        self.max_speed = 32.0
        self.min_rel_speed = -8.0
        self.max_rel_speed = 8.0
        self.min_spacing = 5.0    #space between two vehicles
        self.max_spacing = 200.0
        self.prediction_horizon = 30 + 1
        self.prediction_method = prediction_method # 'constant', 'idm', or 'data'
        self.prediction_rate = 1 # how often we predict in terms of frames (1 means every frame)
        self.delta_t = 0.1
        self.gamma = 0.95


        self.min_spacing_error = -20.0
        self.max_spacing_error = 20.0
        self.min_accel = -3.0
        self.max_accel = 3.0
        self.min_accel_limit = 0.1
        self.max_accel_limit = 3.0
        self.desired_headway = np.exp(self.mu - self.sigma **2)
        self.mu = 0.422618
        self.sigma = 0.43659
        



       

        self.low_state = np.array([self.min_speed,
                                #    self.min_accel,
                                   self.min_spacing,
                                   self.min_spacing_error,
                                   self.min_rel_speed,
                                   self.min_accel,
                                   self.min_speed,
                                   self.min_speed,
                                   self.min_accel,
                                   self.min_accel,
                                   self.min_accel_limit], dtype=np.float32)
        self.high_state = np.array([self.max_speed,
                                    # self.max_accel,
                                    self.max_spacing,
                                    self.max_spacing_error,
                                    self.max_rel_speed,
                                    self.max_accel,
                                    self.max_speed,
                                    self.max_speed,
                                    self.max_accel,
                                    self.max_accel,
                                    self.max_accel_limit], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        self.is_stall = 0
        self.is_collision = 0
        self.ttc_threshold = 4.0
        self.penalty = -100.0   

        self.w_jerk = 3.0
        self.w_ttc = 2.0
        self.w_timehwy = 1.0
        self.w_ss = 4.0
        self.collision_count = 0

        
        self.ss_gamma = 0.99
        self.ss_window = 20

       

    def step(self, action: np.ndarray):

        accel = np.interp(action[0], [self.min_action, self.max_action], [self.min_accel, self.max_accel])

        if self.prediction_method == 'constant':
            # lv_speed = self.lv_speed[self.timestep] * np.ones(self.prediction_horizon)
            lv_speed = self.lv_speed[self.timestep]
            lv_accel = self.lv_accel[self.timestep]

        elif self.prediction_method == 'data':
            lv_speed = self.lv_speed[np.minimum(
            self.timestep + np.arange(0, self.prediction_horizon), self.lv_speed.shape[0]-1)] 
            lv_accel = self.lv_accel[np.minimum(
            self.timestep + np.arange(0, self.prediction_horizon), self.lv_accel.shape[0]-1)]
        
        elif self.prediction_method == 'idm':
            raise NotImplementedError
        
        self.timestep += 1
        # next_speed = np.zeros(self.prediction_horizon)
        # next_spacing = np.zeros(self.prediction_horizon)
        # next_rel_speed = np.zeros(self.prediction_horizon)
        #next_acc = np.zeros(self.prediction_horizon)


        next_speed = np.clip(self.state[0] + accel*self.delta_t, self.min_speed, self.max_speed)
        next_rel_speed = np.clip(lv_speed[0] - next_speed, self.min_rel_speed, self.max_rel_speed)
        next_spacing = self.state[1] + next_rel_speed*self.delta_t
        next_spacing_error = np.clip(next_spacing - self.desired_headway*next_speed, self.min_spacing_error, self.max_spacing_error)
        next_accel_limit = np.clip(self.ss_gamma*np.maximum(0.1, np.max(
            np.abs(lv_accel))), self.min_accel_limit, self.max_accel_limit)


        # next_speed[0] = self.state[0] + action[0]*self.delta_t   #state[0] = speed of ego vehicle current state 
        # next_rel_speed[0] = lv_speed[0] - next_speed[0]
        # next_spacing[0] = self.state[1] + next_rel_speed[0]*self.delta_t  #state[1] = spacing

        # for t in range(1, self.prediction_horizon):
        #     next_speed[t] = next_speed[t-1] + action[0]*self.delta_t   #state[0] = speed of ego vehicle current state 
        #     next_rel_speed[t] = lv_speed[t] - next_speed[t] # assume LV with constant speed
        #     next_spacing[t] = next_spacing[t-1] + next_rel_speed[t]*self.delta_t  #state[1] = spacing
        
        if next_speed <= self.min_speed:
            # next_speed = 0.00001 * np.ones(self.prediction_horizon)
            self.is_stall = 1
        else:
            self.is_stall = 0

        if next_spacing <= self.min_spacing:
            self.is_collision = 1
            print('collision!')
            

        # calculate reward
        jerk = (accel-self.last_accel)/ self.delta_t
        r_jerk = -(jerk ** 2)/3600.0
        self.last_accel = accel

        
        if next_rel_speed < 0.0:
                ttc = -next_spacing /next_rel_speed
        else:
                ttc = 1000.0

        timehwy = next_spacing/next_speed
        

        ttc_min = min(ttc)
        if ttc_min >= 0 and ttc_min <= self.ttc_threshold:
            r_ttc = np.log(ttc_min/self.ttc_threshold) 
        else:
            r_ttc = 0.0

        # mu = 0.422618  
        # sigma = 0.43659

        # mu = 0.5  
        # sigma = 1.0
        # desired_headway = np.exp(mu - sigma**2)

        timehwy_mean = np.mean(timehwy)
        timehwy_mean = 0.0
        for n in range(len(timehwy)):
            timehwy_mean = timehwy_mean + self.gamma**n * timehwy[n]

        if timehwy_mean <= 0:
            r_timehwy = -1.0
        else:
            r_timehwy = (np.exp(-(np.log(timehwy_mean) - self.mu) ** 2 / (2 * self.sigma ** 2)) / (timehwy_mean * self.sigma * np.sqrt(2 * np.pi))) 

         
        
        if timehwy[0] < self.desired_headway and next_speed[0] > lv_speed[0]:
            # r_ss = -(next_speed[0]-self.ss_gamma*lv_speed[0])**2
            r_ss = -np.max(next_speed[0]-self.ss_gamma*self.lv_speed[max(0, self.timestep-self.ss_window):self.timestep+1])**2
        else:
            r_ss = 0.0
        
        # calculate the reward
        reward = self.w_jerk*r_jerk + self.w_ttc*r_ttc + self.w_timehwy*r_timehwy + self.w_ss*r_ss - self.penalty * self.is_collision - self.penalty * self.is_stall 

        # check terminal state
        if self.timestep == self.lv_speed.shape[0] or self.is_collision == 1:
            done = True
        else:
            done = False
     
        self.state = np.array([next_speed[0], next_spacing[0], *next_rel_speed[0::self.prediction_rate]], dtype=np.float32) # next state

        return self.state, reward, done, False, {}

    def reset(self, data):
        self.timestep = 0
        self.lv_speed = data[:, 3] # spacing, ego vehicle speed, relative speed, lead vehicle speed
        self.state = np.array([data[0, 1], data[0, 0], *[data[0, 2]]*int(1+(self.prediction_horizon-1)/self.prediction_rate)], dtype=np.float32)  # spacing, cur_speed, rel_speed
        self.is_stall = 0   #depends on next_speed
        self.is_collision = 0   #depends on next_spacing
        self.last_action = 0.0
        self.collision_count = 0
        return np.array(self.state, dtype=np.float32), {}

        
    def render(self):
        raise NotImplementedError   

    
