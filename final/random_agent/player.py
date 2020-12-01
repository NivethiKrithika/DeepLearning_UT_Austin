import numpy as np
import pystk
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
def to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)
class HockeyPlayer:
    def __init__(self, player_id = 0):
        self.kart = 'konqi'
        self.puck_ahead_thresh = 20
        self.goal_ahead_thresh = 80
        self.puck_close_thresh = 0.2
        self.kart_stuck_thresh = 0.05
        self.kart_stuck_counter = 0
        self.puck_super_close_thresh = 0.01
        self.reverse_steering_coeff = 4.5
        self.history = defaultdict(list)
        self.framestep = 0
        self.goal_close_thresh = 0.2
    def puck_is_close(self): 
        # Check if puck is close based on heuristic
        return np.sqrt(self.puck_x**2 + self.puck_y**2) < self.puck_close_thresh
    def goal_is_close(self): 
        # Check if puck is close based on heuristic
        return np.sqrt(self.goal_x**2 + self.goal_y**2) < self.goal_close_thresh
    def puck_is_ahead(self): 
        # Check if puck is ahead within heuristic range
        return np.arccos(self.kart_puck_dp)*180/np.pi < self.puck_ahead_thresh
    def goal_is_ahead(self): 
        # Check if puck is ahead within heuristic range
        return np.arccos(self.kart_goal_dp)*180/np.pi < self.goal_ahead_thresh
    def kart_is_stuck(self):
        return np.linalg.norm(np.array(self.history['kart_location'][-1]) - \
            np.array(self.history['kart_location'][-5 if self.framestep >=5 else -self.framestep])) < self.kart_stuck_thresh
    def reverse(self): 
        # Reset the threshold the the base heuristic
        self.puck_ahead_thresh = 20
        # Angle to steering towards puck, corrected with 4.5 as a heuristic I used in HW#5
        puck_steer = self.reverse_steering_coeff*self.puck_x
        # Compute steering direction and magnitude 
        steer_direction = -np.sign(self.kart_puck_vec_norm[0])
        steer_magnitude = abs(puck_steer) if abs(puck_steer) >= 0.5 else 0.5 # heuristic to make sure we do not reverse without actually steering in some direction
        action = {'acceleration': 0, 'brake': True, 'drift': False, 'nitro': False, 'rescue': False, 'steer': steer_direction*steer_magnitude}
        return action 
    def circle_drive(self, x, y): 
        # Reset the threshold the the base heuristic
        self.puck_ahead_thresh = 45 
        r2 = x**2 + y**2
        a = x/2
        b = y/2
        y = b/2 # this is a heuristic of how far on the circle we wanna move per frame
        x = (a + np.sign(x) * np.sqrt(r2 - (y - b)**2))*(4 if self.goal_is_ahead() else 2) # these are heuristic coefficients to allow for sufficient turn
        acceleration = 0.75 if self.goal_is_ahead() else 1.0
        action = {'acceleration': acceleration, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': x}
        return action
    def release_stuck(self):
        self.kart_stuck_counter += 1
        if self.kart_stuck_counter < 5:
            return {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        elif self.kart_stuck_counter < 10:
            return {'acceleration': 0, 'brake': True, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        else:
            self.kart_stuck_counter = 0
            return {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': True, 'steer': 0}
    def steer_towards_goal(self,x,y,puck_x,puck_y,wheelbase,goal_end1x,goal_end1y, goal_end2x,goal_end2y):
        print("kart goal distance is {}".format(self.kart_goal_dp))
        print("x and y of goal are {} {}".format(x,y))
        print("puck x and y are {} {}".format(puck_x,puck_y))
        print("kart puck vector norm is {}".format(self.kart_puck_vec_norm))
        print("kart goal vec norm is  {}".format(self.kart_goal_vec_norm))
        print("goal end points 1 are {}{}".format(goal_end1x,goal_end1y))
        print("goal end points 2 are {}{}".format(goal_end2x,goal_end2y))
        steer_direction = np.sign(self.kart_goal_vec_norm[0])
        tan_steer_angle = self.kart_goal_vec_norm[1]/self.kart_goal_vec_norm[0]
        steer_angle = (math.atan(tan_steer_angle) * 180)/np.pi;
        print("steer angle is {}".format(steer_angle))
        #final_angle = steer_angle
        #steer_angle_fraction = final_angle/90
        #b = y/2
        #y = b/2 # this is a heuristic of how far on the circle we wanna move per frame
        #x = (a + np.sign(x) * np.sqrt(r2 - (y - b)**2))*(4 if self.puck_is_close() else 2) # these are heuristic coefficients to allow for sufficient turn
        #acceleration = 0.75 if self.goal_is_close() else 1.0
        acceleration = 0.50
        action = {'acceleration': acceleration, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': (abs(steer_angle)/90)* steer_direction}
        return action
        
    def act(self, image, player_info, state):
        # Perform 3D vecotr manipulation
        self.kart_front_vec = np.array(player_info.kart.front) - np.array(player_info.kart.location)
        self.kart_front_vec_norm = self.kart_front_vec / np.linalg.norm(self.kart_front_vec)
        #print("puck location is {}".format(state.soccer.ball.location))
        #print("kart location is {}".format(player_info.kart.location))
        self.kart_puck_vec = np.array(state.soccer.ball.location) - np.array(player_info.kart.location)
        print("kart puck vector is {}".format(self.kart_puck_vec))
        self.kart_puck_vec_norm = self.kart_puck_vec / np.linalg.norm(self.kart_puck_vec)
        print("kart puck vec norm is {}".format(self.kart_puck_vec_norm))
        self.kart_puck_dp = self.kart_front_vec_norm.dot(self.kart_puck_vec_norm)
        print("kart puck dp is {}".format(self.kart_puck_dp))
        self.goal_line = [(i+j)/2 for i, j in zip(state.soccer.goal_line[1][0], state.soccer.goal_line[1][1])]
        self.kart_goal_vec = np.array(self.goal_line) - np.array(player_info.kart.location)
        self.kart_goal_vec_norm = self.kart_goal_vec / np.linalg.norm(self.kart_goal_vec)
        self.kart_goal_dp = self.kart_front_vec_norm.dot(self.kart_goal_vec_norm)
        self.history['kart_location'].append(player_info.kart.location)
        self.framestep += 1
        #print("wheelbase is  {}".format(player_info.kart.wheel_base))
        # self.puck_goal_vec = 
        # Project 3D vectors to the kart camera view 2D plane
        self.proj = np.array(player_info.camera.projection).T
        self.view = np.array(player_info.camera.view).T
        self.puck_x, self.puck_y = to_image(state.soccer.ball.location, self.proj, self.view)
        self.kart_x, self.kart_y = to_image(player_info.kart.location, self.proj, self.view)
        self.goal_x, self.goal_y = (to_image(state.soccer.goal_line[1][0], self.proj, self.view) + 
                         to_image(state.soccer.goal_line[1][1], self.proj, self.view)) / 2
        self.goal_end1_vec = np.array(state.soccer.goal_line[1][0]) - np.array(player_info.kart.location)
        self.goal_end2_vec = np.array(state.soccer.goal_line[1][1]) - np.array(player_info.kart.location)
        self.goal_end1_x,self.goal_end1_y = to_image(state.soccer.goal_line[1][0],self.proj,self.view)
        self.goal_end2_x,self.goal_end2_y = to_image(state.soccer.goal_line[1][1],self.proj,self.view)
        self.current_vel = np.linalg.norm(player_info.kart.velocity)
        print("goal end 1 vector is {}".format(self.goal_end1_vec))
        print("goal end 2 vector is {}".format(self.goal_end2_vec))
        # puck_is_super_close =  np.sqrt(self.puck_x**2 + self.puck_y**2) < self.puck_super_close_thresh
        #if self.kart_is_stuck():
         #   return self.release_stuck()
        #self.kart_stuck_counter = 0
        #if self.puck_is_ahead():
            # print("Hit here")
             #print("goal line is {},{}".format(self.goal_x, self.goal_y))
             #if self.puck_is_close() and self.goal_is_ahead():
                # print("Hitting close here")
        #if self.goal_is_ahead():
        return self.steer_towards_goal(self.goal_x, self.goal_y,self.puck_x,self.puck_y,player_info.kart.wheel_base,self.goal_end1_x,self.goal_end1_y,self.goal_end2_x,self.goal_end2_y)
            # else:
             #    return self.circle_drive(self.puck_x, self.puck_y)
        #else:
         #     print("Hit hereeeeeeee")
          #    return self.reverse()
        #return {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        # return action
