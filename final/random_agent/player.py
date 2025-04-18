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
        self.player_id = player_id%2
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
        x = (a + np.sign(x) * np.sqrt(r2 - (y - b)**2))*(4 if self.puck_is_close() else 2) # these are heuristic coefficients to allow for sufficient turn
        acceleration = 0.75 if self.puck_is_close() else 1.0
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
    def steer_towards_goal(self,puck_x,puck_y):
      print("kart goal distance is {}".format(self.kart_goal_dp))
      #print("x and y of goal are {} {}".format(x,y))
      print("puck x and y are {} {}".format(self.puck_x,self.puck_y))
      print("kart puck vector norm is {}".format(self.kart_puck_vec_norm))
      print("kart goal vec norm is  {}".format(self.kart_goal_vec_norm))
      #print("goal end points 1 are {}{}".format(goal_end1x,goal_end1y))
      #print("goal end points 2 are {}{}".format(goal_end2x,goal_end2y))
      print("abs value is {}".format(abs(self.kart_puck_vec[0])))
      print("the radius is {}".format(((self.puck_x) ** 2 )+ ((self.puck_y)**2)))
      radius = ((self.puck_x) ** 2 )+ ((self.puck_y)**2)
      print("kart goal dp is {}".format("kart_goal_dp"))
      
        #if(abs(puck_x) > 0.015):
          #print("Hitting here in threshold")
          #threshold_to_reverse = 10
        #print("steer angle is {}".format(steer_angle))
      if((self.ball_location[0] > -10.5 ) and (self.ball_location[0] < 10.5)):
        if(self.kart_goal_dp > 0):
          threshold_to_reverse = 6
          if(abs(self.kart_puck_vec[0]) > 8):
            threshold_to_reverse = abs(self.kart_puck_vec[0])
          print("thr to rev is {}".format(threshold_to_reverse))
          if(self.kart_puck_vec[2] > threshold_to_reverse):           
            if(abs(self.puck_x) > 0.40):
              return self.reverse()
            if((radius < 0.040) and (abs(self.puck_x) > 0.15)):
              print("Hitting reverse")
              #return self.circle_drive(puck_x+)
              #return self.reverse()
              action = {'acceleration': 0, 'brake': True, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
              return action
              #action2['steer'] = action2['steer'] * 1.25            
            else:
              action1 =  self.circle_drive(puck_x,puck_y)
              print("The action is  {}".format(action1))
              action1['acceleration'] = 0.40
              return action1
          else:
            print("Hitting in else")
            if((abs(self.puck_x) > 0.15)):
              action = {'acceleration': 0, 'brake': True, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
              return action
            else:
              action = self.circle_drive(puck_x,puck_y)
              action['acceleration'] = 0.40
              return action
        else:
          action = self.circle_drive(puck_x,puck_y)
          action['acceleration'] = 0.40
          return action   
      else:
        action = self.circle_drive(puck_x,puck_y)
        action['acceleration'] = 0.40
        return action 
      

            #steer_direction = np.sign(self.kart_goal_vec_norm[0])
            #tan_steer_angle = x/y
            #steer_angle = (math.atan(tan_steer_angle/(wheelbase/2)) * 180)/np.pi;
            #print("steer angle is {}".format(steer_angle))
            #tan_steer_angle = (puck_x/puck_y)
            #steer_direction = np.sign(self.kart_puck_vec_norm[0])
            #tan_steer_angle = self.kart_puck_vec_norm[1]/self.kart_puck_vec_norm[0]
            #steer_angle = (math.atan(tan_steer_angle/(wheelbase/2)) * 180)/np.pi
            #final_angle = steer_angle
    #if(final_angle > 60):

     # action.drift = 1
            #steer_angle_fraction = final_angle/90
    #action.steer = steer_angle_fraction
            #steer_direction = np.sign(self.kart_puck_vec_norm[0])
            #tan_steer_angle = puck_x/puck_y
            #steer_angle = (math.atan(tan_steer_angle/(wheelbase/2)) * 180)/np.pi;
             
            #return self.circle_drive(self.puck_x,self.puck_y)
        #return self.circle_drive(puck_x,puck_y)
        #steer_direction = np.sign(self.kart_puck_vec_norm[0])
        #tan_steer_angle = puck_x/puck_y
        #steer_angle = (math.atan(tan_steer_angle/(wheelbase/2)) * 180)/np.pi;
        """
        if((self.goal_puck_vec_norm[1]) < 0.2):
          tan_steer_angle = puck_x/puck_y
          steer_angle = (math.atan(tan_steer_angle) * 180)/np.pi;
          acceleration = 0.50
          action = {'acceleration': acceleration, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': (abs(steer_angle)/90)* steer_direction}
          return action

        if(self.kart_puck_vec_norm[1] < 0.20 ):
          if(np.sign(kart_puck_vec_norm[0]) == -1):
            if(abs(self.goal_end2_vec[0]) >= abs(kart_puck_vec_norm[0])):
                tan_steer_angle = puck_x/puck_y
                #tan_steer_angle = self.kart_goal_vec_norm[1]/self.kart_goal_vec_norm[0]
                steer_angle = (math.atan(tan_steer_angle) * 180)/np.pi;
            else:
              return self.reverse()
          else:
            if(abs(self.goal_end1_vec[0]/2) >= abs(kart_puck_vec_norm[0])):
                tan_steer_angle = puck_x/puck_y
                #tan_steer_angle = self.kart_goal_vec_norm[1]/self.kart_goal_vec_norm[0]
                steer_angle = (math.atan(tan_steer_angle) * 180)/np.pi;
            else:
              return self.reverse()
          
        else:
          return self.reverse()
        """
        #tan_steer_angle = x/y
        #tan_steer_angle = self.kart_goal_vec_norm[1]/self.kart_goal_vec_norm[0]
        #steer_angle = (math.atan(tan_steer_angle) * 180)/np.pi;
        #print("steer angle is {}".format(steer_angle))
        #final_angle = steer_angle
        #steer_angle_fraction = final_angle/90
        #b = y/2
        #y = b/2 # this is a heuristic of how far on the circle we wanna move per frame
        #x = (a + np.sign(x) * np.sqrt(r2 - (y - b)**2))*(4 if self.puck_is_close() else 2) # these are heuristic coefficients to allow for sufficient turn
        #acceleration = 0.75 if self.goal_is_close() else 1.0
        acceleration = 0.20
        action = {'acceleration': acceleration, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': (abs(steer_angle)/90)* steer_direction}
        #action = {'acceleration': acceleration, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': (abs(steer_angle)/90)* steer_direction}
        return action
        
    def act(self, image, player_info, state):
        # Perform 3D vecotr manipulation
        self.kart_front_vec = np.array(player_info.kart.front) - np.array(player_info.kart.location)
        self.kart_front_vec_norm = self.kart_front_vec / np.linalg.norm(self.kart_front_vec)
        #print("puck location is {}".format(state.soccer.ball.location))
        #print("kart location is {}".format(player_info.kart.location))
        self.kart_puck_vec = np.array(state.soccer.ball.location) - np.array(player_info.kart.location)
        print("player id is  {}".format(self.player_id))
        print("kart puck vector is {}".format(self.kart_puck_vec))
        self.kart_puck_vec_norm = self.kart_puck_vec / np.linalg.norm(self.kart_puck_vec)
        print("kart puck vec norm is {}".format(self.kart_puck_vec_norm))
        self.kart_puck_dp = self.kart_front_vec_norm.dot(self.kart_puck_vec_norm)
        print("kart puck dp is {}".format(self.kart_puck_dp))
        print("goal line 1 is {}".format(state.soccer.goal_line[self.player_id ^ 1][0]))
        print("goal line 2 is {}".format(state.soccer.goal_line[self.player_id ^ 1][1]))
        self.goal_line = [(i+j)/2 for i, j in zip(state.soccer.goal_line[self.player_id ^1][0], state.soccer.goal_line[self.player_id ^ 1][1])]
        self.ball_location = state.soccer.ball.location
        self.goal_puck_vec = np.array(self.goal_line) - np.array(player_info.kart.location)
        self.goal_puck_vec_norm = self.goal_puck_vec / np.linalg.norm(self.goal_puck_vec)
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
        #self.goal_x, self.goal_y = (to_image(state.soccer.goal_line[1][0], self.proj, self.view) + 
                       #  to_image(state.soccer.goal_line[1][1], self.proj, self.view)) / 2
        #self.goal_end1_vec = np.array(state.soccer.goal_line[self.player_id][0] & 1) - np.array(player_info.kart.location)
        #self.goal_end2_vec = np.array(state.soccer.goal_line[self.player_id][1] & 1) - np.array(player_info.kart.location)
        #self.goal_end1_x,self.goal_end1_y = to_image(state.soccer.goal_line[1][0],self.proj,self.view)
        #self.goal_end2_x,self.goal_end2_y = to_image(state.soccer.goal_line[1][1],self.proj,self.view)
        self.current_vel = np.linalg.norm(player_info.kart.velocity)
        #print("goal end 1 vector is {}".format(self.goal_end1_vec))
        #print("goal end 2 vector is {}".format(self.goal_end2_vec))
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
        return self.steer_towards_goal(self.puck_x,self.puck_y)
            # else:
             #    return self.circle_drive(self.puck_x, self.puck_y)
        #else:
         #     print("Hit hereeeeeeee")
          #    return self.reverse()
        #return {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        # return action
