import pystk
import numpy as np
import math
import matplotlib.pyplot as plt

class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)

def to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

class Tournament:
    _singleton = None

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
        self.race_config.players.pop()
        
        self.active_players = []
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)
        
        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()

    def play(self, save=None, max_frames=50):
        state = pystk.WorldState()
        if save is not None:
            import PIL.Image
            import os
            if not os.path.exists(save):
                os.makedirs(save)

        for t in range(max_frames):
            print('\rframe %d' % t, end='\r')

            state.update()
            

            list_actions = []
            for i, p in enumerate(self.active_players):
                player = state.players[i]
                image = np.array(self.k.render_data[i].image)
                puck_location = state.soccer.ball.location
                
                proj = np.array(player.camera.projection).T
                view = np.array(player.camera.view).T
                aim_point = to_image(puck_location,proj,view)
                front = to_image(player.kart.front,proj,view)
                #print("aim point is {}".format(aim_point))
                kart_location = to_image(player.kart.location,proj,view)

                #kart_location_2 = state.players[i].kart.location

                goal_line_1 = state.soccer.goal_line[1-i][0]

                goal_line_2 = state.soccer.goal_line[1-i][1]
                goal_line = (to_image(goal_line_1, proj, view)+to_image(goal_line_2, proj, view))/2
                #control(aim_point,np.linalg.norm(player.kart.velocity))
                action = pystk.Action()
                #player_action = p(image, player)
                player_action = control(aim_point,np.linalg.norm(player.kart.velocity),kart_location,goal_line,front)
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)
                fig, ax = plt.subplots(1, 1)

                ax.imshow(image)


                if save is not None:
                    PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))
                    WH2 = np.array([self.graphics_config.screen_width, self.graphics_config.screen_height]) / 2
                    #ax.add_artist(plt.Circle(WH2*(1+to_image(kart_location_1, proj, view)), 10, ec='r', fill=False, lw=1.5))

                    #ax.add_artist(plt.Circle(WH2*(1+to_image(kart_location_2, proj, view)), 10, ec='k', fill=False, lw=1.5))

                    #ax.add_artist(plt.Circle(WH2*(1+to_image(player.kart.front, proj, view)), 10, ec='y', fill=False, lw=1.5))

                    ax.add_artist(plt.Circle(WH2*(2+to_image(goal_line_1, proj, view)+to_image(goal_line_2, proj, view))/2, 10, ec='m', fill=False, lw=1.5))
                    ax.add_artist(plt.Circle(WH2*(2+to_image(goal_line_1, proj, view)+to_image(goal_line_1, proj, view))/2, 10, ec='b', fill=False, lw=1.5))                    


            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                dest = os.path.join(save, 'player%02d' % i)
                output = save + '_player%02d.mp4' % i
                subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])
        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k



def control(aim_point, current_vel,kart_loc,goal_line,front):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """

    print("aim point is {}".format(aim_point))
    #print("kart location is {}".format(kart_loc))
    #print("goal line is {}".format(goal_line))
    print("front is {}".format(front))
    action = pystk.Action()
    if(((aim_point[0] -front[0]) < 0.001) and ((aim_point[1]-front[1])<0.001)):
      aim_point = goal_line
    target_velocity  = 20
    #target acceleration = 20/25
    M_PI = 3.14
    

    sin_steer_angle = -(aim_point[0]/aim_point[1])
    #print("radius is {}".format(sin_steer_angle))
    steer_angle = (math.atan(sin_steer_angle/0.32) * 180)/M_PI;
    final_angle = steer_angle

    if(final_angle > 60):
      action.drift = 1
    steer_angle_fraction = final_angle/90

    
    action.steer = steer_angle_fraction
    #print("current vel is {}".format(current_vel))
    #print("target vel is {}".format(target_velocity))
    if(current_vel >= target_velocity):
      action.acceleration = 0
      
    else:
      acceler = 1
      action.acceleration = acceler
    #print("acceleration is {}".format(action.acceleration)) 
    
    
    if(action.acceleration > 0):
      action.nitro = 1
    else:
      action.nitro = 0
    
    action.drift = 0  
    action.brake = False
    if((aim_point[1] > 0) or aim_point[1] == 0):
       action.brake = True
       action.acceleration = 0
       action.nitro= 0
       action.drift = 0
       print("Hitting")

    #print(action.acceleration)
    #print(action.brake)
    action1 = {'acceleration': action.acceleration, 'brake': action.brake, 'drift': action.drift, 'nitro': action.nitro, 'rescue': False, 'steer': action.steer}

    return action1
