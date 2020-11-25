import pystk
import numpy as np
import math

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
                print("aim point is {}".format(aim_point))
                #control(aim_point,np.linalg.norm(player.kart.velocity))
                action = pystk.Action()
                #player_action = p(image, player)
                player_action = control(aim_point,np.linalg.norm(player.kart.velocity))
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

                if save is not None:
                    PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))

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



def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """

    action = pystk.Action()
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

  
    action.brake = False
    

    action1 = {'acceleration': action.acceleration, 'brake': action.brake, 'drift': action.drift, 'nitro': action.nitro, 'rescue': False, 'steer': action.steer}

    return action1
