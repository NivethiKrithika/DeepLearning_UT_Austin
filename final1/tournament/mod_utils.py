import pystk
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import importlib

def to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    # return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)
    return np.array([p[0] / p[-1], -p[1] / p[-1]])

class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)


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
                # Perform 3D vecotr manipulation
                kart_front_vec = np.array(player.kart.front) - np.array(player.kart.location)
                kart_front_vec_norm = kart_front_vec / np.linalg.norm(kart_front_vec)
                kart_puck_vec = np.array(state.soccer.ball.location) - np.array(player.kart.location)
                kart_puck_vec_norm = kart_puck_vec / np.linalg.norm(kart_puck_vec)
                kart_puck_dp = kart_front_vec_norm.dot(kart_puck_vec_norm)
                print(kart_puck_dp)
                action = pystk.Action()
                player_action = p(image, player)
                print("player is {}{}".format(player,player.kart.front))
                
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

                puck_location = state.soccer.ball.location
                proj = np.array(player.camera.projection).T
                view = np.array(player.camera.view).T

                if save is not None:
                  if (i%2 == 0):
                    PIL.Image.fromarray(image).save(os.path.join(save, save+'_player%02d_%05d.png' % (i, t)))
                    puck_location_image = to_image(puck_location, proj, view)
                    player_state = [player.kart.location,player.kart.front,state.soccer.ball.location]  
                    print("player state is {}".format(np.array(player_state)))                  
                    with open(os.path.join(save,save+'_player%02d_%05d.csv' % (i, t)), 'w') as f:
                        if (kart_puck_dp>0.0) and (-1.0 <= round(puck_location_image[0],1) <= 1.0) and (-1.0 <= round(puck_location_image[1],1) <= 1.0) :
                            f.write('%0d,%0.1f,%0.1f' % tuple((1,round(puck_location_image[0],1),round(puck_location_image[1],1))))
                            np.save(os.path.join(save, save+'_player%02d_%05d.npy' % (i, t)), np.array(player_state))
                        else:
                            f.write('%0d,%0.1f,%0.1f' % tuple((0,2,2)))
                            np.save(os.path.join(save, save+'_player%02d_%05d.npy' % (i, t)), np.array(player_state))

                        # f.write('%0.2f,%0.1f,%0.1f' % tuple((kart_puck_dp,puck_location_image[0],puck_location_image[1])))
                    

            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                
                dest = os.path.join(save, save+'_player%02d' % i)
                output = save + '_player%02d.mp4' % i
                subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])
        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k


