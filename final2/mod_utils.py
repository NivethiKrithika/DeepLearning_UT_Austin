import pystk
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import importlib

def to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

class Player:
    def __init__(self, player, team):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info, state):
        return self.player.act(image, player_info, state)


class Tournament:
    _singleton = None

    def __init__(self, players, game_no, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.game_no = game_no
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

        state.update()
            
            #-0.47587478160858154, 6.643647193908691, 20.54435932636261
            
        pos_ball = state.soccer.ball.location
        pos_ball[0] = -30
        pos_ball[1] = 0.9000000002980232
        pos_ball[2] =  42
        state.set_ball_location(position=pos_ball)

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
                classification = 1 if kart_puck_dp>0 else 0
                action = pystk.Action()
                player_action = p(image, player, state)
                #print("player is {}{}".format(player,player.kart.front))
                #print("ball location is {}".format(state.soccer.ball.location))
                #print("player location is {}".format(player.kart.location))
                
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

                puck_location = state.soccer.ball.location
                proj = np.array(player.camera.projection).T
                view = np.array(player.camera.view).T

                # print(player.kart.name)
                # print(player.kart.name == 'konqi')
                if (save is not None) and (player.kart.name == 'Konqi'):
                    PIL.Image.fromarray(image).save(os.path.join(save, f"g{self.game_no}_f{t}_p{i}_c{classification}.png"))
                    puck_x, puck_y = to_image(puck_location, proj, view)    
                    if classification == 1:               
                        with open(os.path.join(save,f"g{self.game_no}_f{t}_p{i}_c{classification}.csv"), 'w') as f:
                            f.write(f"{puck_x}, {puck_y}")
                        f.close()


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

