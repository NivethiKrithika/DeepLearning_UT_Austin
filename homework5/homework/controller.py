import pystk
import math

import math

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    print("aim points is {}".format(aim_point))
    print("velocity is {}".format(current_vel))
    #x_points = [0]
    #y_points = [0]
    #orient = [0]
    action = pystk.Action()
    target_velocity  = 20
    #target acceleration = 20/25
    M_PI = 3.14
    #adj = aim_point[1] -y_points[-1]     
    #opp = aim_point[0] - x_points[-1]
    #steer_angle = math.atan(opp/adj)
    # radius = (opp**2 + adj**2)/(2*opp)
    #radius = ((aim_point[0] ** 2)+(aim_point[1] ** 2))/(2*aim_point[0])
    #sin_steer_angle  = 0.76/radius
    sin_steer_angle = -(aim_point[0]/aim_point[1])
    print("radius is {}".format(sin_steer_angle))
    steer_angle = (math.atan(sin_steer_angle/0.32) * 180)/M_PI;
    final_angle = steer_angle
    print("steering angle is {}".format(final_angle))
    """
    while( new_angle >  2*M_PI ):
        new_angle -= 2*M_PI
    while( new_angle < -2*M_PI ):
       new_angle += 2*M_PI

    if( new_angle > M_PI):
       new_angle -= 2*M_PI
    elif( new_angle < -M_PI):
       new_angle += 2*M_PI
    final_angle = new_angle 
    print("final steer angle is {}".format(final_angle))
    """
    if(final_angle > 60):
      action.drift = 1
    steer_angle_fraction = final_angle/90
    """
    if(steer_angle_fraction > 0.1):
       steer_angle_fraction = 0.1
    elif(steer_angle_fraction < -0.1):
      steer_angle_fraction = -0.1
    """
    print("steer_angle_fraction is {}".format(steer_angle_fraction))
    
    action.steer = steer_angle_fraction
    if(current_vel >= target_velocity):
      action.acceleration = 0
      
    else:
      acceler = 1
      action.acceleration = acceler
    print("acceleration is {}".format(action.acceleration)) 
    
    
    if(action.acceleration > 0):
      action.nitro = 1
    else:
      action.nitro = 0 
    if(steer_angle_fraction < 0.15):
      dt = 0.15
    else:
      dt = 0.25
    new_x = aim_point[0] + current_vel * math.cos(final_angle) * dt
    new_y = aim_point[1] + current_vel * math.sin(final_angle) * dt

    print("new x is {}".format(new_x))
    print("new y is {}".format(new_y))
    #x_points.append(new_x)
    #y_points.append(new_y)
     
    #if((steer_angle > 1) or (steer_angle < -1)):
     # action.drift = True
    action.brake = False
    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    return action



if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
