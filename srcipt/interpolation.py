import argparse
from scipy.spatial.transform import Rotation as RR
import numpy as np
import math

def main(input_path_traj, input_path_timestamps):
    lidar_path_name = input_path_traj.split('/')
    file_name = lidar_path_name[-1].split('.')
    time1 = time2 = time_inter = 0.0
    time_stamps = []
    q1 = q2 = RR.from_quat([0.0, 0.0, 0.0, 1.0])
    trans1 = trans2 = np.array([0.0, 0.0, 0.0])
    is_initial = True
    with open(input_path_timestamps, 'r') as f_cam:
        for line in f_cam.readlines():
            time_stamps.append(float(line))
    
    time_stamps = iter(time_stamps)

    with open(input_path_traj, 'r') as f, open(file_name[0] + '_interpolation.txt', 'w+') as new_file:
        for line in f.readlines():
            tmp_txt = str.split(line.strip())
            
            if is_initial:
                while time_inter < float(tmp_txt[0]):
                    time_inter = next(time_stamps)
                time1 = float(tmp_txt[0])
                q1 = RR.from_quat([float(tmp_txt[4]), float(tmp_txt[5]), float(tmp_txt[6]), float(tmp_txt[7])])
                trans1 = np.array([float(tmp_txt[1]), float(tmp_txt[2]), float(tmp_txt[3])])
                is_initial = False
                
            time2 = float(tmp_txt[0])
            q2 = RR.from_quat([float(tmp_txt[4]), float(tmp_txt[5]), float(tmp_txt[6]), float(tmp_txt[7])])
            trans2 = np.array([float(tmp_txt[1]), float(tmp_txt[2]), float(tmp_txt[3])])
            
            while (time1 < time_inter) & (time2 > time_inter):
                print("time1 = {0}, time2 = {1}, time_inter = {2}".format(time1, time2, time_inter))
                q_inter = interpolate_rotation(time1, q1, time2, q2, time_inter)
                trans_inter = interpolate_translation(time1, trans1, time2, trans2, time_inter) 
                write_txt = "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(time_inter, 
                    trans_inter[0], trans_inter[1], trans_inter[2], 
                    q_inter.as_quat()[0], q_inter.as_quat()[1], q_inter.as_quat()[2], q_inter.as_quat()[3])
                new_file.writelines(write_txt)
                
                time_inter = next(time_stamps, None)
                if time_inter is None:
                    return
            
            time1 = time2
            q1 = q2
            trans1 = trans2

def interpolate_translation(t1, trans1, t2, trans2, t_inter):
    return trans1 + (trans2 - trans1) / (t2 - t1) * (t_inter - t1)

def interpolate_rotation(t1, q1, t2, q2, t_inter):
    theta = (t_inter - t1) / (t2 - t1)
    q1_2 = RR.from_quat([-q1.as_quat()[0], -q1.as_quat()[1], -q1.as_quat()[2], q1.as_quat()[3]]) * q2
    q_inter = q1 * exp_quat(theta * log_quat(q1_2))
    return q_inter
    # return q1

# Method of implementing this function that is accurate to numerical precision from
# Grassia, F. S. (1998). Practical parameterization of rotations using the exponential map. journal of graphics, gpu, and game tools, 3(3):29–48.
def exp_quat(dx):
    theta = np.linalg.norm(dx)
    # na is 1/theta sin(theta/2)
    na = 0
    if is_less_then_epsilon_4th_root(theta):
        one_over_48 = 1.0 / 48.0
        na = 0.5 + (theta * theta) * one_over_48
    else:
        na = math.sin(theta * 0.5) / theta
    ct = math.cos(theta * 0.5)
    return RR.from_quat([dx[0]*na, dx[1]*na, dx[2]*na, ct])

def log_quat(q):
    q_imagi = q.as_quat()[:3]
    na = np.linalg.norm(q_imagi)
    eta = q.as_quat()[3]
    scale = 0.0
    # use eta because it is more precise than na to calculate the scale. No singularities here.
    if abs(eta) < na: 
    	# check sign of eta so that we can be sure that log(-q) = log(q)
        if eta >= 0:
            scale = math.acos(eta) / na
        else:
            scale = -math.acos(-eta) / na
    else:
     ###
     # In this case more precision is in na than in eta so lets use na only to calculate the scale:
     #
     # assume first eta > 0 and 1 > na > 0.
     #               u = asin (na) / na  (this implies u in [1, pi/2], because na i in [0, 1]
     #    sin (u * na) = na
     #  sin^2 (u * na) = na^2
     #  cos^2 (u * na) = 1 - na^2
     #                              (1 = ||q|| = eta^2 + na^2)
     #    cos^2 (u * na) = eta^2
     #                              (eta > 0,  u * na = asin(na) in [0, pi/2] => cos(u * na) >= 0 )
     #      cos (u * na) = eta
     #                              (u * na in [ 0, pi/2] )
     #                 u = acos (eta) / na
     #
     # So the for eta > 0 it is acos(eta) / na == asin(na) / na.
     # From some geometric considerations (mirror the setting at the hyper plane q==0) it follows for eta < 0 that (pi - asin(na)) / na = acos(eta) / na.
     ###
        if eta > 0:
            scale = arc_sin_x_over_x(na)
        else:
            scale = -arc_sin_x_over_x(na)
    return q_imagi * (2.0 * scale)

def is_less_then_epsilon_4th_root(x):
    return x < pow(np.finfo(np.float64).eps, 1.0/4.0)
    
def arc_sin_x_over_x(x):
    if is_less_then_epsilon_4th_root(abs(x)):
        return 1.0 + x * x * (1.0/6.0)
    return math.asin(x) / x
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_traj", required=True)
    parser.add_argument("--input_path_timestamps", required=True)
    args = parser.parse_args()
    main(args.input_path_traj, args.input_path_timestamps)
