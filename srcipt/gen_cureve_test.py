
from itertools import count
from typing import Dict, List

PathPointUID = int


class GeneralPathPoint:
    _uid_to_objs = {}  # type: Dict[PathPointUID, GeneralPathPoint]
    _uid_counter = count()

    def __init__(self, pos, yaw, trans_type="STRAIGHT"):
        # type: (List[float], float, str) -> ...
        self.uid = next(self._uid_counter)  # type: PathPointUID
        """unique id for path point"""
        self.pos = pos
        self.yaw = yaw
        self.trans_type = trans_type
        """2D coordinate"""
        self._uid_to_objs[self.uid] = self

    @classmethod
    def get(cls, uid):
        return cls._uid_to_objs[uid]

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def __str__(self):
        return f'({self.uid}: position:{self.pos[0]:.1f},{self.pos[1]:.1f} yaw:{self.yaw}) trans_type:{self.trans_type}'

import math
from typing import List, Iterable
from sympy import Point, Line, pi


def calc_orientation(src: List[int], dst: List[int]) -> float:
    diff_x = dst[0] - src[0]
    diff_y = dst[1] - src[1]
    return math.atan2(diff_y, diff_x)


def eq(a: Iterable, b: Iterable, tol=1e-2):
    return all(abs(x - y) < tol for x, y in zip(a, b))

def wrap_to_pi(angle):
    while angle > math.pi :
        angle = angle - 2 * math.pi
    while angle < -math.pi :
        angle = angle + 2 * math.pi
    return float(angle)

def to_sympy_point( pt:GeneralPathPoint):
    return Point(pt.pos[0], pt.pos[1])

def line_angle( line:Line):
    return math.atan2(line.direction[1],line.direction[0])

def straight_line_to_curve( input_path: List[GeneralPathPoint], nav_type):

    interval = 0.1
    turning_radium = 0.7

    if len(input_path) < 3:
        return input_path

    CLW = 0 # clockwise
    CCLW = 1 # counter-clockwise
    new_path_points=[]
    sym_path_points = []

    for pt in input_path:
        sym_point=to_sympy_point(pt)
        sym_path_points.append(sym_point)

    
    new_path_points.append(input_path[0])

    for i in range(1, len(sym_path_points)-1):
        
        last_point=to_sympy_point(new_path_points[-1])
        
        l1 = Line(last_point, sym_path_points[i])
        l2 = Line(sym_path_points[i], sym_path_points[i+1])
    
        angle_diff = wrap_to_pi(line_angle(l2) - line_angle(l1))
        rotate_direction = CLW if angle_diff < 0  else  CCLW 

        # consider as a stright line
        # if abs(angle_diff) < 0.3 or abs(angle_diff) > pi/2 + 1.2 : 
        #    new_path_points.append(input_path[i])
        #    continue
        
        # if l1.length < turning_radium or l2.length < turning_radium : 
        #    new_path_points.append(input_path[i])
        #    continue
        
        mid_angle = line_angle(l2) + (pi - angle_diff)/2
        mid_angle = float(wrap_to_pi(mid_angle - pi)) if rotate_direction == CLW else float(wrap_to_pi(mid_angle))
        origin_x = sym_path_points[i][0] + abs(turning_radium/math.cos(angle_diff/2)) * math.cos(mid_angle)
        origin_y = sym_path_points[i][1] + abs(turning_radium/math.cos(angle_diff/2)) * math.sin(mid_angle)
        
        p0 = Point(origin_x, origin_y)
        p1 = l1.projection(p0)
        p2 = l2.projection(p0)

        start_angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        end_angle = math.atan2(p2[1] - p0[1], p2[0] - p0[0])

        # print("line 1 angle:", line_angle(l1), " line2 angle:" , line_angle(l2),"angle diff:", angle_diff)
        # print(" origin X:", float(sym_path_points[i][0]), " Y:" , float(sym_path_points[i][1]))
        # print("start angle:", start_angle, "end angle:" , end_angle, "mid angle:", mid_angle)
        
        for j in range(0, int(abs(angle_diff/interval))):
            j = -j if rotate_direction == CLW else j  
            new_point_x = p0[0] + turning_radium * math.cos(start_angle + j * interval)
            new_point_y = p0[1] + turning_radium * math.sin(start_angle + j * interval)
            new_point_yaw = wrap_to_pi(line_angle(l1) + j * interval) 
            new_path_points.append(GeneralPathPoint((float(new_point_x), float(new_point_y)), float(new_point_yaw),'CURVE'))
            
    new_path_points.append(input_path[-1]) # remember to add last point back
    if nav_type =="BACKWARD":
        new_path_points.reverse()
    # print(new_path_points[0])
    # print(new_path_points[1])
    # print(new_path_points[-2])
    # print(new_path_points[-1])

    return new_path_points        

from typing import List
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import math

def plot_path(path_points: List[GeneralPathPoint]):
    fig, ax = plt.subplots()
    verts = []
    codes = []
    verts.append((path_points[0].pos[0], path_points[0].pos[1]))
    codes.append(Path.MOVETO) 
    
    for i in range(0, len(path_points)):   
        current_point = path_points[i]
        # Add this point
        verts.append((current_point.pos[0], current_point.pos[1]))
        codes.append(Path.MOVETO)   
        ax.text(current_point.pos[0], current_point.pos[1], str(current_point.uid))

        #Add an arrow from last point to this point
        x = current_point.pos[0]
        y = current_point.pos[1]
        mag = 0.1
        dx = mag / math.sqrt(1+ math.tan(current_point.yaw) * math.tan(current_point.yaw))
        dx = -dx if current_point.yaw > math.pi/2 or current_point.yaw < -math.pi/2 else dx
        dy = dx * math.tan(current_point.yaw)
        ax.arrow(x=x, y=y, dx=np.float(dx), dy=np.float(dy), width=0.05, facecolor='red')
        # print('point id:',  current_point.uid, "x:", round(x, 2), "y:", round(y), " yaw:", round(current_point.yaw,2))      
        
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    ax.add_patch(patch)

    xs, ys = zip(*verts)
    ax.set_aspect('equal', adjustable='box')
    ax.plot(xs, ys, 'x--', lw=2, color='black', ms=10)

    plt.show()
    return 

path_point1 = GeneralPathPoint((0, 0), 0, 'STRAIGHT')
path_point2 = GeneralPathPoint((1, 0), 1.57, 'STRAIGHT')
path_point3 = GeneralPathPoint((1, 1), 1.57, 'STRAIGHT')


input_list_all = [path_point1,path_point2,path_point3]


new_path_points = straight_line_to_curve(input_list_all,nav_type = 'FORWARD')
# new_path_points = PathGeneration.straight_line_to_curve(input_list,'FORWARD')


for i in new_path_points:
    print(i)

plot_path(new_path_points)    