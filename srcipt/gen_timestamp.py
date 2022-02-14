import time

def gen_globel_pose(step):
    const_0 = 0.0000000
    const_1 = 1.0000000

    T_x = step
    T_y = T_z = Q_x =Q_y=Q_z =const_0 
    Q_w = const_1

    return T_x,T_y , T_z , Q_x ,Q_y,Q_z ,Q_w


file1 = open("fake_orb.txt","w+")

start_time = time.time()
current_time=time.time()
a = 4500
while 1:
    if current_time - start_time >= 1/30:
        start_time = current_time
        print(current_time)
        T_x,T_y , T_z , Q_x ,Q_y,Q_z ,Q_w = gen_globel_pose(0.01)
        t_r_txt = " "+str(T_x)+" "+str(T_y)+" "+str(T_z)+" "+str(Q_x)+" "+str(Q_y)+" "+str(Q_z)+" "+str(Q_w)
        print(t_r_txt)
        file1.write(str(current_time)+t_r_txt+"\n")

    current_time=time.time()

file1.close()

