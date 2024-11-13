import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
def moni_plt(x1_list,y1_list,x2_list,y2_list,x_name,y_name,title):
    f_shuju1 = interp1d(x1_list, y1_list, kind="cubic")
    x_dens = np.linspace(min(x1_list),max(x1_list),100)
    y_dens = f_shuju1(x_dens)
    f_shuju2 = interp1d(x2_list, y2_list, kind="cubic")
    x_dens2 = np.linspace(min(x1_list),max(x1_list),100)
    y_dens2 = f_shuju2(x_dens2)
    plt.figure()
    # 绘制原始数据点
    plt.scatter(x1_list, y1_list, color='blue')
    plt.scatter(x2_list, y2_list, color='blue')  
     # 绘制回归线
    plt.plot(x_dens,y_dens,label='U(R)',color='red')
    plt.plot(x_dens2,y_dens2,label='U(C)',color='blue')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.grid(True)
    # plt.ylim(-180,0)
    plt.legend()
    plt.show()

def moni_plt2(x1_list,y1_list,x2_list,y2_list,x_name,y_name,title):
    f_shuju1 = interp1d(x1_list, y1_list, kind="cubic")
    x_dens = np.linspace(min(x1_list),max(x1_list),100)
    y_dens = f_shuju1(x_dens)
    f_shuju2 = interp1d(x2_list, y2_list, kind="cubic")
    x_dens2 = np.linspace(min(x1_list),max(x1_list),100)
    y_dens2 = f_shuju2(x_dens2)
    plt.figure()
    # 绘制原始数据点
    plt.scatter(x1_list, y1_list, color='blue')
    plt.scatter(x2_list, y2_list, color='blue')  
     # 绘制回归线
    plt.plot(x_dens,y_dens,label='1000',color='red')
    plt.plot(x_dens2,y_dens2,label='270',color='blue')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.grid(True)
    # plt.ylim(-180,0)
    plt.legend()
    plt.show()


def one_plot(x_list, y_list, x_name, y_name, title):
    f_shu = interp1d(x_list, y_list, kind="cubic")
    x_dens = np.linspace(min(x_list),max(x_list),300)
    y_dens = f_shu(x_dens)
    plt.figure()
    plt.plot(x_dens,y_dens, color="red")
    plt.scatter(x_list, y_list)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


#------------------------------RC---------------------------------------------------#
rc_f1 = [300, 700, 1000, 1300, 1600, 1900, 2200, 2500, 2800, 3100, 3400, 3700, 4000, 4300, 4600, 4800, 5000]
rc_jao = [78.9,66.3,60.1,52.4,44.9,41.3,40.8,36.7,30.4,30.5,28.3,22.2,20.6,24.9,17.9,15.4,14.6]
rc_u_s1 = [10.6,10.6,10.6,10.4,10.4,10.4,10.4,10.4,10.4,10.4,10.4,10.4,10.4,10.4,10.4,10.4,10.4]
rc_u_r = [2,4.16,5.600,6.56,7.36,7.92,8.40,8.80,9.1,9.0,9.20,9.40,9.60,9.60,10.0,10.0,10.0]
u_r1 = [x/y for x,y in zip(rc_u_r,rc_u_s1)]
rc_f2 = [300,700,1100,1500,1900,2300,2700,3100,3500,3900,4300,4700,5000]
rc_u_s2 = [10.6,10.6,10.6,10.6,10.6,10.6,10.4,10.4,10.4,10.4,10.4,10.4,10.4]
rc_u_c = [10.6,10.0,9,7.8,7,6.4,5.6,5.2,4.6,4.2,4,3.6,3.4]
rc_jao_rs = [-9.5,-20.9,-34.4,-43.1,-48.6,-60.2,-62.4,-70.6,-70.7,-67.0,-69.6,-70.4,-71.8]
u_c = [x/y for x,y in zip(rc_u_c,rc_u_s2)]
# ----------------------------rlc----------------------------------------------------#
rlc_f = [3,13,23,33,43,53,63,73,83,93,100]
rlc_u_s = [10.4,10.8,10.6,10.2,10.2,10.6,10.6,10.6,10.6,10.6,10.6]
rlc_u_r_1000 = [0.472,2.28,5.12,8.00,7.04,5.28,4.16,3.36,2.96,2.56,2.32]
rlc_jiao = [87,74.7,54.7,10.2,-27.4,-50.2,-57.1,-67.1,-71.9,-73.0,-75.0]
rlc_u_r_270 = [0.136,0.64,1.64,4.48,2.88,1.76,1.28,1.00,0.82,0.72,0.64]
f = 1608
c = 47e-9
r = 2000
f0 = 1/(2*np.pi*r*c)
print(f'实验测定频率:{f}')
print (f'理论上的频率：{f0}')
wu_cha = (f0-f)/f0
print(f'误差：{wu_cha}')
f2 = 35.19e3
r1 = 1000
l = 10e-3
c2 = 2.2e-9
f02 = 1/(2*np.pi*pow(l*c2,0.5))
wu_cha2 = (f02-f2)/f02
print(f'实验测定频率:{f2}')
print (f'理论上的频率：{f02}')
print(f'误差：{wu_cha2}')
q = [2*np.pi*f2*l/x for x in [1000,270]]
print(*q)

""" moni_plt(rc_f1,u_r1,rc_f2,u_c,"f(Hz)","U","RC")
moni_plt(rc_f1,rc_jao,rc_f2,rc_jao_rs,"f(HZ)","phase difference",'RC')
moni_plt2(rlc_f,rlc_u_r_1000, rlc_f,rlc_u_r_270,"f(KHZ)",'U(V)',"RLC")
one_plot(rlc_f,rlc_jiao,"f(KHZ)","phase","RLC")
 """