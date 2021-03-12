from HeadMoves import *
from Bandwidth import *
from DDP import *
from DDPOnline import *
from Bola3d import *
from Video import *

N = 200
D = 10
M = 5
buffer_size = 20
delta = 5
gamma = 2
t_0 = delta / 10
sizes = [i for i in range(M)]
v_coeff = 1
values = [i for i in range(M)]

min_band = 10
max_band = 50

video = Video(N, delta, D, values, sizes, buffer_size)
bandwidth = Bandwidth(min_band, max_band)

bola3d = Bola3d(video, gamma, v_coeff)
ddp = DDP(video,buffer_size, bandwidth, gamma, t_0)
ddp_online = DDPOnline(video,buffer_size, bandwidth, gamma, t_0)

