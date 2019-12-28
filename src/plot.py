#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random


x = [[-10846.3, -2582.08, -2.69819, 8.59548, -0.064526, -0.0291316], [-10846.6, -2582.23, -2.70077, 8.59431, -0.064526, -0.0291316], [-10846.9, -2582.37, -2.70335, 8.59315, -0.064526, -0.0291316], [-10847.2, -2582.52, -2.70593, 8.59198, -0.064526, -0.0291316], [-10847.5, -2582.66, -2.70851, 8.59082, -0.064526, -0.0291316], [-10847.9, -2582.81, -2.7111, 8.58965, -0.064526, -0.0291316], [-10848.2, -2582.95, -2.71368, 8.58849, -0.064526, -0.0291316], [-10848.5, -2583.09, -2.71626, 8.58732, -0.064526, -0.0291316], [-10848.8, -2583.23, -2.71884, 8.58616, -0.064526, -0.0291316], [-10849.1, -2583.37, -2.72142, 8.58499, -0.064526, -0.0291316], [-10849.4, -2583.51, -2.724, 8.58383, -0.064526, -0.0291316], [-10849.7, -2583.65, -2.72658, 8.58266, -0.064526, -0.0291316], [-10850.1, -2583.79, -2.72916, 8.5815, -0.064526, -0.0291316], [-10850.4, -2583.93, -2.73174, 8.58033, -0.064526, -0.0291316], [-10850.7, -2584.06, -2.73432, 8.57917, -0.064526, -0.0291316], [-10851, -2584.2, -2.73691, 8.578, -0.064526, -0.0291316], [-10851.3, -2584.33, -2.73949, 8.57684, -0.064526, -0.0291316], [-10851.6, -2584.47, -2.74207, 8.57567, -0.064526, -0.0291316], [-10851.9, -2584.6, -2.74465, 8.57451, -0.064526, -0.0291316], [-10852.3, -2584.73, -2.74723, 8.57334, -0.064526, -0.0291316], [-10852.6, -2584.86, -2.74981, 8.57217, -0.064526, -0.0291316], [-10852.9, -2585, -2.75239, 8.57101, -0.064526, -0.0291316], [-10853.2, -2585.12, -2.75497, 8.56984, -0.064526, -0.0291316], [-10853.5, -2585.25, -2.75755, 8.56868, -0.064526, -0.0291316], [-10853.8, -2585.38, -2.76013, 8.56751, -0.064526, -0.0291316], [-10854.2, -2585.51, -2.76272, 8.56635, -0.064526, -0.0291316], [-10854.5, -2585.64, -2.7653, 8.56518, -0.064526, -0.0291316], [-10854.8, -2585.76, -2.76788, 8.56402, -0.064526, -0.0291316], [-10855.1, -2585.89, -2.77046, 8.56285, -0.064526, -0.0291316], [-10855.4, -2586.01, -2.77304, 8.56169, -0.064526, -0.0291316], [-10855.8, -2586.13, -2.77562, 8.56052, -0.064526, -0.0291316], [-10856.1, -2586.25, -2.7782, 8.55936, -0.064526, -0.0291316], [-10856.4, -2586.38, -2.78078, 8.55819, -0.064526, -0.0291316], [-10856.7, -2586.5, -2.78336, 8.55703, -0.064526, -0.0291316], [-10857, -2586.62, -2.78595, 8.55586, -0.064526, -0.0291316], [-10857.4, -2586.73, -2.78853, 8.5547, -0.064526, -0.0291316], [-10857.7, -2586.85, -2.79111, 8.55353, -0.064526, -0.0291316], [-10858, -2586.97, -2.79369, 8.55237, -0.064526, -0.0291316], [-10858.3, -2587.09, -2.79627, 8.5512, -0.064526, -0.0291316], [-10858.6, -2587.2, -2.79885, 8.55003, -0.064526, -0.0291316], [-10859, -2587.32, -2.80143, 8.54887, -0.064526, -0.0291316], [-10859.3, -2587.43, -2.80401, 8.5477, -0.064526, -0.0291316], [-10859.6, -2587.54, -2.80659, 8.54654, -0.064526, -0.0291316], [-10859.9, -2587.65, -2.80917, 8.54537, -0.064526, -0.0291316], [-10860.3, -2587.76, -2.81176, 8.54421, -0.064526, -0.0291316], [-10860.6, -2587.88, -2.81434, 8.54304, -0.064526, -0.0291316], [-10860.9, -2587.98, -2.81692, 8.54188, -0.064526, -0.0291316], [-10861.2, -2588.09, -2.8195, 8.54071, -0.064526, -0.0291316], [-10861.6, -2588.2, -2.82208, 8.53955, -0.064526, -0.0291316], [-10861.9, -2588.31, -2.82466, 8.53838, -0.064526, -0.0291316]]

cov = [[0.00361139, 0.00611165, 0.00381487, 0.0102369, 0.0640082, 0.0101376], [0.00444011, 0.00875116, 0.00484415, 0.0108246, 0.0704082, 0.0102816], [0.00557407, 0.0126274, 0.00609873, 0.0114452, 0.0768082, 0.0104256], [0.00709173, 0.0181096, 0.0075991, 0.0120992, 0.0832082, 0.0105696], [0.00908511, 0.0256404, 0.00936574, 0.012787, 0.0896082, 0.0107136], [0.0116605, 0.0357424, 0.0114191, 0.0135091, 0.0960082, 0.0108576], [0.0149389, 0.0490246, 0.0137797, 0.0142659, 0.102408, 0.0110016], [0.0190571, 0.0661887, 0.016468, 0.0150579, 0.108808, 0.0111456], [0.0241676, 0.0880364, 0.0195045, 0.0158856, 0.115208, 0.0112896], [0.0304396, 0.115475, 0.0229097, 0.0167494, 0.121608, 0.0114336], [0.0380591, 0.149526, 0.026704, 0.0176498, 0.128008, 0.0115776], [0.0472294, 0.191328, 0.030908, 0.0185873, 0.134408, 0.0117216], [0.0581716, 0.242149, 0.035542, 0.0195623, 0.140808, 0.0118656], [0.0711244, 0.303389, 0.0406266, 0.0205752, 0.147208, 0.0120096], [0.0863447, 0.376588, 0.0461823, 0.0216266, 0.153608, 0.0121536], [0.104108, 0.463433, 0.0522296, 0.0227168, 0.160008, 0.0122976], [0.124707, 0.565764, 0.0587889, 0.0238464, 0.166408, 0.0124416], [0.148455, 0.685584, 0.0658806, 0.0250158, 0.172808, 0.0125856], [0.175682, 0.825064, 0.0735254, 0.0262255, 0.179208, 0.0127296], [0.206737, 0.986548, 0.0817436, 0.027476, 0.185608, 0.0128736], [0.241987, 1.17256, 0.0905558, 0.0287676, 0.192008, 0.0130176], [0.281817, 1.38583, 0.0999824, 0.0301009, 0.198408, 0.0131616], [0.326633, 1.62925, 0.110044, 0.0314763, 0.204808, 0.0133056], [0.376855, 1.90595, 0.120761, 0.0328943, 0.211208, 0.0134496], [0.432922, 2.21926, 0.132154, 0.0343553, 0.217608, 0.0135936], [0.495292, 2.57271, 0.144243, 0.0358598, 0.224008, 0.0137376], [0.564438, 2.97009, 0.157049, 0.0374083, 0.230408, 0.0138816], [0.640851, 3.41539, 0.170592, 0.0390012, 0.236808, 0.0140256], [0.725037, 3.91286, 0.184893, 0.040639, 0.243208, 0.0141696], [0.81752, 4.46698, 0.199972, 0.0423221, 0.249608, 0.0143136], [0.918837, 5.08251, 0.21585, 0.044051, 0.256008, 0.0144576], [1.02954, 5.76445, 0.232547, 0.0458262, 0.262408, 0.0146016], [1.1502, 6.51808, 0.250084, 0.0476481, 0.268808, 0.0147456], [1.2814, 7.34895, 0.268482, 0.0495172, 0.275208, 0.0148896], [1.42372, 8.2629, 0.287759, 0.051434, 0.281608, 0.0150336], [1.57778, 9.26606, 0.307938, 0.0533988, 0.288008, 0.0151776], [1.7442, 10.3649, 0.329039, 0.0554122, 0.294408, 0.0153216], [1.9236, 11.566, 0.351082, 0.0574747, 0.300808, 0.0154656], [2.11663, 12.8767, 0.374087, 0.0595866, 0.307208, 0.0156096], [2.32394, 14.3041, 0.398075, 0.0617485, 0.313608, 0.0157536], [2.54617, 15.856, 0.423067, 0.0639608, 0.320008, 0.0158976], [2.78401, 17.5405, 0.449083, 0.066224, 0.326408, 0.0160416], [3.03813, 19.3659, 0.476144, 0.0685385, 0.332808, 0.0161856], [3.3092, 21.341, 0.504269, 0.0709048, 0.339208, 0.0163296], [3.59791, 23.4748, 0.53348, 0.0733234, 0.345608, 0.0164736], [3.90496, 25.7769, 0.563797, 0.0757947, 0.352008, 0.0166176], [4.23103, 28.2571, 0.59524, 0.0783191, 0.358408, 0.0167616], [4.57683, 30.9255, 0.62783, 0.0808972, 0.364808, 0.0169056], [4.94305, 33.7929, 0.661588, 0.0835294, 0.371208, 0.0170496], [5.3304, 36.8702, 0.696533, 0.0862161, 0.377608, 0.0171936]]


state_x = np.array([each[0] for each in x])
state_y = np.array([each[1] for each in x])
state_t = np.array([each[2] for each in x])
state_v = np.array([each[3] for each in x])
state_w = np.array([each[4] for each in x])
state_a = np.array([each[5] for each in x])

cov_x = np.sqrt(np.array([each[0] for each in cov]))
cov_y = np.sqrt(np.array([each[1] for each in cov]))
cov_t = np.sqrt(np.array([each[2] for each in cov]))
cov_v = np.sqrt(np.array([each[3] for each in cov]))
cov_w = np.sqrt(np.array([each[4] for each in cov]))
cov_a = np.sqrt(np.array([each[5] for each in cov]))


# fig, (xy, t, v, w, a, _) = plt.subplot(2,1, sharex=True)

fig, axes = plt.subplots(3, 2)

axes[0, 0].plot(state_t)
axes[0, 0].fill_between([x for x in range(50)], state_t - cov_t, state_t + cov_t, where = state_t + cov_t > state_t - cov_t, color='lightblue')
axes[0, 0].set_title("theta")

axes[0, 1].plot(state_v)
axes[0, 1].fill_between([x for x in range(50)], state_v - cov_v, state_v + cov_v, where = state_v + cov_v > state_v - cov_v, color='lightblue')
axes[0, 1].set_title("velocity")


axes[1, 0].plot(state_w)
axes[1, 0].fill_between([x for x in range(50)], state_w - cov_w, state_w + cov_w, where = state_w + cov_w > state_w - cov_w, color='lightblue')
axes[1, 0].set_title("yaw_rate")


axes[1, 1].plot(state_a)
axes[1, 1].fill_between([x for x in range(50)], state_a - cov_a, state_a + cov_a, where = state_a + cov_a > state_a - cov_a, color='lightblue')
axes[1, 1].set_title("accelaration")


axes[2, 0].plot(state_x,state_y)
for coef in range(100):
    
    axes[2, 0].plot(state_x - cov_x, state_y - cov_y)
    axes[2, 0].plot(state_x + cov_x, state_y + cov_y)
    axes[2, 0].plot(state_x - cov_x, state_y + cov_y)
    axes[2, 0].plot(state_x + cov_x, state_y - cov_y)
axes[2, 0].set_title("position")

plt.show()
# plt.show()
# plt.plot(state_t)
# plt.show()
# plt.plot(state_v)
# plt.show()
# plt.plot(state_w)
# plt.show()
# plt.plot(state_a)
# plt.show()