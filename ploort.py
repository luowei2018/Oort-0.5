#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
labelsize_b = 13
linewidth = 2
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
colors = ['#ED4974','#16B9E1','#58DE7B','#F0D864','#FF8057','#8958D3']
colors  =['#FD0707','#0D0DDF','#DDDB03','#129114','#FF8A12','#8402AD']
markers = ['o','^','s','>','P','D']
hatches = ['/' ,'\\','--','x', '+', 'O','-',]
linestyles = ['solid','dashed','dotted','dashdot',(0, (1, 10)),(5, (10, 3))]


def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,lfsize=labelsize_b,legloc='best',
				xticks=None,yticks=None,ncol=None, yerr=None, xticklabel=None,yticklabel=None,xlim=None,ylim=None,ratio=None,
				use_arrow=False,arrow_coord=(0.4,30),markersize=8,bbox_to_anchor=None):
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if yerr is None:
			plt.plot(xx, yy, color = color[i], marker = markers[i], 
				# linestyle = linestyles[i], 
				label = label[i], 
				linewidth=2, markersize=markersize)
		else:
			plt.errorbar(xx, yy, yerr=yerr[i], color = color[i], 
				marker = markers[i], label = label[i], 
				linestyle = linestyles[i], 
				linewidth=2, markersize=markersize)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	if xticks is not None:
		plt.xticks(xticks,fontsize=lfsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lfsize)
	if xticklabel is not None:
		ax.set_xticklabels(xticklabel)
	if yticklabel is not None:
		ax.set_yticklabels(yticklabel)
	if use_arrow:
		ax.text(
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=-45, size=lbsize-8,
		    bbox=dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=2))
	plt.tight_layout()
	if ncol!=0:
		if ncol is None:
			plt.legend(loc=legloc,fontsize = lfsize)
		else:
			if bbox_to_anchor is None:
				plt.legend(loc=legloc,fontsize = lfsize,ncol=ncol)
			else:
				plt.legend(loc=legloc,fontsize = lfsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
	if ratio is not None:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')
	plt.close()

# no loss
y = [[0.09999999999999999, 0.10053514376996804, 0.10053514376996804, 0.11823482428115017, 0.24571086261980835, 0.41146365814696473, 0.5690694888178914, 0.6616214057507988, 0.7350678913738019, 0.7964237220447286, 0.8319848242811501, 0.8550459265175718, 0.8834305111821086, 0.9000698881789138, 0.9078594249201277, 0.9153314696485623, 0.9194149361022363, 0.9207927316293929, 0.9291972843450479, 0.933458466453674], [0.09999999999999999, 0.10027755591054313, 0.10027755591054313, 0.12076277955271564, 0.19415934504792334, 0.3006170127795528, 0.40749400958466453, 0.4780850638977636, 0.5484285143769967, 0.6171785143769968, 0.6582248402555909, 0.6937619808306709, 0.7460383386581468, 0.7765135782747604, 0.8042711661341853, 0.8347563897763578, 0.8493849840255591, 0.8560543130990415, 0.8766773162939296, 0.8892631789137381], [0.09999999999999999, 0.10056509584664537, 0.10056509584664537, 0.12975039936102234, 0.2656729233226837, 0.4371825079872204, 0.5867951277955271, 0.6761661341853035, 0.7459045527156548, 0.8059704472843452, 0.8377116613418529, 0.8571345846645368, 0.8890954472843451, 0.9002256389776357, 0.908560303514377, 0.9165774760383387, 0.919263178913738, 0.920323482428115, 0.9301637380191693, 0.9328095047923324]]
yerr = [[0.0, 0.00107028753993611, 0.00107028753993611, 0.012542895348923168, 0.05551758142832582, 0.06452184714123367, 0.062139082591290154, 0.043340286730634094, 0.03892978620330965, 0.028378577740288492, 0.0229966797355318, 0.024047330962369027, 0.02431666895415823, 0.019387949901691286, 0.018222006287226047, 0.019981558530488865, 0.019905895775283042, 0.01985977643390139, 0.011151611481698643, 0.008256199732875372], [0.0, 0.0008326677316293981, 0.0008326677316293981, 0.014998380033616362, 0.041346888575677074, 0.05955449366472131, 0.06439662496504896, 0.05459057475552142, 0.04638319366236547, 0.0499444302266412, 0.04779865511247092, 0.04302091770292256, 0.04086985095181641, 0.04192966460152943, 0.035289498325885155, 0.025013487874166376, 0.024876950760458955, 0.022911726453197136, 0.02357417324192856, 0.018470533740407236], [0.0, 0.001130412167050239, 0.001130412167050239, 0.02068929361325135, 0.058196801535393895, 0.06387919773835062, 0.05959921926092494, 0.04057942819333001, 0.038114950331299295, 0.02267940972002587, 0.024238951015735605, 0.025077814981584432, 0.02204957241558213, 0.019769024196927634, 0.019249835211712134, 0.02041412648539946, 0.020429654647251435, 0.02013723218109891, 0.010131283036570026, 0.008751545769747819]]

y = np.array(y)*100

x = [[0.1*i for i in range(1,21)] for _ in range(3)]
line_plot(x, y,['L1','L2','L3'],colors,
		'test.eps',
		'xx','yy',
		yerr=yerr)