import matplotlib.pyplot as plt
from math import *
import numpy as np
import pandas as pd
import statsmodels.api as sm

def cdf_plot(x,xmin,alpha,plt_title):
	n = len(x)
	c1 = sorted(x)
	c2 = [k/float(n) for k in range(n,0,-1)]
	q = sorted(x[x>=xmin])
	tt = c2[c1.index(q[0])]
	cq2 = [k/float(len(q)) for k in range(len(q),0,-1)]
	cq2 = [k*float(tt) for k in cq2]
	cf = [pow(float(k)/xmin,1.-alpha) for k in q]
	cf = [k*float(tt) for k in cf]

	data_dict = {'x':np.log(q), 'y':np.log(cq2)}
	df = pd.DataFrame(data_dict, index=np.arange(len(q)))
	df.replace([np.inf, -np.inf], np.nan,inplace=True)
	df.dropna(inplace=True)

	model = sm.OLS(df['y'], sm.add_constant(df['x'])).fit()
	model.summary()
	df['y_hat'] = df['x']*model.params[1]+model.params[0]

	plt.close()
	plt.ion()
	plt.loglog(c1, c2, 'bo',markersize=8,markerfacecolor=[1,1,1],markeredgecolor=[0,0,1],label='CDF')
	plt.loglog(q, cf, 'k--',linewidth=2, label='Power-Law Fit: a=-{}'.format(np.round(alpha,2)))
	plt.loglog(np.exp(df['x'].values), np.exp(df['y_hat'].values), 'k--',linewidth=2,color='red',label='OLS fit: a={}'.format(np.round(model.params[1],2)))
	plt.axvline(x=xmin, color='k', linestyle='--')
	xr1 = pow(10,floor(log(min(x),10)))
	xr2 = pow(10,ceil(log(min(x),10)))
	yr1 = pow(10,floor(log(1./n,10)))
	yr2 = 1

	plt.axhspan(ymin=yr1,ymax=yr2,xmin=xr1,xmax=xr2)
	plt.ylabel('Pr(X >= x)',fontsize=16);
	plt.xlabel('x',fontsize=16)
	plt.title(plt_title)
	plt.legend()
	plt.draw()

def hist_plot(x, xmin, alpha, plt_title):
	n  = len(x)
	c1 = sorted(x)
	c2 = [k/float(n) for k in range(n,0,-1)]
	q  = sorted(x[x>=xmin])
	cf = [pow(float(k)/xmin,1.-alpha) for k in q]
	tt = c2[c1.index(q[0])]
	cf = [k*float(tt) for k in cf]

	nums, bins = np.histogram(q, bins=1000)
	data_dict = {'x':np.log(bins[1:]), 'y':np.log(nums)}
	df = pd.DataFrame(data_dict, index=np.arange(len(nums)))
	df.replace([np.inf, -np.inf], np.nan,inplace=True)
	df.dropna(inplace=True)

	model = sm.OLS(df['y'], sm.add_constant(df['x'])).fit()

	#plt.scatter(df['x'],df['y'],color='blue')
	plt.figure();
	plt.loglog(bins[1:],nums,label='original data')
	plt.loglog(np.exp(df['x']), np.exp(df['x']*model.params[1] + model.params[0]),color='green',label='OLS fit: a={}'.format(np.round(model.params[1],2)))
	plt.loglog(np.exp(df['x']), np.exp(df['x']*-alpha + model.params[0]),color='gray',label='Power-Law fit: a=-{}'.format(np.round(alpha,2)))
	plt.legend()
	plt.title(plt_title)
	plt.ylabel('Frequency of x',fontsize=16);
	plt.xlabel('x',fontsize=16)
	plt.draw()
