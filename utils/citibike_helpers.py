import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import operator
import datetime
import geopandas as gp #might need to install
from shapely.geometry import Point

def get_unique_column_values(df,colname):
	return df[colname].unique()

def generate_clustering_coefficient_plot(g):
    sns.set_style('whitegrid')
    #Ignore nodes with clustering coefficients of zero.
    clustering_coefficients=list(filter
        (lambda y: y[1]>0,sorted(
            nx.clustering(g).items(),key=lambda x: x[1],reverse=True)))
    plt.figure(figsize=(7,7))
    plt.plot(list(map(lambda x: x[1],clustering_coefficients)))
    plt.ylabel("Clustering Coefficient",fontsize=16)
    plt.xlabel("Number of Nodes",fontsize=16)

def get_indegree_and_outdegree(graph):
    """ 
        Return Indegrees
    """
    node_indegrees=list(graph.in_degree().items())
    node_outdegrees=list(graph.out_degree().items())
    return node_indegrees,node_outdegrees

def sort_by_degree(degrees_list,reverse=False):
    return sorted(degrees_list,key=operator.itemgetter(1),reverse=reverse)

def generate_degree_rank_plots(edges_with_weights):
    g=nx.Graph() #Instantiate an Undirected Graph.
    #Add all edges to DiGraph degardless of weight threshold.
    for edge_wt in edges_with_weights:
        g.add_edge(edge_wt['edge'][0],edge_wt['edge'][1])
    deg=list(sorted(nx.degree(g).values(),reverse=True)) #Consider all nodes.
    deg1=deg[10:-10] #Disergard ten nodes with the highest and least degree values.
    deg2=deg[20:-20] #Disregard twenty nodes with the highest and least degree values.
    
    fig,ax=plt.subplots(1,3,figsize=(20,5))
    ax[0].loglog(deg,'b-',marker='o')
    ax[1].loglog(deg1,'b-',marker='o',c='r')
    ax[2].loglog(deg2,'b-',marker='o',c='g')
    ax[0].set_ylabel('Degree',fontsize=18)
    ax[0].set_xlabel('Rank All',fontsize=18)
    ax[1].set_xlabel('Rank excluding \ntop and bottom 10 nodes',fontsize=18)
    ax[2].set_xlabel('Rank excluding \ntop and bottom 20 nodes',fontsize=18)

def generate_degree_rank_plot(edges_with_weights):
    g=nx.Graph() #Instantiate an Undirected Graph.
    #Add all edges to DiGraph degardless of weight threshold.
    for edge_wt in edges_with_weights:
        g.add_edge(edge_wt['edge'][0],edge_wt['edge'][1])
    
    deg=list(sorted(nx.degree(g).values(),reverse=True)) 
    fig,ax=plt.subplots(1,1,figsize=(8,6))
    ax.loglog(deg,'b-',marker='o')
    ax.set_ylabel('Degree',fontsize=18)
    ax.set_xlabel('Rank',fontsize=18)

def load_citibike_data(inputfile,compression='gzip',headerrow=0,separator=','):
    """
        @param : inputfile: full path to the input file.
        @param : compression :  {‘infer’, ‘gzip’, ‘bz2’, ‘zip’, ‘xz’, None}
        @param : headerrow: Row number(s) to use as the column names (default 'infer')
        @param : separator: The symbol used to separate successive values in the file.
        
        Function reads in a csv file in any of the aforementioned `compression` formats and returns a "DataFrame".
    """
    df = pd.read_csv(inputfile,compression=compression,header=headerrow, sep=separator) 
    return df

def calculate_trip_durations_citibike(df):
    
    #convert the Start and End Time columns to datetime.
    df['Start Time']=pd.to_datetime(df['Start Time'])
    df['Stop Time']=pd.to_datetime(df['Stop Time'])

    #Trip Duration is End - Start time. This will result in datetime.timedelta objects, stored in the 'Trip Duration' column.
    df['Trip Duration']=df['Stop Time'] - df['Start Time']  #This is still timedelta.

    #Convert datetime.timedelta object Trip Duration to floating point number of minutes for ease of plotting.
    df['Trip Duration Minutes']=df['Trip Duration'].apply(lambda x: datetime.timedelta.total_seconds(x)/60.0)
    return df

def create_subset_graph(edges_with_weights,thr=0.005,graphtype='Directed'):
    """
        Creates a directed graph 
    """
    #edges[:len(weights)*0.3]
    if graphtype=='Directed':
        g=nx.DiGraph() #Instantiate a Directed Graph Object from NetworkX.
    elif graphtype=='UnDirected':
        g=nx.Graph() #Instantiate an Undirected Graph Object from NetworkX.
    
    thr=thr #Get top 0.5% edges by weight.
    edges_with_weights_new=list(filter(lambda edg: edg['edge'][0]!=edg['edge'][1],
                                   sorted(edges_with_weights,reverse=True,
                                          key=operator.itemgetter('weight'))))[:int(len(edges_with_weights)*thr)]

    #[:len(edges_with_weights)*0.1]
    for edge_wt in edges_with_weights_new:
        g.add_edge(edge_wt['edge'][0],edge_wt['edge'][1],weight=edge_wt['weight'])
    
    return g


def infer_weighted_station_station_network(df):
    dir_edges=dict()
    seen=list()
    for row in df.iterrows():
        st=row[1]['Start Station ID']
        end=row[1]['End Station ID']
        tripduration=row[1]['Trip Duration Minutes']
        if not st in seen:
            seen.append(st)
            dir_edges[st]={end:tripduration}
        else:
            try:
                dir_edges[st][end]+=tripduration
            except KeyError:
                dir_edges[st][end]=tripduration 
    
    edges_with_weights=list()
    #weights=list()
    for st_stn in seen:  
        end_stns=list(dir_edges[st_stn].keys())   
        num_end_stations=len(end_stns)
        for end_stn in end_stns:
            norm_wt=dir_edges[st_stn][end_stn]/num_end_stations
            edges_with_weights.append({'edge':(st_stn,end_stn),'weight':norm_wt})
            
    return edges_with_weights

def create_geodf_citibike_nyc(df,station_ids):
    
    geo_df_dict={'geometry':list(),'station_ids':list()}
    for stn_id in station_ids:  #Iterate over all station_ids.
        _df=df[df['Start Station ID']==stn_id]  #Filter rows where Start Station ID equals stn_id .
        if _df.shape[0]>0:
            lat=_df['Start Station Latitude'].values[0]  #Get the lat value of the particular station.
            lon=_df['Start Station Longitude'].values[0] #Get the lon value of the particular station.
            geo_df_dict['geometry'].append(Point(lon,lat))  #Add this as a Shapely.Point value under the 'geometry' key.
            geo_df_dict['station_ids'].append(stn_id)
        else: 
            _df=df[df['End Station ID']==stn_id]
            if _df.shape[0]>0:
                lat=_df['End Station Latitude'].values[0]
                lon=_df['End Station Longitude'].values[0]
                geo_df_dict['geometry'].append(Point(lon,lat))
                geo_df_dict['station_ids'].append(stn_id)
                
    geo_df_dict['geometry']=list(geo_df_dict['geometry'])
    geo_stations=gp.GeoDataFrame(geo_df_dict)
    geo_stations.drop(geo_stations[geo_stations['geometry']==Point(0,0)].index,inplace=True)
    geo_stations.reset_index(inplace=True)
    geo_stations.to_crs = {'init': 'epsg:4326'}
    return geo_stations

def plot_network(g,node_dist=1.0,nodecolor='g',nodesize=1200,nodealpha=0.6,edgecolor='k',\
                 edgealpha=0.2,figsize=(9,6),title=None,titlefontsize=20,savefig=False,\
                 filename=None,bipartite=False,bipartite_colors=None,nodelabels=None,
                 edgelabels=None):
    pos=nx.spring_layout(g,k=node_dist)
    nodes=g.nodes()
    edges=g.edges()
    plt.figure(figsize=figsize)
    
    nx.draw_networkx_edges(g,pos=pos,edge_color=edgecolor,alpha=edgealpha)
    #nx.draw_networkx_edges(g,pos=pos,edge_color=edgecolor,alpha=edgealpha)
    if bipartite and bipartite_colors!=None:
        bipartite_sets=nx.bipartite.sets(g)
        _nodecolor=[]
        for _set in bipartite_sets:
            _clr=bipartite_colors.pop()
            for node in _set:
                _nodecolor.append(_clr)

        nx.draw_networkx_nodes(g,pos=pos,node_color=_nodecolor,alpha=nodealpha,node_size=nodesize)
    else:
        nx.draw_networkx_nodes(g,pos=pos,node_color=nodecolor,alpha=nodealpha,node_size=nodesize)

    labels={}
    for idx,node in enumerate(g.nodes()):
        labels[node]=str(node)
    
    if nodelabels!=None:
        nx.draw_networkx_labels(g,pos,labels,font_size=16)

    plt.xticks([])
    plt.yticks([])
    if title!=None:
        plt.title(title,fontsize=titlefontsize)
    if savefig and filename!=None:
        plt.savefig(filename,dpi=300)

