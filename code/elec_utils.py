
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import shapely
import networkx as nx
import pysal as ps
import random
from matplotlib.colors import ListedColormap

import geopandas as gp


from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

cm_bright = ListedColormap(['#0099ff', 'red'])

def plot_counties(polyg):
    """Plot county vote distribution"""
    fig, ax = plt.subplots(figsize=(15,11))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Vote Distribution by County - 2014', fontsize=20, fontweight='bold')
    plot_polygons(ax=ax, geoms=polyg.geometry, linewidth=.91, values=polyg.goppc, colormap='RdBu_r')

def plot_polygons(geoms, ax, values=None, colormap='Set1', facecolor=None, edgecolor=None,
                            alpha=1.0, linewidth=1.0, **kwargs):
    """Makes a MatPlotLib PatchCollection out of Polygon and/or MultiPolygon geometries 
     Thanks to http://stackoverflow.com/a/33753927 and David Sullivan"""
    
    # init list to store 
    patches = []
    newvals = []
    
    for polynum in range(len(geoms)):                          # for  polygon # i
        poly = geoms.iloc[polynum]                              # find data.geometry[i] 
        if type(poly) != shapely.geometry.polygon.Polygon:     # if that is not a shapely Polygon object
            for currpoly in poly.geoms:                         # then for data.geometry[i].geoms
                a = np.asarray(currpoly.exterior)                  # make a an array of those exterior values and
                patches.append(Polygon(a))                           # append ato patches
                if values is not None:                               # if values, add value to newvals
                    newvals.append(values.iloc[polynum])
        else:
            a = np.asarray(poly.exterior)
            patches.append(Polygon(a))
            if values is not None:
                newvals.append(values.iloc[polynum])

    patches = PatchCollection(patches, 
                              facecolor=facecolor, linewidth=linewidth, edgecolor=edgecolor, alpha=alpha, 
                              **kwargs)
    if values is not None:
        patches.set_array(np.asarray(newvals))
        patches.set_cmap(colormap)
        norm = matplotlib.colors.Normalize()
        norm.autoscale(newvals)
        patches.set_norm(norm)
    ax.add_collection(patches, autolim=True)
    ax.autoscale_view()
    return patches


def randomize_election(election):

    # Init Graph Instance
    G = nx.Graph()

    # This calculates the weights between the edges for our graph
    neighbors = ps.weights.Contiguity.Rook.from_dataframe(election)


    # To make graph, we must first initialize nodes
    G.add_nodes_from(range(len(election.state)))


    # now read the pysal neighbors structure
    # and add edges to the graph accordingly
    for i, Ni in neighbors:                # i corresponds to county, Ni: neighboring counties
        edges = [(i, j) for j in Ni]      # create an edge between i and each of its neighbors
        G.add_edges_from(edges)            # add this edge to graph
          
    # and now add the state affiliation of each as a node attribute
    for i in G.nodes():
        G.node[i]['state'] = election.loc[i].state
        
    G.nodes(data=True)[:5] # take a look at the first 5 nodes, for reference
    # now make some 'seed' counties
    state_ids = set(election.state)
    seeds = []
    for s in state_ids: # for each state                  -> add county to that state
        this_state = [n[0] for n in G.nodes(data=True) if n[1]['state'] == s]
        seeds.append(random.choice(this_state))
    node_shortest_paths = {n: (1000, -1, "XX") for n in G.nodes()}

    # for the islands, initialize so that the values won't be overwritten
    for x in neighbors.islands:
        node_shortest_paths[x] = (0, 0, election.loc[x].state)   
        
    # Determine shortest paths to all nodes from the seed counties
    distances_from_seeds = [nx.single_source_shortest_path_length(G, n) for n in seeds]

    # Now iterate
    for (seed, distances, state_id) in zip(seeds, distances_from_seeds, state_ids):
        for target, d in distances.items():     #(0,46), (1,43), (2,48), ...
            if d < node_shortest_paths[target][0]:   # find closest county to each county
                node_shortest_paths[target] = (d, seed, state_id)   # update so (shortest_distance, seed, )
                
    nearest_states = [node_shortest_paths[i][2] for i in node_shortest_paths]
    election.newstate = nearest_states
    states = make_states(election, st='newstate')
    return states


def plot_single_new_states(states, annotate=True, plot_title="", sup_title="", lwd=0):
    fig, ax = plt.subplots(figsize=(15,11))
    for p, t in zip(states.geometry, states.newstate):
        # set up text
        if annotate:
            ax.annotate(xy=(p.centroid.x-40000, p.centroid.y), s=t)
        # plot states
        ax.set_title(plot_title, fontsize=20)
        fig.suptitle(sup_title, fontsize=20, fontweight='bold')
        ax.axis('off')
    states.plot(ax=ax, column='newstate', cmap="Vega20", linewidth=lwd)
    plt.show()



def make_states(c, st='state'):
    st = c.dissolve(by=st, aggfunc='sum', as_index=False)
    st.dempc = st.dem / st.votes
    st.goppc = st.gop / st.votes
    st.margin = st.goppc - st.dempc
    st['win'] = 'D'
    st.loc[st.gop > st.dem, 'win'] = 'R'
    return st



def plot_multiple_new_states(polyg, k=4, fig_size = (20, 17)):
    """Plots k/2 iterations of randomly generated states, alongside their vote distribution"""
    
    if k%2==1:
        raise ValueError("k must be an even integer")
    
    fig = plt.figure(figsize=(fig_size[0],fig_size[1]))
    for i in xrange(k):
        ax = fig.add_subplot(k/2.,2,i+1)
        if i % 2 == 0:
            new_states = randomize_election(polyg)
            for p, t in zip(new_states.geometry, new_states.newstate):
                # set up text
                ax.annotate(xy=(p.centroid.x-40000, p.centroid.y), s=t)
                # plot states
                ax.set_title("Randomly Generated States", fontsize=20, fontweight="bold")
                ax.axis('off')
            new_states.plot(ax=ax, column='newstate', cmap="Vega20", linewidth=0.05)
        else:
            for p, state_ab, winner in zip(new_states.geometry, new_states.newstate, new_states.win):
                # set up text
                ax.annotate(xy=(p.centroid.x-49000, p.centroid.y), s=state_ab, size=13)
                #ax.annotate(xy=(p.centroid.x-40000, p.centroid.y-60000), s=winner, size=8)
                # plot states
                ax.set_title("Republican vs Democrat Victory", fontsize=20, fontweight="bold")
                ax.axis('off')
            new_states.plot(ax=ax, column='win', cmap=cm_bright, linewidth=.2, legend=True)
        plt.tight_layout()
    plt.show()





def get_max_idx(L):
    """Finds max element of L"""
    max_i = 0
    for i in range(len(L)):
        if L[i] >= L[max_i]:
            max_i = i
    return max_i

def remove_i(L, i):
    """Removes item i from L"""
    return L[:i] + L[i+1:]

def insert_i(L, i, x):
    """Inserts item x into position i of L"""
    return L[:i] + [x] + L[i:]

def apportion(pops, states, seats_to_assign=435, initial=1, extras=2, exclude='DC'):
    pops = list(pops)
    states = list(states)
    assigned = [initial] * len(pops)
    ex = states.index(exclude)
    assigned = remove_i(assigned, ex)
    pops = remove_i(pops, ex)
    remaining = seats_to_assign - sum(assigned)
    while remaining > 0:
        priorities = [p / np.sqrt(a * (a + 1)) for p, a in zip(pops, assigned)]
        max_priority = get_max_idx(priorities)
        assigned[max_priority] += 1
        remaining -= 1
    assigned = insert_i(assigned, ex, 1)
    assigned = [__ + 2 for __ in assigned]
    return assigned

def run_election(df, st='state', pop='population', ev='ev'):
    states = make_states(election, st)
    states[ev] = apportion(states[pop], states[st])
    return {'gop': sum(states.ev[states.win == 'R']), 'dem': sum(states.ev[states.win == 'D'])}