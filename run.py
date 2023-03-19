import GammaHarmoniousLabelingFinder as ghlf
import networkx as nx

if __name__ == '__main__':
    graph = nx.cycle_graph(9)
    nx.draw_kamada_kawai(graph)
    
    mods = (3,3)
    
    labels = ghlf._get_labeling_set(mods)
    
    ghlf.get_labeling(graph,labels,mods)
    ghlf.get_all_labelings(graph,labels,mods)