import networkx as nx
from node2vec import Node2Vec
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Create a graph
#graph = nx.fast_gnp_random_graph(n=100, p=0.5)
# nx.davis_southern_women_graph #karate_club_graph()#


# total_df_ = pd.read_excel("graph.xlsx")
# def printGraph(triples):
#     G = nx.Graph()
#     for triple in triples:
#         G.add_node(triple[0])
#         G.add_node(triple[1])

#         G.add_edge(triple[0], triple[1])


#     pos = nx.spring_layout(G)
    
#     #font = FontProperties(fname="SimHei.ttf", size=14)  # 步骤二
#     plt.figure()
#     nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
#             node_size=500, node_color='seagreen', alpha=0.9,
#             labels={node: node for node in G.nodes()}, font_family='SimSun')
#     plt.axis('off')
    
#     #plt.legend(prop=font)
#     plt.show()
# triples = [[1,2],[5,6],[6,7]]
# printGraph(triples)

# input()


def plot_net(nx_G, pos, node_labels, node_color=[[.7, .7, .7]]):

    nx.draw(nx_G, pos, labels=node_labels,
            with_labels=True, node_color=node_color)
    # nx.draw_networkx_labels(nx_G,pos,node_labels)
    plt.show()
    plt.savefig('graph.png')



def get_graph_from_txt(file_name):
    node_index = []
    node_feature = []
    with open(file_name, "r") as txt:
        lines = txt.readlines()
        for line in lines[1:]:
            #print(line[:-1].split())
            label, feature = line[:-1].split()[0], line[:-1].split()[1:]
            node_index .append(int(label))
            node_feature.append(list(map(lambda x: float(x), feature)))
    return dict(list(zip(node_index, node_feature)))


def node2vec_run(graph):
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=32, walk_length=30,
                        num_walks=200, workers=4)  # Use temp_folder for big graphs

    # Embed nodes
    # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings

    # Save embeddings for later use
    EMBEDDING_FILENAME = "graph"
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)

    EMBEDDING_MODEL_FILENAME = "graph_model"
    model.save(EMBEDDING_MODEL_FILENAME)

    # Embed edges using Hadamard method
    from node2vec.edges import HadamardEmbedder

    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    # Look for embeddings on the fly - here we pass normal tuples
    edges_embs[('1', '2')]
    ''' OUTPUT
    array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
        ... ... ....
        ..................................................................],
        dtype=float32)
    '''

    # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
    edges_kv = edges_embs.as_keyed_vectors()

    # Look for most similar edges - this time tuples must be sorted and as str
    edges_kv.most_similar(str(('1', '2')))
    EDGES_EMBEDDING_FILENAME = "graph_edges"
    # Save embeddings for later use
    edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)




def graph_cluster(graph):
    node2vec_run(graph)
    pos = get_graph_from_txt("graph")
    pos = sorted(pos.items(), key=lambda x: x[0])
    features = list(map(lambda x: x[1], pos))
    features = np.array(features)
    reducer = umap.UMAP(random_state=42)
    #umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3,
                        #n_components=3).fit_transform(df[feat_cols][:6000].values)
    embedding = reducer.fit_transform(features)
    pos = list(zip(list(range(len(embedding))), embedding))
    pos = dict(pos)
    node_labels = dict([(i, i) for i in graph.nodes()])
    plot_net(graph, pos, node_labels)

if __name__ =="__main__":

    graph = nx.karate_club_graph()
    pos = nx.kamada_kawai_layout(graph)
    node_labels = dict([(i, i) for i in graph.nodes()])
    plot_net(graph, pos, node_labels)
    graph_cluster(graph)