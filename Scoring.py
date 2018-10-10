import numpy as np
import networkx as nx
from scipy.special import loggamma
import matplotlib.pyplot as plt
import pandas as pd

def random_graph(nodes_number, neighbors_upper_bound=3):
	G = nx.DiGraph()
	ordering = np.random.permutation(nodes_number)
	for i, son in enumerate(ordering):
		G.add_node(son, score=0)
		if i > 0:
			to_sample = min(np.random.randint(0, neighbors_upper_bound), i)
			parents = np.random.choice(ordering[:i], to_sample, replace=False)
			G.add_edges_from(list(zip(parents, [son] * parents.shape[0])))
	nx.draw(G)
	return G

def give_m_alpha(son, parents, data):
	# We get the desired arrays of shape (j, r_i)
	m = data.loc[:,parents + [son]].groupby(parents + [son]).size().unstack(fill_value = 0).values 
	alpha = np.ones(m.shape)
	alpha_0 = np.sum(alpha, axis=1) #shape (j,)
	m_0 = np.sum(m, axis=1) #shape (j,)
	return m, m_0, alpha, alpha_0

def initialize_Bayes_score(G, data):
	total_score = 0
	for son in G.nodes:
		parents =  [predecessor for predecessor in G.predecessors(son)]
		if len(parents) > 0:
			m, m_0, alpha, alpha_0 = give_m_alpha(son, parents, data)
			score = np.sum(loggamma(alpha_0) - loggamma(alpha_0 + m_0)) + np.sum(loggamma(alpha + m) - loggamma(alpha))
			G.node[son]['score'] = score
			total_score += score
		else:
			G.node[son]['score'] = 0
	return

def score_up_edge(G, added_edge, data):
	son = added_edge[1]
	initial_score = G.node[son]['score']
	parents = list(G.predecessors(son))
	parents.append(added_edge[0])
	m, m_0, alpha, alpha_0 = give_m_alpha(son, parents, data)
	new_score = np.sum(loggamma(alpha_0) - loggamma(alpha_0 + m_0)) + np.sum(loggamma(alpha + m) - loggamma(alpha))
	return new_score - initial_score

def score_up_remove(G, rm_edge, data):
	son = rm_edge[1]
	initial_score = G.node[son]['score']
	parents =  list(G.predecessors(son))
	if len(parents) > 1:
		parents.remove(rm_edge[0])
		m, m_0, alpha, alpha_0 = give_m_alpha(son, parents, data)
		new_score = np.sum(loggamma(alpha_0) - loggamma(alpha_0 + m_0)) + np.sum(loggamma(alpha + m) - loggamma(alpha))
	else:
		new_score = 0
	return new_score - initial_score

def local_search(G, steps, data):
	score = 0
	print(G.nodes.data())
	for node in G.nodes:
		score += G.nodes[node]['score']
	print('initial score: ' + str(score))
	for _ in range(steps):
		best_score_update = 0
		action = 0
		update_edge = -1
		for i, node in enumerate(G.nodes):
			existing_parents = set(G.predecessors(node))
			legal_edges = list(zip(list(G.nodes)[:i], i * [node]))
			for edge in legal_edges:
				if edge[0] not in existing_parents:
					score_update = score_up_edge(G, edge, data)
					if score_update > best_score_update:
						best_score_update = score_update
						action = 'edge_add'
						update_edge = edge
		for edge in G.edges:
			score_update = score_up_remove(G, edge, data)
			if score_update > best_score_update:
				best_score_update = score_update
				action = 'edge_rm'
				update_edge = edge
		if action == 'edge_add':
			G.add_edge(*update_edge)
		elif action == 'edge_rm':
			G.remove_edge(*update_edge)
		else:
			exit
		score += best_score_update
		print('score updated to: ' + str(score) + ' after step ' + str(_ + 1))
		print(G.edges)
		nx.draw(G)
	print('finished')
	return G

if __name__ == "__main__":
	#Parameters
	n = 3
	nsteps = 10
	
	small_dataset = pd.read_csv("../small.csv")

	dic = {}
	cpt = 0
	for col in small_dataset.columns:
		dic[col] = cpt
		cpt+=1
	small_dataset = small_dataset.rename(dic, axis='columns')
	del dic

	G = random_graph(len(small_dataset.columns), neighbors_upper_bound=n)

	print('test nodes')
	print(G.node)
	print('test G.edges')
	print(G.edges)

	initialize_Bayes_score(G, small_dataset)
	local_search(G, nsteps, small_dataset)
	


	
