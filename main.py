from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Neuron(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = 0
        self.behavior = None

    def choose(self):
        self.behavior = random.choice(['send', 'receive', 'integrate'])

    def act(self):
        if self.behavior == 'send':
            self.send()
        elif self.behavior == 'receive':
            self.receive()
        else:
            self.integrate()

    def send(self):
        print(f"Agent {self.unique_id} sends")

    def receive(self):
        print(f"Agent {self.unique_id} receives")

    def integrate(self):
        print(f"Agent {self.unique_id} integrates")

    def step(self):
        self.choose()
        self.act()

    
class NetworkModel(Model):
    def __init__(self, N, beta):
        self.num_agents = N
        self.beta = beta
        self.schedule = SimultaneousActivation(self)
        self.weight_matrix = np.zeros((N, N))

        for i in range(self.num_agents):
            agent = Neuron(i, self)
            self.schedule.add(agent)


    def update_weights(self, active_agents):
        for agent_i in active_agents:
            for agent_j in active_agents:
                if agent_i != agent_j:
                    self.update_weight(agent_i, agent_j)

    def update_weight(self, agent_i, agent_j):
        if agent_i.behavior == 'send':
            if agent_j.behavior == 'receive':
                self.weight_matrix[agent_i.unique_id, agent_j.unique_id] += 1
                self.weight_matrix[agent_j.unique_id, agent_i.unique_id] += 1
            elif agent_j.behavior == 'integrate':
                self.weight_matrix[agent_i.unique_id, agent_j.unique_id] += 1
                self.weight_matrix[agent_j.unique_id, agent_i.unique_id] += 1                
            elif agent_j.behavior == 'send':
                self.weight_matrix[agent_i.unique_id, agent_j.unique_id] -= 1
                self.weight_matrix[agent_j.unique_id, agent_i.unique_id] -= 1


    def step(self):
        active_agents = random.sample(self.schedule.agents, k = random.randint(1, self.beta))
        for agent in active_agents:
            agent.step()

        self.update_weights(active_agents)

        self.schedule.step()

    def network(self):
        G = nx.Graph()

        for i in range (self.num_agents):
            for j in range (i+1, self.num_agents):
                if self.weight_matrix[i,j] != 0:
                    G.add_edge(i, j, weight = self.weight_matrix[i,j])

        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color = 'lightblue', with_labels = True, node_size = 500, font_size = 10)
        nx.draw_networkx_edges(G, pos, edgelist = edges, width = [weight for weight in weights])

        plt.show()
