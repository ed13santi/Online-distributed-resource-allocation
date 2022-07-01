import numpy as np
import random
import matplotlib.pyplot as plt



##################################################################################################
##################################### CLASS DECLARATIONS #########################################
##################################################################################################

class Node:
    def __init__(self, capacities):
        self.capacities = capacities # resource capacities for each resource type in the node
        self.remaining_capacities = self.capacities
        self.processing_tasks = []

    def process_new_task(self, task):
        # check that task being added doesn't exceed available resources
        assert task.w > 0
        assert all(rc >= d for rc, d in zip(self.remaining_capacities, task.ds))

        self.processing_tasks.append(task)
        self.remaining_capacities = [rc - d for rc, d in zip(self.remaining_capacities, task.ds)]


class TaskType:
    def __init__(self, feature_vector, wait_time):
        self.mean_service_time = feature_vector[0] # exponential distribution
        self.mean_utility_rate = feature_vector[1] # Poisson distribution
        self.max_wait_time = wait_time # deterministic
        self.mean_demand_resources = feature_vector[2:] # Poisson distribution


class Task:
    def __init__(self, task_type):
        self.s = np.random.exponential(task_type.mean_service_time)
        self.u = np.random.poisson(task_type.mean_utility_rate)
        self.w = task_type.max_wait_time
        self.ds = np.random.poisson(task_type.mean_demand_resources)
               

class Cluster:
    def __init__(self, nodes, network_adjacency_matrix_row, receive_external_tasks_bool, external_task_means=()):
        self.nodes = nodes
        self.received_tasks = []
        self.received_tasks_external = []
        self.tasks_to_be_received = []
        if receive_external_tasks_bool:
            self.external_task_means = external_task_means
        else:
            self.external_task_means = ()
        self.neighbors = self.get_neighbors(network_adjacency_matrix_row)

    def local_utility(self):
        tot = 0
        for node in self.nodes:
            for task in node.processing_tasks:
                tot += task.u
        return tot

    def get_neighbors(self, network_adjacency_matrix_row):
        neighbors = []
        for i in range(network_adjacency_matrix_row.shape[0]):
            if network_adjacency_matrix_row[i] != 0:
                neighbors.append((i,network_adjacency_matrix_row[i]))
        return neighbors

    def initialize_nodes_uniform_distribution(self, n_nodes, lows, highs):
        nodes = []
        for _ in range(n_nodes):
            capacities = []
            for low, high in zip(lows, highs):
                c = np.random.uniform(low, high)
                capacities.append(c)
            nodes.append(Node(capacities))
        self.nodes = nodes

    def receive_external_tasks_func(self, task_types):
        for mean, task_type in zip(self.external_task_means, task_types):
            no_tasks_of_current_type = np.random.poisson(mean)
            for _ in range(no_tasks_of_current_type):
                self.received_tasks.append(Task(task_type))

    def greedy_allocation(self):
        # sort received tasks in ascending order of utility
        self.received_tasks.sort(reverse=False, key=(lambda task: task.u))
        # sort nodes in descending order of node capacity (first CPU and then network)
        self.nodes.sort(reverse=True, key=(lambda node: node.capacities))
        # iterate through tasks in reverse order so that if you delete current element
        # it does not affect the index of the next element in the iteration
        for i in range(len(self.received_tasks)-1, -1, -1):
            for j in range(len(self.nodes)):
                enough_capacity = True
                for c, d in zip(self.nodes[j].remaining_capacities, self.received_tasks[i].ds):
                    if d > c:
                        enough_capacity = False
                if enough_capacity:
                    self.nodes[j].process_new_task(self.received_tasks[i])
                    del self.received_tasks[i]
                    self.nodes.sort(reverse=True, key=(lambda node: node.capacities))
                    break

    def selectAndAllocate(self):
        self.greedy_allocation()

    def random_forwarding(self):
        out = []
        for task in self.received_tasks:
            selected_neighbor = random.choice(self.neighbors)
            out.append((task, selected_neighbor))
        self.received_tasks = []
        return out

    def forwardUnallocatedTasks(self):
        return self.random_forwarding()

    def advanceTime(self, task_types):
        # reduce transfer time by 1 for tasks being transferred and when it reaches 0 add the tasks to received_tasks
        for i in range(len(self.tasks_to_be_received)-1, -1, -1):
            task, transfer_time = self.tasks_to_be_received[i]
            transfer_time -= 1
            if transfer_time == 0:
                self.received_tasks.append(task)
                del self.tasks_to_be_received[i]

        # add externally received tasks to self.received_tasks
        self.receive_external_tasks_func(task_types)

        # reduce processing time by 1 for processing tasks and remove once it reaches 0
        for node in self.nodes:
            for i in range(len(node.processing_tasks)-1, -1, -1):
                task = node.processing_tasks[i]
                task.s -= 1
                if task.s == 0:
                    del node.processing_tasks[i]

    def allocationAndRouting(self):
        self.selectAndAllocate()
        return self.forwardUnallocatedTasks()


##################################################################################################
#################################### AUXILIARY FUNCTIONS #########################################
##################################################################################################

def add_connection(matrix, a, b):
    matrix[a][b] = 1
    matrix[b][a] = 1


##################################################################################################
########################################### MAIN #################################################
##################################################################################################

def main():
    no_clusters = 16

    # matrix of transfer times between clusters (if 0 it means no connection so actually infinity in terms of time)
    A = np.zeros((no_clusters, no_clusters))
    for i in range(no_clusters-1):
        add_connection(A,  i,  i+1)
    add_connection(A, no_clusters-1,  0)
    add_connection(A,  3,  12)
    add_connection(A,  4,  11)

    # task receivers
    C_receivers = [5,6,9,10]

    # received tasks means (tasks follow Poisson distribution)
    received_tasks_means_heavy = [(40, 4,20, 8),
                                  (28, 4,16, 4),
                                  (32,16, 4, 8),
                                  (28,24, 8, 6)]

    received_tasks_means_light = [tuple(el/2 for el in listEl) for listEl in received_tasks_means_heavy]

    # light or heavy task load
    heavy = True # if false we use light

    if heavy:
        received_tasks_means = received_tasks_means_heavy
    else:        
        received_tasks_means = received_tasks_means_light

    # cluster is the list of clusters of respip ources
    clusters = []
    for i in range(no_clusters):
        if i in C_receivers:
            clusters.append(Cluster([], A[i], True, received_tasks_means[C_receivers.index(i)]))
        else:
            clusters.append(Cluster([], A[i], False))
    n_nodes_each_cluster = [32, 72, 52, 84, 64, 76, 60, 64, 76, 60, 80, 44, 64, 88, 56, 52]
    for c, n in zip(clusters, n_nodes_each_cluster):
        c.initialize_nodes_uniform_distribution(n, [50,50], [150,150]) # set nodes to have a random capacity for both CPU and netowrk in range [50,150]
        ### !EXCLUDES HIGH BUT IT DOESN'T MATTER IF VALUES TAKE CONTINUOUS DISTRIBUTION


    # task types
    task_types = [TaskType((20,1,9,8),   10),
                  TaskType((30,5,45,8),  10),
                  TaskType((35,6,15,48), 10),
                  TaskType((50,25,47,43),10)]


    # resources types 
    R = ["CPU", "Network"]


    # communication limit
    communication_limit = 300



    utility_rates = []
    proc_tasks = []

    for time_step in range(100):
        for c in clusters:
            c.advanceTime(task_types)
        for c in clusters:
            forwarded_tasks = c.allocationAndRouting()
            for (task, (cluster_index, transfer_time)) in forwarded_tasks:
                clusters[cluster_index].tasks_to_be_received.append((task, transfer_time))

        cluster_utility_rates = [c.local_utility() for c in clusters]
        utility_rate = sum(cluster_utility_rates)
        utility_rates.append(utility_rate)
        #proc_tasks.append(sum([len(c.tasks_to_be_received) for c in clusters]))

        print(time_step)

    plt.plot(utility_rates)
    plt.show()
            

        


if __name__ == "__main__":
    main()