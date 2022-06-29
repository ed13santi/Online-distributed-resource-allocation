import numpy as np




##################################################################################################
##################################### CLASS DECLARATIONS #########################################
##################################################################################################

class Node:
    def __init__(self, capacities):
        self.capacities = capacities # resource capacities for each resource type in the node

class Cluster:
    def __init__(self, nodes):
        self.nodes = nodes
        self.tasks_to_be_received = []
        self.received_tasks = []
        self.received_tasks_external = []
        self.receive_external_tasks = False
        self.external_task_means = ()
        self.processing_tasks = []

    def initialize_nodes_uniform_distribution(self, n_nodes, lows, highs):
        nodes = []
        for _ in range(n_nodes):
            capacities = []
            for low, high in zip(lows, highs):
                c = np.random.uniform(low, high)
                capacities.append(c)
            nodes.append(capacities)
        self.nodes = nodes

    def receive_external_tasks(self, task_types):
        new_tasks = []
        for mean, task_type in zip(self.external_task_means, task_types):
            no_tasks_of_current_type = np.random.poisson(mean)
            for _ in range(no_tasks_of_current_type):
                new_tasks.append(Task(task_type))

        self.received_tasks_external = new_tasks 

    def receive_all_tasks(self, task_types):
        self.receive_external_tasks(task_types)
        self.received_tasks = self.tasks_to_be_received + self.received_tasks_external

    def iterate(self, task_types):
        self.receive_all_tasks(task_types)
        # ALLOCATED <-- selectAndAllocate(TASKS)
        # TASKS <-- TASKS \ ALLOCATED
        # forward tasks remaining in TASKS
        # update status of currently processing tasks


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

    # cluster is the list of clusters of respip ources
    clusters = [Cluster([]) for _ in range(16)]
    n_nodes_each_cluster = [32, 72, 52, 84, 64, 76, 60, 64, 76, 60, 80, 44, 64, 88, 56, 52]
    for c, n in zip(clusters, n_nodes_each_cluster):
        c.initialize_nodes_uniform_distribution(n, [50,50], [150,150]) # set nodes to have a random capacity for both CPU and netowrk in range [50,150]
        ### !EXCLUDES HIGH BUT IT DOESN'T MATTER IF VALUES TAKE CONTINUOUS DISTRIBUTION


    # matrix of transfer times between clusters
    A = np.zeros((len(clusters), len(clusters)))
    for i in range(15):
        add_connection(A,  i,  i+1)
    add_connection(A, 15,  0)
    add_connection(A,  3,  12)
    add_connection(A,  4,  11)


    # task types
    task_types = [TaskType((20,1,9,8),   10),
                  TaskType((30,5,45,8),  10),
                  TaskType((35,6,15,48), 10),
                  TaskType((50,25,47,43),10)]


    # resources types 
    R = ["CPU", "Network"]


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

    for i, c in enumerate(clusters):
        if i in C_receivers:
            c.receive_external_tasks = True
            c.external_task_means = received_tasks_means[C_receivers.index(i)]


    # communication limit
    communication_limit = 300



    for time_step in range(2000):
        # receive tasks
        for c in clusters:
            c.iterate()

        


if __name__ == "main":
    main()