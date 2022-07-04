from math import prod
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
        self.taskType = self.task_type
               

class Cluster:
    def __init__(self, nodes, network_adjacency_matrix_row, receive_external_tasks_bool, taskTypes, external_task_means=()):
        self.nodes = nodes
        self.received_tasks = []
        self.received_tasks_external = []
        self.tasks_to_be_received = []
        if receive_external_tasks_bool:
            self.external_task_means = external_task_means
        else:
            self.external_task_means = ()
        self.neighbors = self.get_neighbors(network_adjacency_matrix_row)
        self.taskTypes = taskTypes
        self.Q = self.initialize_Q_table()

    def initialize_Q_table(self):
        max_received_tasks = [100, 50, 25, 25] # 100 means [0, 99]
         # kinda random, find better nubmers or automatic way to determine this
        # task_states = prod(max_received_tasks)

        try:
            no_resource_types = len(self.nodes[0].capacities)
        except:
            raise Exception("Possible that the list of nodes in the cluster is empty")
        #no_nodes = len(self.nodes)
        #resource_states =  self.levels ** (no_nodes, no_resource_types)
        dims_resource_states = [len(self.nodes) for _ in range(self.levels * no_resource_types)]

        return np.zeros(max_received_tasks + dims_resource_states)
        

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

    def get_current_state(self, allocable_tasks): # I think it is calculated using allocable_tasks but I am not 100% sure
        taskState = np.zeros((len(self.taskTypes),))
        for task in allocable_tasks:
            taskTypeIndex = self.taskTypes.index(task.taskType)
            taskState[taskTypeIndex] += 1

        levelsBoundaries = [25, 75]
        n_levels = len(levelsBoundaries) + 1
        n_resource_types = len(self.taskTypes[0].mean_demand_resources)
        resourceState = np.zeros((n_levels, n_resource_types)) # [level0CPU, level1CPU, level2CPU, level0Network, level1Network, level2Network]
        for node in self.nodes:
            for resource_idx, c in enumerate(node.remaining_capacities):
                idx = 0
                for i, el in enumerate(levelsBoundaries):
                    if c > el:
                        idx == i + 1
                resourceState[resource_idx*n_levels+idx] += 1

        return np.concatenate((taskState, resourceState))
        

    def get_allocable(self):
        allocable_tasks = []
        for task in self.received_tasks:
            for node in self.nodes:
                enough_capacity = True
                for c, d in zip(node.remaining_capacities, task.ds):
                    if d > c:
                        enough_capacity = False
                        break
                if enough_capacity:
                    allocable_tasks.append(task)
                    break


    def learned_policy(self, s):
        tmp = self.Q
        for el in s:
            tmp = tmp[el]
        
        

    def learned_local_allocation(self):
        allocable_tasks = self.get_allocable(self)
        while len(allocable_tasks) > 0:
            s = self.get_current_state(allocable_tasks)
            t = self.learnedPolicy(s) # to be defined
            if t == None:
                allocable_tasks = []
            else:
                self.allocate_task(t) # to be implmented, should add task to processing tasks and remove from reeived tasks
                allocable_tasks = self.get_allocable(self)
                self.learn(s, t) # to be implemented


    def selectAndAllocate(self):
        #self.greedy_allocation()
        self.learned_local_allocation()

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
    heavy = False # if false we use light

    if heavy:
        received_tasks_means = received_tasks_means_heavy
    else:        
        received_tasks_means = received_tasks_means_light

    


    # task types
    task_types = [TaskType((20,1,9,8),   10),
                  TaskType((30,5,45,8),  10),
                  TaskType((35,6,15,48), 10),
                  TaskType((50,25,47,43),10)]


    # cluster is the list of clusters of respip ources
    clusters = []
    for i in range(no_clusters):
        if i in C_receivers:
            clusters.append(Cluster([], A[i], True, task_types, received_tasks_means[C_receivers.index(i)]))
        else:
            clusters.append(Cluster([], A[i], False, task_types))
    n_nodes_each_cluster = [32, 72, 52, 84, 64, 76, 60, 64, 76, 60, 80, 44, 64, 88, 56, 52]
    for c, n in zip(clusters, n_nodes_each_cluster):
        c.initialize_nodes_uniform_distribution(n, [50,50], [150,150]) # set nodes to have a random capacity for both CPU and netowrk in range [50,150]
        ### !EXCLUDES HIGH BUT IT DOESN'T MATTER IF VALUES TAKE CONTINUOUS DISTRIBUTION

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