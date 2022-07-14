from cmath import inf
from math import prod
from unittest.mock import NonCallableMagicMock
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
    def __init__(self, feature_vector, wait_time, type_index):
        self.mean_service_time = feature_vector[0] # exponential distribution
        self.mean_utility_rate = feature_vector[1] # Poisson distribution
        self.max_wait_time = wait_time # deterministic
        self.mean_demand_resources = feature_vector[2:] # Poisson distribution
        self.task_type_idx = type_index

    def __eq__(self, other): 
        if not isinstance(other, Task):
            # don't attempt to compare against unrelated types
            return NotImplemented

        cond1 = self.mean_service_time == other.mean_service_time
        cond2 = self.mean_utility_rate == other.mean_utility_rate
        cond3 = self.max_wait_time == other.max_wait_time 
        cond4 = self.mean_demand_resources == other.mean_demand_resources 
        cond5 = self.task_type_idx == other.task_type_idx

        return cond1 and cond2 and cond3 and cond4 and cond5


class Task:
    def __init__(self, task_type):
        self.s = np.random.exponential(task_type.mean_service_time)
        self.u = np.random.poisson(task_type.mean_utility_rate)
        self.w = task_type.max_wait_time
        self.ds = np.random.poisson(task_type.mean_demand_resources)
        self.taskType = task_type

    def __eq__(self, other): 
        if not isinstance(other, Task):
            # don't attempt to compare against unrelated types
            return NotImplemented

        cond1 = self.s == other.s
        cond2 = self.u == other.u
        cond3 = self.w == other.w 
        cond4 = all(self.ds == other.ds) 
        cond5 = self.taskType == other.taskType 

        return cond1 and cond2 and cond3 and cond4 and cond5
               

class Cluster:
    def __init__(self, network_adjacency_matrix_row, receive_external_tasks_bool, taskTypes, n_nodes, lows, highs, external_task_means=(), alpha=0.1, gamma=0.99, levelBoundaries=[25, 75]):
        self.nodes = self.initialize_nodes_uniform_distribution(n_nodes, lows, highs)
        self.levelsBoundaries = levelBoundaries
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
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.prev_state = None
        self.prev_a = None
        self.prev_r = None


    def initialize_Q_table(self):
        # max_received_tasks = [100, 50, 25, 25] # 100 means [0, 99]
        # # kinda random, find better nubmers or automatic way to determine this
        # # task_states = prod(max_received_tasks)

        # try:
        #     no_resource_types = len(self.nodes[0].capacities)
        # except:
        #     raise Exception("Possible that the list of nodes in the cluster is empty")
        # #no_nodes = len(self.nodes)
        # #resource_states =  self.levels ** (no_nodes, no_resource_types)
        # no_levels = len(self.levelsBoundaries) + 1
        # dims_resource_states = [len(self.nodes) for _ in range(no_levels * no_resource_types)]

        # no_actions = [len(max_received_tasks) + 1]

        # length_Q_table = max_received_tasks + dims_resource_states + no_actions

        # return np.zeros(length_Q_table)

        return {}
        

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
        return nodes


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

        n_levels = len(self.levelsBoundaries) + 1
        n_resource_types = len(self.taskTypes[0].mean_demand_resources)
        resourceState = np.zeros((n_levels * n_resource_types,)) # [level0CPU, level1CPU, level2CPU, level0Network, level1Network, level2Network]
        for node in self.nodes:
            for resource_idx, c in enumerate(node.remaining_capacities):
                idx = 0
                for i, el in enumerate(self.levelsBoundaries):
                    if c > el:
                        idx == i + 1
                resourceState[resource_idx*n_levels+idx] += 1

        return np.concatenate((taskState, resourceState))
        

    def get_allocable(self):
        allocable_tasks = []
        task_type_indexes = []
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

        return allocable_tasks


    def init_nested_dict(self, keys, val):
        tmp = val
        for k in np.flip(keys):
            tmp = {k: tmp}
        return tmp

    def modify_nested_dict(self, d, keys, val, i):
        tmp = self.init_nested_dict(keys[i:], val) 
        tmp_dict = d
        for k in keys[:i]:
            if k not in tmp_dict:
                tmp_dict[k] = {}
            tmp_dict = tmp_dict[k]
        tmp_dict[keys[i]] = tmp[keys[i]]
        return d
        

    def Q_s(self, s):
        tmp = self.Q
        for i, el in enumerate(s):
            if el in tmp:
                tmp = tmp[el]
            else:
                self.modify_nested_dict(self.Q, s, [0 for _ in range(len(self.taskTypes) + 1)], i)
                return [0 for _ in range(len(self.taskTypes) + 1)]
        return tmp


    #def learned_policy(self, s):
        #tmp = self.Q_s(s)

        # mask = []
        # for i in range(len(self.taskTypes)):
        #     if s[i] > 0.0:
        #         mask.append(1)
        #     else:
        #         mask.append(0)


        # a = None
        # for i, (switch, val) in enumerate(zip(mask, tmp[:len(self.taskTypes)])):
        #     if switch == 1:
        #         if a == None:
        #             a = i
        #         else:
        #             if val > tmp[a]:
        #                 a = i

        # if a == None:
        #     a = len(self.taskTypes)

        # return a


    def allocate_task(self, a, allocable_tasks):
        task = None
        # for each task, check if it is of the type determined by the action 
        # if it is, pick the one with the highest utility
        for t in allocable_tasks:
            if t.taskType.task_type_idx == a:
                if task == None:
                    task = t
                elif t.u > task.u:
                    task = t

        if task != None:
            # sort nodes in decreasing capacity
            self.nodes.sort(reverse=True, key=(lambda node: node.capacities))
            for j in range(len(self.nodes)):
                enough_capacity = True
                for c, d in zip(self.nodes[j].remaining_capacities, task.ds):
                    if d > c:
                        enough_capacity = False
                if enough_capacity:
                    self.nodes[j].process_new_task(task)
                    for z, t in enumerate(self.received_tasks):
                        if t == task:
                            del self.received_tasks[z]
                    self.nodes.sort(reverse=True, key=(lambda node: node.capacities))
                    return task.u
        
        return 0
        # which node does the task go to? Currently best-first allocation

    
    def Q_s_a(self, s, a):
        return self.Q(s)[a]


    def pi(self, s):
        tmp = self.Q_s(s)
        mask = []
        nonzero = 0
        # create mask of actions that are available and
        # count number of available action types
        for i in range(len(self.taskTypes)):
            if s[i] > 0.0:
                mask.append(1)
                nonzero += 1
            else:
                mask.append(0)

        # add mask element for the None action which is always possible
        mask.append(1)
        nonzero += 1 

        a = [0 for _ in range(len(self.taskTypes)+1)]
        max = -inf
        maxidx = 0
        for i, mask_el in enumerate(mask):
            if mask_el == 1:
                a[i] = self.epsilon / nonzero
                if tmp[i] > max:
                    maxidx = i
        a[maxidx] = self.epsilon / nonzero + 1 - self.epsilon

        return a


    def act(self, s):
        action_probabilities = self.pi(s)
        return random.choices(range(len(action_probabilities)), action_probabilities, k = 1)[0]



    def learn(self, s, a, r, s_new):
        tmp = self.Q_s(s)
        Q_s_new = self.Q_s(s_new)

        tmp[a] = (1-self.alpha)*tmp[a] + self.alpha*(r + self.gamma*(sum([pi_a*Q_s_a for pi_a, Q_s_a in zip(self.pi(s_new),Q_s_new)])))

        self.epsilon *= max(self.epsilon_decay, 0.001) # minimum epsilon is 0.001


    def learned_local_allocation(self):
        allocable_tasks = self.get_allocable()
        s = self.get_current_state(allocable_tasks)
        if self.prev_state != None and self.prev_a != None:
            self.learn(self.prev_state, self.prev_a, self.prev_r, s)
        while len(allocable_tasks) > 0:
            allocable_tasks = self.get_allocable()
            a = self.act(s) # policy can only output doable actions (can't output action of task type that is not the set of currently allocable tasks)
            if a == len(self.taskTypes):
                allocable_tasks = []
                self.prev_state = self.get_current_state(allocable_tasks) # not sure i should be doing this bootstrapping between cycles
                self.prev_a = a
                self.prev_r = 0
            else:
                r = self.allocate_task(a, allocable_tasks) # add task to processing tasks and remove from received tasks
                allocable_tasks = self.get_allocable()
                if allocable_tasks != []:
                    s_new = self.get_current_state(allocable_tasks)
                    self.learn(s, a, r, s_new) 
                else:
                    self.prev_state = s
                    self.prev_a = a
                    self.prev_r = r
                # THIS WAY THE Q VALUE OF THE None action is never updated
                # it is possible that the action a is not feasible and no task is allocated and it gets stuck in a long loop of choosing something that doesnt change the allocable tasks and it keeps choosing
                # the same action


    def selectAndAllocate(self, learned):
        if learned:
            self.learned_local_allocation()
        else:
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


    def allocationAndRouting(self, learnedAllocation):
        self.selectAndAllocate(learnedAllocation)
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

    


    # task types
    task_types = [TaskType((20,1,9,8),   10, 0),
                  TaskType((30,5,45,8),  10, 1),
                  TaskType((35,6,15,48), 10, 2),
                  TaskType((50,25,47,43),10, 3)]


    # cluster is the list of clusters of respip ources
    clusters = []
    n_nodes_each_cluster = [32, 72, 52, 84, 64, 76, 60, 64, 76, 60, 80, 44, 64, 88, 56, 52]
    lows = [50,50]
    highs = [150,150]
    for i, n_nodes in enumerate(n_nodes_each_cluster):
        if i in C_receivers:
            clusters.append(Cluster(A[i], True, task_types, n_nodes, lows, highs, received_tasks_means[C_receivers.index(i)]))
        else:
            clusters.append(Cluster(A[i], False, task_types, n_nodes, lows, highs))

    # resources types 
    R = ["CPU", "Network"]


    # communication limit
    communication_limit = 300



    utility_rates = []
    utility_rates_l = []


    n_trials = 1
    episode_length = 500

    for _ in range(n_trials):
        # learned allocation
        clusters = []
        for i, n_nodes in enumerate(n_nodes_each_cluster):
            if i in C_receivers:
                clusters.append(Cluster(A[i], True, task_types, n_nodes, lows, highs, received_tasks_means[C_receivers.index(i)]))
            else:
                clusters.append(Cluster(A[i], False, task_types, n_nodes, lows, highs))
        for time_step in range(episode_length):
            for c in clusters:
                c.advanceTime(task_types)
            for c in clusters:
                forwarded_tasks = c.allocationAndRouting(True)
                for (task, (cluster_index, transfer_time)) in forwarded_tasks:
                    clusters[cluster_index].tasks_to_be_received.append((task, transfer_time))

            cluster_utility_rates = [c.local_utility() for c in clusters]
            utility_rate = sum(cluster_utility_rates)
            utility_rates.append(utility_rate)
            #proc_tasks.append(sum([len(c.tasks_to_be_received) for c in clusters]))

            print(time_step)

        #greedy allocation

        clusters = []
        for i, n_nodes in enumerate(n_nodes_each_cluster):
            if i in C_receivers:
                clusters.append(Cluster(A[i], True, task_types, n_nodes, lows, highs, received_tasks_means[C_receivers.index(i)]))
            else:
                clusters.append(Cluster(A[i], False, task_types, n_nodes, lows, highs))
        for time_step in range(episode_length):
            for c in clusters:
                c.advanceTime(task_types)
            for c in clusters:
                forwarded_tasks = c.allocationAndRouting(False)
                for (task, (cluster_index, transfer_time)) in forwarded_tasks:
                    clusters[cluster_index].tasks_to_be_received.append((task, transfer_time))

            cluster_utility_rates = [c.local_utility() for c in clusters]
            utility_rate = sum(cluster_utility_rates)
            utility_rates_l.append(utility_rate)
            #proc_tasks.append(sum([len(c.tasks_to_be_received) for c in clusters]))

            print(time_step)

    utility_rates_ave = []
    utility_rates_ave_l = []
    l = len(utility_rates) // n_trials
    for i in range(l):
        s = 0
        sl = 0
        for j in range(n_trials):
            s += utility_rates[i+j*l] / n_trials
            sl += utility_rates_l[i+j*l] / n_trials
        utility_rates_ave.append(s)
        utility_rates_ave_l.append(sl)

    plt.plot(utility_rates_ave)
    plt.plot(utility_rates_ave_l)
    plt.show()
            

        


if __name__ == "__main__":
    main()