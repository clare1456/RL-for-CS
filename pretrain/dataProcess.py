import os
import sys
sys.path.append("D:\Code\RL-for-CS")
import numpy as np
import json, tqdm
import gurobipy as gp
from CGAlgs import GraphTool


class MILPSolver:
    """ solve MILP to get labels
    Methods:
        solve(self, present_columns, new_columns, nodeNum) -> solve MILP to get labels

    ps: only consider new columns selection
    """
    def __init__(self, epsilon1=0.00001, epsilon2=0.00001):
        # weight of minimize columnNum
        self.epsilon1 = epsilon1 # coef for present columns
        self.epsilon2 = epsilon2 # coef for new columns

    def solve(self, present_columns, new_columns, nodeNum):
        """ solve new model to get labels"""
        # building model
        MILP = gp.Model("MILP")
        # add columns into MILP
        columns = new_columns + present_columns
        ## add variables
        theta_list = list(range(len(columns)))
        theta = MILP.addVars(theta_list, vtype="C", name="theta")
        y_list = list(range(len(new_columns))) # ! only consider new columns selection
        y = MILP.addVars(y_list, vtype="I", name="y")
        ## set objective
        MILP.setObjective(gp.quicksum((theta[i] * columns[i]["distance"]) for i in range(len(columns))) 
                          + gp.quicksum((y[i] * self.epsilon2) for i in range(len(new_columns))), gp.GRB.MINIMIZE)
        ## set constraints
        MILP.addConstrs(theta[i] <= y[i] for i in range(len(new_columns)))
        MILP.addConstrs(gp.quicksum(theta[i] * columns[i]["onehot_path"][j] for i in range(len(columns))) >= 1 for j in range(1, nodeNum))
        ## set params
        MILP.setParam("OutputFlag", 0)
        MILP.optimize()
        labels = []
        for i in range(len(new_columns)):
            labels.append(round(y[i].X))
        return labels
 

class SLProcessor:
    """ process data for supervice learning
    Methods:
        single_process(self, file_name) -> process single file
        run(self) -> process all files in file_list

    ps: only consider new columns selection
    """
    def __init__(self, file_list, save_path, graph_path_base, columns_path_base, seed=1):
        self.file_list = file_list
        self.save_path = save_path
        self.graph_path_base = graph_path_base
        self.columns_path_base = columns_path_base
        # set random seed
        np.random.seed(seed)
        
    def _read_data(self, graph_path, columns_path):
        """read column generation process data from json file and process 

        Args:
            graph_path (string): graph file path in string form
            columns_path (string): columns file path in string form
        """
        # process graph data
        graph = GraphTool.Graph(graph_path)
        self.nodeNum = graph.nodeNum
        # process columns data
        columns_data = json.load(open(columns_path, 'r'))
        ## preprocess columns
        columnSet = columns_data["columnSet"]
        IterDualValue = columns_data["IterDualValue"] 
        for name, column in columnSet.items():
            path = column["path"]
            onehot_path = np.zeros(self.nodeNum)
            for node in path:
                onehot_path[node] = 1
            column["onehot_path"] = onehot_path.tolist()
            column["demand"] = sum(graph.demand[path])
        ## split iter columns
        IterColumns = columns_data["IterColumns"]
        mini_batches = []
        present_columns = []
        for cg_cnt, column_names in IterColumns.items():
            mini_batch = {"present_columns": present_columns.copy(), "new_columns": []}
            for name in column_names:
                mini_batch["new_columns"].append(columnSet[name]) 
                present_columns.append(columnSet[name]) 
            if len(mini_batch["present_columns"]) > 0 and len(mini_batch["new_columns"]) > 0:
                mini_batch["dual_values"] = IterDualValue[cg_cnt]
                mini_batches.append(mini_batch)
        return mini_batches, graph
    
    def _get_labels(self, mini_batches):
        """add labels to mini_batches with MILP

        Args:
            mini_batches (List[Dict]): iteration data for MILP
                ["present_columns", "new_columns", "dual_values"]
                column: Dict["onehot_path", "distance"]
        """
        for mini_batch in mini_batches:
            labels = self.milp_solver.solve(mini_batch["present_columns"], mini_batch["new_columns"], self.nodeNum)
            mini_batch["labels"] = labels

    def _minibatch2state(self, mini_batches, graph):
        """transfer mini_batches to states 

        Args:
            mini_batches (List[Dict]): iteration data for MILP, 
                ["present_columns", "new_columns", "dual_values], 
                column: Dict["onehot_path", "distance"]
            graph (GraphTool.Graph): graph data

        Returns:
            states (List[Dict]): ["columns_state", "constraints_feature"]
        """
        # process each state
        states = []
        for mini_batch in mini_batches:
            # columns features
            columns_features = []
            # columns = mini_batch["present_columns"] + mini_batch["new_columns"]
            columns = mini_batch["new_columns"]
            labels = mini_batch["labels"]
            duals = np.array(mini_batch["dual_values"])
            for column in columns:
                path = column["path"]
                dual_sum = sum(duals[path])
                column_features = [dual_sum, column["distance"], column["demand"]]
                columns_features.append(column_features) # dim = 3
            # constraints features            
            constraints_features = []
            for node in range(graph.nodeNum):
                constraint_features = [duals[node], graph.location[node][0], graph.location[node][1], graph.demand[node], graph.readyTime[node], graph.dueTime[node]]
                constraints_features.append(constraint_features) # dim = 6
            # edges
            edges = [[], []]
            for ci, column in enumerate(columns):
                path = column["path"]
                for ni in path: # column to node
                    edges[0].append(ci+graph.nodeNum)
                    edges[1].append(ni)
            # pack state
            state = {"columns_state": columns_features, "constraints_state": constraints_features, "edges" : edges, "labels" : labels}
            states.append(state)
        
        return states

    def standardlization(self, states):
        column_features_max = [-np.inf] * 3
        constraint_features_max = [-np.inf] * 6
        column_features_min = [np.inf] * 3
        constraint_features_min = [np.inf] * 6
        max_min_info = {"column_state_max": column_features_max, "column_state_min": column_features_min,
                        "constraint_state_max": constraint_features_max, "constraint_state_min": constraint_features_min}
        # save max/min feature value
        for state in states:
            columns_features = state["columns_state"]
            constraints_features = state["constraints_state"]
            for ci in range(len(columns_features)):
                column_features = columns_features[ci]
                for fi in range(len(column_features)):
                    column_features_max[fi] = max(column_features_max[fi], column_features[fi])
                    column_features_min[fi] = min(column_features_min[fi], column_features[fi])
            for ci in range(len(constraints_features)):
                constraint_features = constraints_features[ci]
                for fi in range(len(constraint_features)):
                    constraint_features_max[fi] = max(constraint_features_max[fi], constraint_features[fi])
                    constraint_features_min[fi] = min(constraint_features_min[fi], constraint_features[fi])
        # max-min standardlization
        for state in states:
            for column_features in state["columns_state"]:
                for fi in range(len(column_features)):
                    column_features[fi] = (column_features[fi] - column_features_min[fi]) / (column_features_max[fi] - column_features_min[fi])
            for constraint_features in state["constraints_state"]:
                for fi in range(len(constraint_features)):
                    constraint_features[fi] = (constraint_features[fi] - constraint_features_min[fi]) / (constraint_features_max[fi] - constraint_features_min[fi])
        return max_min_info
   
    def single_process(self, file_name):
        """
        process single instance
        """
        # build MILPSolver
        self.milp_solver = MILPSolver()
        # read data and process data
        graph_path = self.graph_path_base + file_name + ".json"
        columns_path = self.columns_path_base + file_name + ".json"
        mini_batches, graph = self._read_data(graph_path, columns_path)
        # add labels to mini_batches with MILP
        self._get_labels(mini_batches)
        # save processed data
        mini_batches = self._minibatch2state(mini_batches, graph)
        return mini_batches
    
    def run(self):
        # process data
        states = []
        for file_name in tqdm.tqdm(self.file_list):
            states += self.single_process(file_name) # main process
        # pack file
        max_min_info = self.standardlization(states)
        file = {
            "states" : states, 
            "max_min_info" : max_min_info
        }
        # save file
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(self.save_path + f"mini_batches_standard_{len(self.file_list)}.json", 'w') as f:
            json.dump(file, f)


if __name__ == "__main__":
    # get file list
    graph_path_base = "pretrain\\dataset_solved\\GH200_400_s1-10hard_olved\\GH_instance_1-10hard\\"
    columns_path_base = "pretrain\\dataset_solved\\GH200_400_s1-10hard_olved\\GH_instance_1-10hard_solved\\"
    total_file_list = os.listdir(columns_path_base) 
    file_list = [file_name[:-5] for file_name in total_file_list]
    # set save path
    save_path = "pretrain/dataset_processed/"
    # run
    sl_processor = SLProcessor(file_list, save_path, graph_path_base, columns_path_base)
    sl_processor.run()

    
    # tmp: transfer non standard to standard
    # states = json.load(open("pretrain\dataset_processed\mini_batches_60.json", "r"))
    # max_min_info = sl_processor.standardlization(states)
    # file = {
    #     "states" : states,
    #     "max_min_info" : max_min_info
    # }
    # with open("pretrain\dataset_processed\mini_batches_standard_60.json", 'w') as f:
    #     json.dump(file, f)
