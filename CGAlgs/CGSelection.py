'''
File: CGSelection.py
Project: CGAlgs
Description: 
-----
Author: CharlesLee
Created Date: Friday March 10th 2023
'''

from ColumnGeneration import *

class CGWithSelection(ColumnGenerationWithLabeling):
    def reset(self):
        super().__init__(self.graph) # no need to build SP

    def column_generation_before_selection(self):
        # solve RLMP and get duals
        is_feasible = self.solve_RLMP_and_get_duals()
        if is_feasible != 1:
            return 0
        # solve SP
        self.solve_SP()
        # record information
        self.cg_iter_cnt += 1
        self.output_info()
        # break if can't improve anymore
        if self.SP_obj >= -self.EPS:
            return 1
        return -1 # -1 means continue column generation

    def get_column_selection_info(self):
        """ get states of routes (columns) """
        routes = self.labeling_routes
        columns_state = []
        for route in routes:
            dual_sum = 0
            dist_sum = 0
            visit_num = len(route)
            visited = [0] * self.graph.nodeNum
            for i in range(1, len(route)):
                dual_sum += self.duals_of_RLMP[f"R{route[i]}"] 
                dist_sum += self.graph.disMatrix[route[i-1], route[i]]
                visited[route[i]] = 1
            state = [dual_sum, dist_sum, visit_num] + visited # dim = (len(columns), 3 + nodeNum)
            columns_state.append(state)
        """ get states of nodes (constraints) """
        constraints_state = []
        for ni in range(self.graph.nodeNum):
            dual_value = self.duals_of_RLMP[f"R{ni}"]
            demand = self.graph.demand[ni]
            ready_time = self.graph.readyTime[ni]
            due_time = self.graph.dueTime[ni]
            service_time = self.graph.serviceTime[ni]
            state = [dual_value, demand, ready_time, due_time, service_time] # dim = (len(constraints), 5)
            constraints_state.append(state)
        """ get edge_index of columns and constraints """ 
        edge_index = [[], []]
        for ri in range(len(routes)):
            for ni in routes[ri][1:]:
                edge_index[0].append(ri) # dim = (2, len(columns) * len(constraints))
                edge_index[1].append(ni+len(routes)) # ! make idxs different from columns'
        info = {
            "columns_state" : np.array(columns_state), 
            "constraints_state" : np.array(constraints_state), 
            "edge_index" : np.array(edge_index)
        }
        return info

    def select_columns(self, select_result=None):
        if len(self.labeling_routes) == 0:
            return
        if select_result is None:
            delete_idx = np.random.randint(len(self.labeling_routes))
            self.labeling_routes.pop(delete_idx)
        else:
            self.labeling_routes = [self.labeling_routes[i] for i in range(len(self.labeling_routes)) if select_result[i]] 

    def column_generation(self):
        while True:
            CG_flag = self.column_generation_before_selection()
            if CG_flag != -1:
                break
            self.select_columns()
            self.get_columns_and_add_into_RLMP()
        return CG_flag

if __name__ == "__main__":
    file_name = "problems\R101.txt"
    graph = GraphTool.Graph(file_name)
    # test: randomly choose a column to delete
    alg = CGWithSelection(graph) # result: optimal 828.936, time 68.0s
    alg.reset()
    routes = alg.run()
    obj = graph.evaluate(routes)
    print("obj = {}".format(obj))
    graph.render(routes)