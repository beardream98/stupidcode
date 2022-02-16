class Solution:
    def canFinish(self, numCourses, prerequisites) -> bool:
        node_state=[0]*numCourses
        neibor_node=[[] for i in range(numCourses)]
        for request in prerequisites:
            neibor_node[request[0]].append(request[1])
        global_flag=True
        def dfs(node_id):
            nonlocal global_flag
            if node_state[node_id]==1:
                global_flag=False
            elif node_state[node_id]==0 :
                if  len(neibor_node[node_id])!=0:
                    node_state[node_id]=1
                    for neibor_id in neibor_node[node_id]:
                        dfs(neibor_id)
                node_state[node_id]=2
        for i in range(numCourses):
            if node_state[i]==0:
                dfs(i)
            if global_flag==False:
                return False
        
        return global_flag
s=Solution()
numCourses = 2
prerequisites = [[1,0],[0,1]]
print(s.canFinish(numCourses,prerequisites))

                
            

