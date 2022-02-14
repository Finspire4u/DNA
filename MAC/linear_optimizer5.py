#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.optimize import linprog
import numpy as np
import math
from scipy.linalg import lu
import scipy
import sympy


# In[2]:


#be careful: max_channel_capacity = 2*max data rate
#the antenna need to turn back and force to receive and forward messages


# In[3]:


print(scipy.__version__)
print(sympy.__version__)


# In[4]:


#extend and combine numpy matrix a&b in format:
#    a \ 0
#    0 \ b
def diagnal_extend(input_a,input_b): #a,b should be 2D matrix
    
    c = np.zeros(( np.size(input_a, 0)+np.size(input_b, 0),(np.size(input_a, 1))+(np.size(input_b, 1)) ))
    
    #print(c)
    
                
    total_row_of_a = np.size(input_a, 0)
    
    total_col_of_a = np.size(input_a, 1)
    
    
    for a in range(total_row_of_a):
        
        for b in range(total_col_of_a):            
            
            c[a][b] = input_a[a][b]
            
            
    for a in range(np.size(input_b, 0)):
        
        for b in range(np.size(input_b, 1)):            
            
            c[a+total_row_of_a][b+total_col_of_a]  = input_b[a][b]    
            
            
    output = c
    
    return output


# In[5]:


# extend numpy array in aix-0

def numpy_extend_along_aix0(input_a,input_b): 
    
    len_of_a = input_a.shape[0]
    
    len_of_b = input_b.shape[0]
    
    total_length = len_of_a +len_of_b
    
    output = np.zeros(total_length)
    
    #print('check a b len:',len_of_a,len_of_b)
    
    for a in range(len_of_a):
        
        output[a] = input_a[a]
        
    for a in range(len_of_b):
        
        output[len_of_a + a] = input_b[a]
        
    return output
    


# In[6]:


# extend and copy past input matrix horezentally n times; Input: matrix_i; output: matrix_i\matrix_i\matrix_i\.....
def horizental_extend(input_matrix,how_many_times):
    
    if how_many_times == 0:
        
        print('return original matrix')
        
        return input_matrix
    
    want_to_have_copy = how_many_times+1
    
    output_shape_row = input_matrix.shape[0]
    
    output_shape_col = input_matrix.shape[1] * want_to_have_copy
    
    output = np.zeros((output_shape_row,output_shape_col))
    
    #print(input_matrix.shape[0],input_matrix.shape[1],want_to_have_copy)
    
    for a in range(input_matrix.shape[0]):
        
        for b in range(input_matrix.shape[1]):
            
            for c in range(want_to_have_copy):
                
                row = a
                
                col = input_matrix.shape[1]*c + b
                
                #print(row,col)
                
                output[row][col] = input_matrix[a][b]
                
    return output

'''
#test
A = np.array([[1,0,1],[0,1,0]])
A = horizental_extend(A,3)
print(A)
'''


# In[7]:


# elminate linear dependent rows in the np matrix
def make_matrix_full_rank(input_matrix_A,input_matrix_b):
    
    _, inds = sympy.Matrix(input_matrix_A).T.rref()
    
    total_rows_of_full_rank_matrix = len(inds)
    
    output_A = np.zeros((total_rows_of_full_rank_matrix,input_matrix_A.shape[1]))
    
    output_b = np.zeros(total_rows_of_full_rank_matrix)
    
    for a in range(total_rows_of_full_rank_matrix):
        
        row = inds[a]
        
        output_b[a] = input_matrix_b[row]
        
        for b in range(input_matrix_A.shape[1]):
            
            col = b
            
            output_A[a][b] = input_matrix_A[row][col]
            
    return [output_A,output_b]
            
            


# In[8]:


#utlize linear linear optimization method to find the lowest cost paths for traffic flows

#input: 
#    total_number_of_nodes : total numbder of nodes for the network

#    nodes_name_and_demand : traffic flow pairs. gives traffic sender and receiver information. 
#                            format : sender node \ receiver node \ data rate
#                            ex:[[0,2,100]] node 0 send 100kbps traffic to node 2

#    node_edge_cost        : link costs.
#                            format : start node \ end node \ link cost 
#                            ex:[[1,0,0.6],[0,1,0.4]] link from node 1 to node 0 has cost 0.6; node0->node1 is 0.4 ; all other undefined links has infinite costs                 

def traffic_flow_linear_optimizer(total_number_of_nodes,                                  nodes_name_and_demand,                                  node_edge_cost,                                  method_name = 'revised simplex',                                  max_channel_capacity = 5000,                                  enable_add_channel_constrains = 1,                                  max_edge_cost = 99):

    number_of_traffic_flows = len(nodes_name_and_demand)
    #print(number_of_traffic_flows)
    
    # notice that, our final goal is try to have a matrix in the format 
    #projective function : min (CX)
    #constrains:           AX = b; all elements in X are free (xi can be any real number)
    
    #--------------------------- find matrix C---------------------------------
    # C - reprsents link cost from node i to j  
    


    C =np.ones((total_number_of_nodes,total_number_of_nodes)) * max_edge_cost

    for a in node_edge_cost:

        C[a[0]][a[1]] = a[2]

    #reshape C, make it 1D array
    C = np.reshape(C,(1,total_number_of_nodes*total_number_of_nodes))
    #print('check C')
    #print(len(C))
    
    C = C[0]
    
    C_final = np.copy(C)
    
    #test_zeros = np.zeros(total_number_of_nodes*total_number_of_nodes)

    # extend C for multiple flows
    for a in range(number_of_traffic_flows-1):
        
        C_final = numpy_extend_along_aix0(C_final, C)
        #C_final = numpy_extend_along_aix0(C_final, test_zeros)
    
    
    #C_final = np.ones(total_number_of_nodes*total_number_of_nodes)
    

 
    #--------------------------- find matrix A---------------------------------
    
    #A -  node-arc incidence matrix
    #make sure that sender send out rate and receiver receive rate is satisifed
    flow_out_matrix = np.zeros((total_number_of_nodes,total_number_of_nodes))
    
    #flow_in_matrix  = np.zeros((total_number_of_nodes,total_number_of_nodes))

    for a in node_edge_cost:
    
        link_start = a[0]
    
        link_end = a[1]
        
        #flow out
    
        inside_flow_out_matrix_row_number = link_start
    
        #inside_flow_out_matrix_col_number = link_start*total_number_of_nodes + link_end
        inside_flow_out_matrix_col_number = link_end
    
        flow_out_matrix[inside_flow_out_matrix_row_number][inside_flow_out_matrix_col_number] = 1
        
        #flow_in_matrix[inside_flow_out_matrix_col_number][inside_flow_out_matrix_row_number]  = 1
    
    #print('check flow_out_matrix')
    
    #print(flow_out_matrix)

    #combine two matrix to create A
    A = np.zeros((total_number_of_nodes,total_number_of_nodes*total_number_of_nodes))
    
    for a in range(total_number_of_nodes):
        
        start_index = a * total_number_of_nodes
        
        for b in range(total_number_of_nodes):
            
            #print(a,start_index+b)
            
            A[a][start_index+b] = flow_out_matrix[a][b]
        
        
        
    #print('check A')
    
    #print(A)
    
    sqr = float(total_number_of_nodes*total_number_of_nodes)
    
    #print(sqr)

    #fix the vlaue with traffic flow in
    for a in range(total_number_of_nodes):
        
        deal_with_node = a
    
        for b in range(int(sqr)):
            
            #row = math.floor(b/total_number_of_nodes)
            
            col = b%total_number_of_nodes
            
            #print(row,col)
            
            if col == deal_with_node:
                
                row = math.floor(b/total_number_of_nodes)
                
                if col!=row:
                
                    A[a][b] = -flow_out_matrix[row][col]
                
                
    #print(A)
    
    #A = A.tolist()
            
    #print(A)            
                
        
        
        
    # since we have multiple flows, need to have multiple A matrix to build constrains; concatenate A diganally    
    
    #A = diagnal_extend(A,A)
    
    A_original =  np.copy(A)
    
    for a in range(number_of_traffic_flows-1):
    
        A = diagnal_extend(A,A_original)
        
    
   
    #--------------------------- find matrix b---------------------------------
    # b  - demand for each node  ----- +traffic flow out the node;-flow into the node
    b = np.zeros((total_number_of_nodes*number_of_traffic_flows,1))

    count = 0

    for a in nodes_name_and_demand:
    
        b[a[0]+count*total_number_of_nodes] = a[2]
    
        b[a[1]+count*total_number_of_nodes] = -a[2]
    
        count +=1
        
    b = np.reshape(b,(1,total_number_of_nodes*number_of_traffic_flows))
    
    b= b[0]
    
    
    '''
    print('check A_eq=lhs_eq -> A  --- before elimnate linear dependent term')    
    print(A.shape)
    print(A)
    print('---------------------------------------')
    
    print('check b_eq=rhs_eq -> b  --- before elimnate linear dependent term')
    print(b.shape)
    print(b)
    print('---------------------------------------')
    '''
    
    #--------------------------- define X boundaries---------------------------------
    # link traffics Xij represents traffics flow from node i to j
    
    #single_boundary = (-1*float("inf"), float("inf"))
    #single_boundary = (0, float("inf"))
    single_boundary = (0, None)
    
    '''
    all_boundary = []

    for a in range(total_number_of_nodes*total_number_of_nodes*number_of_traffic_flows):
    
        all_boundary.append(single_boundary) 
    '''
        
    #---------------------------------------------------------------------------------------   
    #---------------------------------------------------------------------------------------   
    # do Gauss-elimination for the constrains, elminate terms which are un necessary to have
    
    A,b = make_matrix_full_rank(A,b)
    
    #print('check rank of A ',np.linalg.matrix_rank(A))
    #print('check row of A ',A.shape[0])
            
    #------------------ need to have inequality terms , hence each link flow does not go over max channel capacity-------------
    #max_channel_capacity = 5000
    #sqr = total_number_of_nodes*total_number_of_nodes
    
    A_uneq_single_flow = np.zeros((total_number_of_nodes,int(sqr)))
    
    b_uneq = np.ones(total_number_of_nodes)*max_channel_capacity
    
    #addressing A_uneq_single_flow
    
    for a in node_edge_cost:
        
        link_start = a[0]
        
        link_end   = a[1]
        
        row = link_start
        
        col = link_start*total_number_of_nodes + link_end
        
        A_uneq_single_flow[row][col] = 1
        
        row2 = link_end
        
        A_uneq_single_flow[row2][col] = 1
    
    A_ub = horizental_extend(A_uneq_single_flow,how_many_times=(number_of_traffic_flows-1))
    
    #A_ub,b_uneq = make_matrix_full_rank(A_ub,b_uneq)
    
                
    #------------------conclude all above, convert all numpy matrix to general list format------------------
    #objective_function
    obj = C_final
    
    # <= constrian LHS
    lhs_ineq = A_ub

    # <= constrian RHS
    rhs_ineq = b_uneq

    # = constrain LHS
    lhs_eq = (A)

    # = constrain RHS
    #rhs_eq = (np.reshape(b,(1,total_number_of_nodes*number_of_traffic_flows))).tolist()[0]
    rhs_eq = b

    # bounday of X = x1, x2, x3,...
    #bnd = tuple(all_boundary)
    
   
    '''
    print('check c=obj ->C final')
    print(obj.shape)
    print(obj)
    print('---------------------------------------')
    
       
    print('check A_eq=lhs_eq -> A')    
    print(lhs_eq.shape)
    print(lhs_eq)
    print('---------------------------------------')
    
    print('check b_eq=rhs_eq -> b')
    print(rhs_eq.shape)
    print(rhs_eq)
    print('---------------------------------------')
    
    print('check bounds=single_boundary')
    print(len(single_boundary))
    print(single_boundary)
    print('---------------------------------------')
    
    
    print('check A_ub')
    print(A_ub.shape)
    print(A_ub)
    print('---------------------------------------')
    
    print('check b_uneq')
    print(b_uneq.shape)
    print(b_uneq)
    print('---------------------------------------')
    '''
    
    # -----------------use sovler to solve the LP prblem-----------------
    #opt = linprog(c=obj, A_eq=lhs_eq, b_eq=rhs_eq, bounds=single_boundary,method='simplex',options={"disp": True})
    if enable_add_channel_constrains ==1:
    
        opt = linprog(c=obj, A_ub=lhs_ineq ,b_ub=rhs_ineq ,A_eq=lhs_eq, b_eq=rhs_eq, bounds=single_boundary,method=method_name)
        
        #opt = linprog(c=obj, A_ub=lhs_ineq ,b_ub=rhs_ineq ,A_eq=lhs_eq, b_eq=rhs_eq, bounds=single_boundary,method='interior-point',options={"disp": True})
        
    else:
        
        opt = linprog(c=obj, A_eq=lhs_eq, b_eq=rhs_eq, bounds=single_boundary,method = method_name)
    
    #-----------------check -----------------------------------------
    '''
    print(opt.x)
    check_result = opt.x
    print('check sum')
    
    for a in A_ub:
        
        sum = np.dot(a,check_result)
        
        print(sum)
        
    '''
            
            
    '''   
    print('check A_ub*opt.x')
    
    print(np.matmul(A_ub,opt.x))
    '''
    
    
    #------------------extract the results, convert it into 3D matrix   
    
    final_3D_mateix_formate =np.zeros((1,1))
    
    if opt.success == 1:
    
        two_d_result = (opt.x).reshape(number_of_traffic_flows,int(np.size(opt.x, 0)/number_of_traffic_flows))
        #print(two_d_result)

        final_3D_mateix_formate = np.zeros((number_of_traffic_flows,total_number_of_nodes,total_number_of_nodes))
        #print(final_3D_mateix_formate)
    
        for a in range(np.size(final_3D_mateix_formate, 0)):

            for b in range(np.size(final_3D_mateix_formate, 1)):
    
                for c in range(np.size(final_3D_mateix_formate, 2)):    
            
                    row_in_original = a
            
                    col_in_original = c+(b)*total_number_of_nodes
            
                    #print(row_in_original,col_in_original)
            
                    final_3D_mateix_formate[a][b][c] = two_d_result[row_in_original][col_in_original]
                
        #print(final_3D_mateix_formate)
                
    return [opt.success,final_3D_mateix_formate]      

    


# In[9]:


def eliminate_too_small_numbers(input_matrix,total_number_of_nodes):
    
    #print('check input_matrix ')
    #print(input_matrix)
    #print(input_matrix[0])
    
    if len(input_matrix[0]) != total_number_of_nodes:
        
        return ([])
    
    number_of_flows = input_matrix.shape[0]
    
    row             = input_matrix.shape[1]
    
    col             = input_matrix.shape[2]
    
    
    output_matrix = np.zeros((number_of_flows,row,col))
    
    #print('check output shape ',output_matrix.shape)
    
    
    for flow_count in range(number_of_flows):
        
        #print('flow_count ', flow_count)
    
        for count1 in range(row):
            
            #print('count1 ', count1)
        
            for count2 in range(col):
                
                #print('count2 ',count2)
                
                #print(input_matrix[flow_count][count1][count2])
                
                #print('flown umber, row, col :',flow_count,count1,count2)
            
                if input_matrix[flow_count][count1][count2]>1:
                
                    output_matrix[flow_count][count1][count2] = input_matrix[flow_count][count1][count2]
                
    return output_matrix


# In[10]:


def redistribute_traffic(output_flow_traffics,communication_pairs,nodes_name_and_demand):
    
    total_number_of_flows = len(nodes_name_and_demand)
    
    sub_flow_index_record = [[]]*total_number_of_flows
    
    #print(sub_flow_index_record)
    
    for flow_index in range(total_number_of_flows):
        
        flow_start = nodes_name_and_demand[flow_index][0]
        
        flow_end   = nodes_name_and_demand[flow_index][1]
        
        for sub_flow_index in range(len(communication_pairs)):
            
            subflow_start = communication_pairs[sub_flow_index][0]
            
            subflow_end   = communication_pairs[sub_flow_index][1]
            
            #print(flow_start,flow_end,'; ',subflow_start,subflow_end)
            
            if (subflow_start==flow_start) and (subflow_end==flow_end):
                
                #print(flow_start,flow_end,'; ',subflow_start,subflow_end)
                #print('flow_index ',flow_index,'sub_flow_index ',sub_flow_index)
                
                copy_list = sub_flow_index_record[flow_index].copy()
                
                copy_list.append(sub_flow_index)
                
                sub_flow_index_record[flow_index] = copy_list.copy()
            
    #print(sub_flow_index_record)
    
    # find sum of each traffic flow
    sum_of_traffic = [0]*total_number_of_flows
    #print(sum_of_traffic)
    
    
    flow_index = 0
    
    for sub_flows in sub_flow_index_record:
        
        for sub_flow_index in sub_flows:
            
            sum_of_traffic[flow_index]+=output_flow_traffics[sub_flow_index]
            
        flow_index+=1
            
            
    #print(sum_of_traffic)
    
    # find data ratio of each sublofw for each flow
    traffic_ratio = [0]*len(output_flow_traffics)
    
    for index0 in range(len(output_flow_traffics)):
        
        sub_flow_traffic = output_flow_traffics[index0]
               
        index1 = 0
        
        for sub_flow_index in sub_flow_index_record:
            
            if index0 in sub_flow_index:
                
                ratio = sub_flow_traffic/sum_of_traffic[index1]
                
                traffic_ratio[index0] = ratio
                
                break
                
            index1+=1
            
    #print(traffic_ratio)
    
    # redistribut sub flow traffic rate; make sure the sum is equal to required traffic rate
    adjust_data_rate = output_flow_traffics.copy()
    
    for index0 in range(len(output_flow_traffics)):
        
        flow_index = 0
        
        for a in sub_flow_index_record:
            
            if index0 in a:
                
                break
                
            flow_index+=1
        
        total_flow_rate = nodes_name_and_demand[flow_index][-1]
        
        adjust_data_rate[index0] = total_flow_rate*traffic_ratio[index0]
        
    #print(adjust_data_rate)
    
    return adjust_data_rate
    
'''   
output_flow_traffics=[731.1131654410462, 860.7882020642821, 1241.3334774524947, 4999.999588484706]
communication_pairs =[[1, 10], [1, 10], [1, 10], [11, 19]]  
nodes_name_and_demand = [[1, 10, 5000], [11, 19, 5000]]

redistribute_traffic(output_flow_traffics,communication_pairs,nodes_name_and_demand)
'''


# In[11]:


#main function
#input: total_number_of_nodes             (float); 
#       single_flow_max_data_rate in kbps (float); 
#       nodes_name_and_demand             (3 by n list): sender node \ receiver node \ data rate
#       node_edge_cost                    (3 by n list): start node \ end node \ link cost
def find_paths_by_linear_optimizer(total_number_of_nodes,single_flow_max_data_rate,nodes_name_and_demand,node_edge_cost):

      
    max_channel_capacity = 2*single_flow_max_data_rate
    #max_channel_capacity = single_flow_max_data_rate
    
    the_system_can_be_solved = 0
    
    total_number_of_flows = len(nodes_name_and_demand)
    
    count = 1
    
    the_solved_result =[]
    
    #------pure test linear optimizer-------
    
    '''
    [the_system_can_be_solved,the_solved_result] = traffic_flow_linear_optimizer\
    (total_number_of_nodes,\
     nodes_name_and_demand,\
     node_edge_cost,\
     method_name = 'revised simplex',\
     max_channel_capacity = max_channel_capacity,\
     enable_add_channel_constrains = 1)
        
    the_solved_result = eliminate_too_small_numbers(the_solved_result,total_number_of_nodes)
    '''
    
    #------end pure test-------
        
    
    while the_system_can_be_solved==0 and count<=total_number_of_flows:
        
        count+=1
        
        [the_system_can_be_solved,the_solved_result] =         traffic_flow_linear_optimizer(total_number_of_nodes,                                      nodes_name_and_demand,                                      node_edge_cost,                                      method_name =  'revised simplex',                                      max_channel_capacity = max_channel_capacity,                                      enable_add_channel_constrains = 1)
        
        #if failed try another solver
        
        if the_system_can_be_solved==0:
            
                    [the_system_can_be_solved,the_solved_result] =                 traffic_flow_linear_optimizer(total_number_of_nodes,                                              nodes_name_and_demand,                                              node_edge_cost,                                              method_name = 'interior-point',                                              max_channel_capacity = max_channel_capacity,                                              enable_add_channel_constrains = 1)
        
        
        max_channel_capacity = (2+count)*single_flow_max_data_rate
    
    
    #if the system still can't be solved, take the last try, remove max channel capacity constrains
    
    if the_system_can_be_solved == 0:
        
        print('The solver cannot satisfy wireless channel constrain, use minimum hop based optimization')
        
        [the_system_can_be_solved,the_solved_result] =         traffic_flow_linear_optimizer(total_number_of_nodes,                                      nodes_name_and_demand,                                      node_edge_cost,                                      method_name = 'revised simplex',                                      max_channel_capacity = max_channel_capacity,                                      enable_add_channel_constrains = 0)
       
    
    the_solved_result = eliminate_too_small_numbers(the_solved_result,total_number_of_nodes)
    
    
    
    
    output_occupied_links = []
    output_flow_traffics =[]
    communication_pairs = []
    
    
    
    # find occupied links from the matrix
    if the_system_can_be_solved == 1:
        
        [output_occupied_links,output_flow_traffics] = find_occupied_links_by_solved_matrix(the_solved_result,nodes_name_and_demand)
    
    #renew communication pairs;add splited flows into the list
        
    for a in output_occupied_links:
        
        communication_pairs.append([a[0],a[-1]])
        
        
    output_flow_traffics = redistribute_traffic(output_flow_traffics,communication_pairs,nodes_name_and_demand)
    
    return ([the_system_can_be_solved,the_solved_result,output_occupied_links,output_flow_traffics,communication_pairs])


# In[12]:


def find_occupied_links_by_solved_matrix(solved_matrix,nodes_name_and_demand):
    
    total_number_of_flows = len(nodes_name_and_demand)
    
    total_number_of_nodes = solved_matrix.shape[1]
    
    output_occupied_links = []
    
    output_flow_traffics = []
    
    for a in range(total_number_of_flows):
        
        flow_start = nodes_name_and_demand[a][0]
        
        flow_end = nodes_name_and_demand[a][1]
        
        flow_total_data_rate = nodes_name_and_demand[a][2]
        
        #now take a look at the solved matrix
        
        occpupied_link_list = []
        
        split_into_how_many_flows = 1
        
        max_flow_split_node = flow_start
        
        for row in range(total_number_of_nodes):
            
            #print(row)
        
            [none_zero_term_col,none_zero_term_val] = trace_matrix(solved_matrix[a],row)
            
            if len(none_zero_term_col)>split_into_how_many_flows:
                
                split_into_how_many_flows = len(none_zero_term_col)
                
                max_flow_split_node = row
        
            #print(none_zero_term_col,none_zero_term_val)
            
            if len(none_zero_term_col)>0:
                
                for count in range(len(none_zero_term_col)):
                    
                    occpupied_link_list.append([row,none_zero_term_col[count],none_zero_term_val[count]])
                    
        #print(occpupied_link_list)
        
        #print('split into how many sub flows / find how many paths : ',split_into_how_many_flows)
        
        #print('the split at node : ',max_flow_split_node)
        
        # create a list to record splited flows
        
        split_flow_list = []
        
        for count in range(split_into_how_many_flows):
            
            split_flow_list.append([])
            
        #print(split_flow_list)
        
        [none_zero_term_col,none_zero_term_val] = trace_matrix(solved_matrix[a],max_flow_split_node)
        
        #print('at this node, the link start, end, and data rate is: ',max_flow_split_node,none_zero_term_col,none_zero_term_val )
        
        count = 0
        
        for link_end_node in  none_zero_term_col:
            
            traffic_info = [max_flow_split_node,link_end_node,none_zero_term_val[count]]
            
            split_flow_list[count].append(traffic_info)
            
            count+=1
            
            #print(traffic_info)
            
        #print(split_flow_list)
        
        #finish following nodes
        for count in range(split_into_how_many_flows):
            
            start_node = split_flow_list[count][0][0]
            
            end_node = split_flow_list[count][0][1]
            
            traffic = split_flow_list[count][0][2]
            
            current_node = end_node
            
            while current_node!= flow_end:
                
                for single_link_info in occpupied_link_list:
                    
                    #print('check')
                    
                    #print(single_link_info)
                    
                    record_link_start   = single_link_info[0]
                    
                    record_link_end     = single_link_info[1]
                    
                    record_link_traffic = single_link_info[2]
                    
                    if record_link_start == current_node:
                        
                        current_node = record_link_end 
                        
                        split_flow_list[count].append(single_link_info)
                        
                        
        #finish preceding nodes
        
        for count in range(split_into_how_many_flows):
            
            start_node = split_flow_list[count][0][0]
            
            end_node = split_flow_list[count][0][1]
            
            traffic = split_flow_list[count][0][2]
            
            current_node = start_node  
            
            while current_node!= flow_start:

                for single_link_info in occpupied_link_list:
                    
                    #print('check')
                    
                    #print(single_link_info)
                    
                    record_link_start   = single_link_info[0]
                    
                    record_link_end     = single_link_info[1]
                    
                    record_link_traffic = traffic#single_link_info[2] #here, a large flow split into smaller one
                    
                    if record_link_end == current_node:
                        
                        current_node = record_link_start
                        
                        split_flow_list[count].insert(0,[record_link_start,record_link_end,record_link_traffic])
                
                        
        #print('check,',split_flow_list)
        #print('-------------------------------------------------') 

    
        #output_occupied_links= []  
        #output_flow_traffics = []
        
        
        
        for splited_flow in split_flow_list:
            
            single_flow_output_occupied_links =[]
            
            output_flow_traffics.append(splited_flow[0][-1])
            
            single_flow_output_occupied_links.append(splited_flow[0][0])
            
            for splited_links in splited_flow:
                
                single_flow_output_occupied_links.append(splited_links[1])
                
            output_occupied_links.append(single_flow_output_occupied_links)
                
                
    #print(output_occupied_links)
    
    #print(output_flow_traffics)
    
    return ([output_occupied_links,output_flow_traffics])


# In[13]:


#find none zero cterm for a row in a matrix, return colmune number of the none zero terms

def trace_matrix(input_matrix,row_number):
    
    none_zero_term_col= []
    
    none_zero_term_val= []
    
    for a in range(input_matrix.shape[1]):
    
        if input_matrix[row_number][a]!= 0:
            
            none_zero_term_col.append(a)
            
            none_zero_term_val.append(input_matrix[row_number][a])
            
    return ([none_zero_term_col,none_zero_term_val])
    


# In[14]:


'''
node_edge_cost = [[0, 0, 1.0], [0, 1, 1.0], [0, 2, 1.0], [0, 3, 1.0], [0, 5, 1.0], [0, 6, 1.0], [0, 18, 1.0], \
                  [0, 19, 1.0], [0, 20, 1.0], [1, 0, 1.0], [1, 1, 1.0], [1, 5, 1.0], [1, 16, 1.0], [1, 18, 1.0],\
                  [2, 0, 1.0], [2, 2, 1.0], [2, 3, 1.0], [2, 6, 1.0], [2, 13, 1.0], [2, 21, 1.0], [2, 22, 1.0],\
                  [3, 0, 1.0], [3, 2, 1.0], [3, 3, 1.0], [3, 6, 1.0], [3, 7, 1.0], [3, 18, 1.0], [3, 21, 1.0],\
                  [4, 4, 1.0], [4, 7, 1.0], [4, 9, 1.0], [4, 12, 1.0], [4, 14, 1.0], [4, 15, 1.0], [4, 17, 1.0],\
                  [5, 0, 1.0], [5, 1, 1.0], [5, 5, 1.0], [5, 18, 1.0], [5, 19, 1.0], [5, 20, 1.0], [6, 0, 1.0],\
                  [6, 2, 1.0], [6, 3, 1.0], [6, 6, 1.0], [6, 18, 1.0], [6, 21, 1.0], [7, 3, 1.0], [7, 4, 1.0],\
                  [7, 7, 1.0], [7, 9, 1.0], [7, 10, 1.0], [7, 11, 1.0], [7, 12, 1.0], [7, 14, 1.0], [7, 15, 1.0],\
                  [7, 21, 1.0], [7, 23, 1.0], [8, 8, 1.0], [8, 16, 1.0], [8, 17, 1.0], [9, 4, 1.0], [9, 7, 1.0],\
                  [9, 9, 1.0], [9, 11, 1.0], [9, 12, 1.0], [9, 14, 1.0], [9, 15, 1.0], [9, 23, 1.0], [9, 24, 1.0],\
                  [10, 7, 1.0], [10, 10, 1.0], [10, 11, 1.0], [10, 12, 1.0], [10, 23, 1.0], [10, 24, 1.0], \
                  [11, 7, 1.0], [11, 9, 1.0], [11, 10, 1.0], [11, 11, 1.0], [11, 12, 1.0], [11, 14, 1.0], \
                  [11, 15, 1.0], [11, 23, 1.0], [11, 24, 1.0], [12, 4, 1.0], [12, 7, 1.0], [12, 9, 1.0], \
                  [12, 10, 1.0], [12, 11, 1.0], [12, 12, 1.0], [12, 14, 1.0], [12, 15, 1.0], [12, 23, 1.0], \
                  [12, 24, 1.0], [13, 2, 1.0], [13, 13, 1.0], [13, 22, 1.0], [14, 4, 1.0], [14, 7, 1.0], \
                  [14, 9, 1.0], [14, 11, 1.0], [14, 12, 1.0], [14, 14, 1.0], [14, 15, 1.0], [14, 17, 1.0], \
                  [15, 4, 1.0], [15, 7, 1.0], [15, 9, 1.0], [15, 11, 1.0], [15, 12, 1.0], [15, 14, 1.0], \
                  [15, 15, 1.0], [15, 17, 1.0], [15, 24, 1.0], [16, 1, 1.0], [16, 8, 1.0], [16, 16, 1.0],\
                  [17, 4, 1.0], [17, 8, 1.0], [17, 14, 1.0], [17, 15, 1.0], [17, 17, 1.0], [18, 0, 1.0], \
                  [18, 1, 1.0], [18, 3, 1.0], [18, 5, 1.0], [18, 6, 1.0], [18, 18, 1.0], [19, 0, 1.0], \
                  [19, 5, 1.0], [19, 19, 1.0], [19, 20, 1.0], [20, 0, 1.0], [20, 5, 1.0], [20, 19, 1.0], \
                  [20, 20, 1.0], [21, 2, 1.0], [21, 3, 1.0], [21, 6, 1.0], [21, 7, 1.0], [21, 21, 1.0],\
                  [22, 2, 1.0], [22, 13, 1.0], [22, 22, 1.0], [23, 7, 1.0], [23, 9, 1.0], [23, 10, 1.0],\
                  [23, 11, 1.0], [23, 12, 1.0], [23, 23, 1.0], [23, 24, 1.0], [24, 9, 1.0], [24, 10, 1.0],\
                  [24, 11, 1.0], [24, 12, 1.0], [24, 15, 1.0], [24, 23, 1.0], [24, 24, 1.0]]

number_of_nodes = 25

max_channel_capcity = 5000

nodes_name_and_demand = [[1, 10, 4000], [11, 19, 5000]]


[the_system_can_be_solved,the_solved_result,occupied_links,traffic_flow,communication_pairs] = \
find_paths_by_linear_optimizer\
(total_number_of_nodes     = number_of_nodes,\
 single_flow_max_data_rate = max_channel_capcity,\
 nodes_name_and_demand     = nodes_name_and_demand,\
 node_edge_cost            = node_edge_cost)

print('the_system_can_be_solved',the_system_can_be_solved)
print('the_solved_result, ',the_solved_result)
print('occupied_links, ',occupied_links)
print('traffic_flow, ',traffic_flow)
print('communication_pairs, ',communication_pairs)
'''


# In[ ]:




