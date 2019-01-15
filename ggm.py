import time

from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

N_atom_features = 9
N_bond_features = 5
N_extra_atom_features = 5
N_extra_bond_features = 6


class ggm(torch.nn.Module):
    def __init__(self, args):
        super(ggm, self).__init__()

        dim_of_node_vector = args.dim_of_node_vector
        dim_of_edge_vector = args.dim_of_edge_vector
        dim_of_FC = args.dim_of_FC
        self.dim_of_graph_vector = dim_of_node_vector*2
        self.dim_of_node_vector = dim_of_node_vector 
        
        self.enc_U = nn.ModuleList([nn.Linear(2*dim_of_node_vector+dim_of_edge_vector+4, dim_of_node_vector) for k in range(3)])
        self.enc_C = nn.ModuleList([nn.GRUCell(dim_of_node_vector, dim_of_node_vector) for k in range(3)])
        
        self.init_scaffold_U = nn.ModuleList([nn.Linear(2*dim_of_node_vector+dim_of_edge_vector, dim_of_node_vector) for k in range(3)])
        self.init_scaffold_C = nn.ModuleList([nn.GRUCell(dim_of_node_vector, dim_of_node_vector) for k in range(3)])
        
        self.prop_add_node_U = nn.ModuleList([nn.Linear(3*dim_of_node_vector+dim_of_edge_vector+4, dim_of_node_vector) for k in range(2)])
        self.prop_add_node_C = nn.ModuleList([nn.GRUCell(dim_of_node_vector, dim_of_node_vector) for k in range(2)])
        
        self.prop_add_edge_U = nn.ModuleList([nn.Linear(3*dim_of_node_vector+dim_of_edge_vector+4, dim_of_node_vector) for k in range(2)])
        self.prop_add_edge_C = nn.ModuleList([nn.GRUCell(dim_of_node_vector, dim_of_node_vector) for k in range(2)])
        
        self.prop_select_node_U = nn.ModuleList([nn.Linear(3*dim_of_node_vector+dim_of_edge_vector+4, dim_of_node_vector) for k in range(2)])
        self.prop_select_node_C = nn.ModuleList([nn.GRUCell(dim_of_node_vector, dim_of_node_vector) for k in range(2)])
        
        self.prop_select_isomer_U = nn.ModuleList([nn.Linear(3*dim_of_node_vector+dim_of_edge_vector+4, dim_of_node_vector) for k in range(2)])
        self.prop_select_isomer_C = nn.ModuleList([nn.GRUCell(dim_of_node_vector, dim_of_node_vector) for k in range(2)])
        
        self.add_node1 = nn.Linear(self.dim_of_graph_vector+dim_of_node_vector+4, dim_of_FC)
        self.add_node2 = nn.Linear(dim_of_FC, dim_of_FC)
        self.add_node3 = nn.Linear(dim_of_FC, N_atom_features)

        self.add_edge1 = nn.Linear(self.dim_of_graph_vector+dim_of_node_vector+4, dim_of_FC)
        self.add_edge2 = nn.Linear(dim_of_FC, dim_of_FC)
        self.add_edge3 = nn.Linear(dim_of_FC, N_bond_features)
        
        self.select_node1 = nn.Linear(dim_of_node_vector*2+dim_of_node_vector+4, dim_of_FC)
        self.select_node2 = nn.Linear(dim_of_FC, dim_of_FC)
        self.select_node3 = nn.Linear(dim_of_FC, 1)
        
        self.select_isomer1 = nn.Linear(dim_of_node_vector*1+dim_of_node_vector+4, dim_of_FC)
        self.select_isomer2 = nn.Linear(dim_of_FC, dim_of_FC)
        self.select_isomer3 = nn.Linear(dim_of_FC, 1)

        self.predict_property1 = nn.Linear(dim_of_node_vector*2, 512)
        self.predict_property2 = nn.Linear(512, 512)
        self.predict_property3 = nn.Linear(512, 1)
        
        self.cal_graph_vector1 = nn.Linear(dim_of_node_vector, self.dim_of_graph_vector)
        self.cal_graph_vector2 = nn.Linear(dim_of_node_vector, self.dim_of_graph_vector)
        self.cal_encoded_vector1 = nn.Linear(dim_of_node_vector, dim_of_node_vector)
        self.cal_encoded_vector2 = nn.Linear(dim_of_node_vector, dim_of_node_vector)

        self.init_graph_state1 = nn.Linear(self.dim_of_graph_vector, self.dim_of_graph_vector)
        self.init_node_state1 = nn.Linear(dim_of_node_vector+self.dim_of_graph_vector, dim_of_node_vector)
        self.init_edge_state1 = nn.Linear(self.dim_of_graph_vector+dim_of_edge_vector, dim_of_edge_vector)

        self.original_node_embedding = nn.Linear(N_atom_features+N_extra_atom_features, dim_of_node_vector, bias = False)
        self.original_edge_embedding = nn.Linear(N_bond_features+N_extra_bond_features, dim_of_edge_vector, bias = False)
        self.node_embedding = nn.Linear(N_atom_features, dim_of_node_vector, bias = False)
        self.edge_embedding = nn.Linear(N_bond_features, dim_of_edge_vector, bias = False)
        
        self.mean = nn.Linear(dim_of_node_vector, dim_of_node_vector)
        self.logvar = nn.Linear(dim_of_node_vector, dim_of_node_vector)

    def forward(self, s1, s2, condition1, condition2, beta1 = 0.005):
        #make graph of moleculea and scaffold. *_save means this graph will not be changed
        g_save, h_save, scaffold_g_save, scaffold_h_save = make_graphs(s1, s2)
        if g_save is None and h_save is None:
            return None

        #make same graph of moleculea and scaffold as above. this graph can be changed
        g, h, scaffold_g, scaffold_h = make_graphs(s1, s2, extra_atom_feature=True, extra_bond_feature=True)
        
        #collect losses
        add_node_losses = []
        add_edge_losses = []
        select_node_losses = []
        
        #embede node state of graph
        self.embede_graph(g, h)
        self.embede_graph(scaffold_g, scaffold_h)

        #from numpy to torch tensor
        condition = create_var(torch.from_numpy(np.array(condition1+condition2)).float().unsqueeze(0))

        #encode node state of graph
        self.encode(g, h, condition)

        #make one vector representing graph using all node vectors 
        encoded_vector = self.cal_encoded_vector(h)

        #reparameterization trick. this routine is needed for VAE.
        latent_vector, mu, logvar = self.reparameterize(encoded_vector)
        latent_vector_with_condition = torch.cat([latent_vector, condition], -1)
         
        #encode node state of scaffold graph
        self.init_scaffold_state(scaffold_g, scaffold_h)

        #check which node is included in scaffold and which node is not
        leaves = [i for i in h_save.keys() if i not in scaffold_h.keys()]
        for idx in leaves:
            #determine which node should be added and calculate the loss
            new_node = self.add_node(scaffold_g, scaffold_h, latent_vector_with_condition)
            add_node_losses.append((-h_save[idx]*torch.log(new_node+1e-6)).sum())
            
            #add new node to the graph and initialize the new node state
            scaffold_h_save[idx] = h_save[idx]
            scaffold_h[idx] = self.init_node_state(scaffold_h, scaffold_h_save[idx])

            #find the edge which is connected to the new node
            edge_list = [e for e in g_save[idx] if e[1] in list(scaffold_h.keys())]
            
            for edge in edge_list:
                #determin which edge type is added and calculate the corresponding loss
                new_edge = self.add_edge(scaffold_g, scaffold_h, latent_vector_with_condition)
                add_edge_losses.append((-edge[0]*torch.log(new_edge+1e-6)).sum())
                target = create_var(one_hot(torch.FloatTensor([list(scaffold_h.keys()).index(edge[1])]),len(scaffold_h)-1 ))
                #determin which node is connected through seleccted edge and calculate the corresponding loss
                selected_node = self.select_node(scaffold_g, scaffold_h, latent_vector_with_condition).view(target.size())
                select_node_losses.append((-target*torch.log(1e-6+selected_node)).sum())
                
                #add edge to the graph and initialize the new node state
                if idx not in scaffold_g_save:
                    scaffold_g_save[idx]=[]
                    scaffold_g[idx]=[]
                scaffold_g_save[idx].append(edge)
                scaffold_g[idx].append(( self.init_edge_state(scaffold_h, edge[0]), edge[1]))
        
                if edge[1] not in scaffold_g_save:
                    scaffold_g_save[edge[1]]=[]
                    scaffold_g[edge[1]]=[]
                scaffold_g_save[edge[1]].append((edge[0], idx))
                scaffold_g[edge[1]].append(( self.init_edge_state(scaffold_h, edge[0]), idx))

            #the edge should not be added more. calculate the corresponding loss
            new_edge = self.add_edge(scaffold_g, scaffold_h, latent_vector_with_condition)
            end_add_edge = create_var(one_hot(torch.FloatTensor([4]),5 ))
            add_edge_losses.append((-end_add_edge*torch.log(1e-6+new_edge)).sum())

        #the node should not be added more. calculate the corresponding loss
        new_node = self.add_node(scaffold_g, scaffold_h, latent_vector_with_condition)
        end_add_node = create_var(one_hot(torch.FloatTensor([8]),9 ))
        add_node_losses.append((-end_add_node*torch.log(1e-6+new_node)).sum())
        
        #convert list to the torch tensor
        total_add_node_loss = torch.stack(add_node_losses).sum()
        if len(add_edge_losses)>0:
            total_add_edge_loss = torch.stack(add_edge_losses).sum()
            total_select_node_loss = torch.stack(select_node_losses).sum()
        else:
            total_add_edge_loss = 0.0
            total_select_node_loss = 0.0

        #check whether reconstructed graph is same as the input graph
        if not is_equal_node_type(scaffold_h_save, h_save) :
            print ('node miss match')
            print (s1)
            print (s2)
        if not is_equal_edge_type(scaffold_g_save, g_save) :
            print ('edge miss match')
            print (s1)
            print (s2)

        #reconstruction loss
        total_loss1 = total_add_node_loss + total_add_edge_loss + total_select_node_loss

        #VAE loss
        total_loss2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())*beta1
        #total_loss3 = (c-create_var(torch.from_numpy(a)).type(torch.FloatTensor)).pow(2).sum()*beta2

        #select isomer
        isomers = enumerate_molecule(s1)
        selected_isomer, target = self.select_isomer(s1, latent_vector_with_condition)
        
        #isomer loss
        total_loss4 = (selected_isomer-target).pow(2).sum()
        return scaffold_g, scaffold_h, total_loss1, total_loss2, total_loss4

    def sample(self, s1=None, s2=None, latent_vector = None, condition1 = None, condition2 = None, stochastic = False):
        if s2 is None:
            print ('when you sample, you must give scaffold')
            return None
        elif s1 is not None:
            g_save, h_save, scaffold_g_save, scaffold_h_save = make_graphs(s1, s2)
            if g_save is None and h_save is None:
                return None
            g, h, scaffold_g, scaffold_h = make_graphs(s1, s2, extra_atom_feature=True, extra_bond_feature=True)
            
            self.embede_graph(g, h)
            self.embede_graph(scaffold_g, scaffold_h)

            self.encode(g, h)
            encoded_vector = self.cal_encoded_vector(h)
            latent_vector, mu, logvar = self.reparameterize(encoded_vector)


        elif s1 is None:
            scaffold_g_save, scaffold_h_save = make_graph(s2)
            if scaffold_g_save is None and scaffold_h_save is None:
                return None
            scaffold_g, scaffold_h = make_graph(s2, extra_atom_feature=True, extra_bond_feature=True)
            
            self.embede_graph(scaffold_g, scaffold_h)
            if latent_vector is None:
                latent_vector = create_var(torch.randn(1, self.dim_of_node_vector)) 

        self.init_scaffold_state(scaffold_g, scaffold_h)

        if condition1 is None or condition2 is None:
            a = np.random.uniform(0,1,1)[0]
            b = np.random.uniform(0,1,1)[0]
            condition1 = np.array([[a, 1-a]])
            condition2 = np.array([[b, 1-b]])
        
        condition = create_var(torch.from_numpy(np.concatenate([condition1, condition2], -1)).float())
                        
        latent_vector = torch.cat([latent_vector, condition], -1)

        for null_index1 in range(100):
            
            new_node = self.add_node(scaffold_g, scaffold_h, latent_vector)
            new_node = probability_to_one_hot(new_node, stochastic)
            if np.argmax(new_node.data.cpu().numpy()[0]) == N_atom_features-1:
                break

            idx = len(scaffold_h)
            scaffold_h_save[idx] = new_node
            scaffold_h[idx] = self.init_node_state(scaffold_h, new_node)
            
            for null_index2 in range(5):            
                new_edge = self.add_edge(scaffold_g, scaffold_h, latent_vector)
                new_edge = probability_to_one_hot(new_edge, stochastic)

                if np.argmax(new_edge.data.cpu().numpy()[0]) == N_bond_features-1:
                    break
                selected_node = self.select_node(scaffold_g, scaffold_h, latent_vector).view(1,-1)
                selected_node = list(scaffold_h.keys())[np.argmax(probability_to_one_hot(selected_node, stochastic).data.cpu().numpy()[0])]
                if idx not in scaffold_g_save:
                    scaffold_g_save[idx]=[]
                    scaffold_g[idx]=[]
                scaffold_g_save[idx].append((new_edge, selected_node))
                scaffold_g[idx].append(( self.init_edge_state(scaffold_h, new_edge), selected_node))
        
                if selected_node not in scaffold_g_save:
                    scaffold_g_save[selected_node]=[]
                    scaffold_g[selected_node]=[]
                scaffold_g_save[selected_node].append((new_edge, idx))
                scaffold_g[selected_node].append(( self.init_edge_state(scaffold_h, new_edge), idx))


        return scaffold_g_save, scaffold_h_save


        """ 
        isomers = enumerate_molecule(s1)
        selected_isomer, target = self.select_isomer(s1, latent_vector)
        total_loss4 = (selected_isomer-target).pow(2).sum()
        """

    
    def optimize(self, s1, s2, stochastic = False, lr = 0.01, max_iter = 100, beta1 = 0.01):
        g, h, scaffold_g, scaffold_h = make_graphs(s1, s2, extra_atom_feature=True, extra_bond_feature=True)
        self.embede_graph(g, h)
        self.embede_graph(scaffold_g, scaffold_h)
        self.encode(g, h)
        encoded_vector = self.cal_encoded_vector(h)
        latent_vector, mu, logvar = self.reparameterize(encoded_vector)
        start_point = create_var(encoded_vector.data, True) 
        self.init_scaffold_state(scaffold_g, scaffold_h)
        
        scaffold_state = average_node_state(scaffold_h)
        visited = []
        for iteration in range(max_iter):
            latent_vector, mu, logvar = self.reparameterize(start_point)
            prop = self.predict_property(torch.cat([latent_vector, scaffold_state], 1)).view(-1)
            loss1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            #objective = prop
            #objective = prop[0]-prop[1]-loss1*beta1
            objective = prop[0]-loss1*beta1
            objective.backward(retain_graph=True)
            #grad = torch.autograd.grad(prop, start_point)[0]
            start_point = start_point.data + lr * start_point.grad.data
            start_point = create_var(start_point, True)
            visited.append(start_point)

        retval = []
        for v in visited:
            latent_vector, mu, logvar = self.reparameterize(v)
            new_prop = self.predict_property(torch.cat([latent_vector, scaffold_state], 1)).squeeze().data.cpu().numpy()
            loss1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).data.cpu().numpy()[0]
            #objective = new_prop[0]-new_prop[1]-loss1*beta1
            objective = new_prop[0]-loss1*beta1
            g_gen, h_gen = self.sample(None, s2, latent_vector) 
            try:
                new_s = graph_to_smiles(g_gen, h_gen)
            except:
                new_s = None
            if new_s is None or new_s.find('.')!=-1:
                continue
            isomers = enumerate_molecule(new_s)
            selected_isomer, target = self.select_isomer(s1, latent_vector)
            new_s = isomers[np.argmax(selected_isomer.squeeze().data.cpu().numpy())]
            retval.append((new_s, objective, new_prop[0], loss1, latent_vector.data.cpu().numpy()[0]))
        return retval
    
    def embede_graph(self, g, h):
        for i in h:
            h[i] = self.original_node_embedding(h[i])
        
        for i in g:
            for j in range(len(g[i])):
                g[i][j] = (self.original_edge_embedding(g[i][j][0]), g[i][j][1])

    def encode(self, g, h, condition):
        for k in range(len(self.enc_U)):
            self.mpnn(g, h, self.enc_U[k], self.enc_C[k], condition)
    
    def mpnn(self, g, h, U, C, condition_vector=None):
        if len(g)==0:
            return
        #collect node and edge
        node_list1 = []
        node_list2 = []
        edge_list = []

        #make set of node vector to matrix
        hs = torch.cat([h[v] for v in g.keys()], 0)

        for v in g.keys():
            message = 0.0
            for i in range(len(g[v])):
                #index of connected node
                w = g[v][i][1]
                node_list1.append(h[v])
                node_list2.append(h[w])
                edge_list.append(g[v][i][0])

        node_list1 = torch.cat(node_list1, 0)
        node_list2 = torch.cat(node_list2, 0)
        edge_list = torch.cat(edge_list, 0)
        
        #calculate message
        if condition_vector is None:
            messages = F.relu(U(torch.cat([node_list1, node_list2, edge_list],-1)))
        else:
            ls = torch.cat([condition_vector for i in range(list(node_list1.size())[0])], 0)
            messages = F.relu(U(torch.cat([node_list1, node_list2, edge_list, ls],-1)))

        #summing messages
        index=0
        messages_summed = []
        for v in g.keys():
            message = 0.0
            i1 = index
            for i in range(len(g[v])):
                index+=1
            i2 = index
            messages_summed.append(messages[i1:i2].sum(0))
        messages_summed = torch.stack(messages_summed, 0)

        #update node state
        hs = C(messages_summed, hs)

        #split matrix fo node state to vector
        hs = torch.chunk(hs, len(g), 0)
        for idx,v in enumerate(g.keys()):
            h[v] = hs[idx]
    
    def add_node(self, g, h, latent_vector):
        #propagation
        for k in range(len(self.prop_add_node_U)):
            self.mpnn(g, h, self.prop_add_node_U[k], self.prop_add_node_C[k], latent_vector)
        #calculate graph vector
        graph_vector = self.cal_graph_vector(h)
        retval = torch.cat([graph_vector, latent_vector], -1)
        
        #FC layer
        retval = F.relu(self.add_node1(retval))
        retval = F.relu(self.add_node2(retval))
        retval = self.add_node3(retval)
        retval = F.softmax(retval, -1)
        return retval
    
    def add_edge(self, g, h, latent_vector):
        #propagation
        for k in range(len(self.prop_add_edge_U)):
            self.mpnn(g, h, self.prop_add_edge_U[k], self.prop_add_edge_C[k], latent_vector)
        
        #calculate graph vector
        graph_vector = self.cal_graph_vector(h)
        retval = torch.cat([graph_vector, latent_vector], -1)
        
        #FC layer
        retval = F.relu(self.add_edge1(retval))
        retval = F.relu(self.add_edge2(retval))
        retval = self.add_edge3(retval)
        retval = F.softmax(retval, -1)
        return retval
    
    def select_node(self, g, h, latent_vector):
        #propagation
        for k in range(len(self.prop_select_node_U)):
            self.mpnn(g, h, self.prop_select_node_U[k], self.prop_select_node_C[k], latent_vector)

        #collect node state
        vs = collect_node_state(h, except_last = True) 
        size = vs.size()        
        us = h[list(h.keys())[-1]].repeat(list(size)[0], 1)
        latent_vectors = latent_vector.repeat(list(size)[0], 1)
        retval = torch.cat([vs, us, latent_vectors], -1)
        
        #FC layer
        retval = F.relu(self.select_node1(retval))
        retval = F.relu(self.select_node2(retval))
        retval = self.select_node3(retval)
        retval = F.softmax(retval, 0)
        
        #print (h.size())
        #h = h.view(-1)
        return retval
    
    def select_isomer(self, mother, latent_vector):
        #sample possible isomer
        isomers = enumerate_molecule(mother)
        graph_vectors = []

        #make graph for each isomer
        for s in isomers:
            g, h = make_graph(s, extra_atom_feature=True, extra_bond_feature=True)
            self.embede_graph(g, h)
            for k in range(len(self.prop_select_isomer_U)):
                self.mpnn(g, h, self.prop_select_isomer_U[k], self.prop_select_isomer_C[k], latent_vector)
            graph_vectors.append(average_node_state(h))
        graph_vectors = torch.cat(graph_vectors, 0)
        latent_vectors = latent_vector.repeat(len(isomers), 1)
        retval = torch.cat([graph_vectors, latent_vectors], -1)
        
        #FC layer
        retval = F.relu(self.select_isomer1(retval))
        retval = F.relu(self.select_isomer2(retval))
        retval = self.select_isomer3(retval)
        retval = retval.view(-1)
        retval = F.softmax(retval, 0)
        target = []
        m = Chem.MolFromSmiles(mother)

        #check which isomer is same as mother
        for s in isomers:
            if m.HasSubstructMatch(Chem.MolFromSmiles(s),useChirality=True):
                target.append(1)
            else:
                target.append(0)
        target = create_var(torch.Tensor(target))
        
        return retval, target

    def predict_property(self, latent_vector):
        h = self.predict_property1(latent_vector)
        h = self.predict_property2(h)
        h = self.predict_property3(h)
        return h

    def cal_graph_vector(self, h):
        #h_sum = average_node_state(h)

        if len(h)==0:
            return create_var(torch.zeros(1,self.dim_of_graph_vector))
        inputs = torch.cat([h[i] for i in h.keys()], 0)
        h1 = self.cal_graph_vector1(inputs)
        h2 = F.sigmoid(self.cal_graph_vector2(inputs))
        retval = (h1*h2).mean(0, keepdim=True)
        #print (retval.size())
        return retval
    
    def cal_encoded_vector(self, h):
        #h_sum = average_node_state(h)

        if len(h)==0:
            return create_var(torch.zeros(1,dim_of_node_vector))
        inputs = torch.cat([h[i] for i in h.keys()], 0)
        h1 = self.cal_encoded_vector1(inputs)
        h2 = F.sigmoid(self.cal_encoded_vector2(inputs))
        retval = (h1*h2).mean(0, keepdim=True)
        #print (retval.size())
        return retval

    def init_node_state(self, h, atom_feature):        
        graph_vector = self.cal_graph_vector(h)
        return self.init_node_state1(torch.cat([graph_vector, self.node_embedding(atom_feature)], -1))
    
    def init_edge_state(self, h, edge_feature):
        graph_vector = self.cal_graph_vector(h)
        return self.init_edge_state1(torch.cat([graph_vector, self.edge_embedding(edge_feature)], -1))

    def init_scaffold_state(self, scaffold_g, scaffold_h):
        for k in range(len(self.init_scaffold_U)):
            self.mpnn(scaffold_g, scaffold_h, self.init_scaffold_U[k], self.init_scaffold_C[k])

    def reparameterize(self, latent_vector):
        mu = self.mean(latent_vector)
        logvar = self.logvar(latent_vector)
        std = torch.exp(0.5*logvar)
        eps = create_var(torch.randn(std.size()))

        return eps.mul(std).add_(mu), mu, logvar

