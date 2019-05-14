import random
import numpy as np

from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F

import GGM.utils.util as util


class GGM(nn.Module):

    def __init__(self, args):
        """\
        Parameters
        ----------
        args: argparse.Namespace
            Delivers parameters from the arguments of `vaetrain.py`.
            Currently used attributes are:
                dim_of_node_vector
                dim_of_edge_vector
                dim_of_FC
                N_conditions
        """
        super(GGM, self).__init__()

        self.dim_of_node_vector = args.dim_of_node_vector
        self.dim_of_edge_vector = args.dim_of_edge_vector
        self.dim_of_FC = args.dim_of_FC
        self.N_conditions = args.N_conditions
        self.N_properties = args.N_properties
        self.N_atom_features = util.N_atom_features
        self.N_bond_features = util.N_bond_features
        self.N_extra_atom_features = util.N_extra_atom_features
        self.N_extra_bond_features = util.N_extra_bond_features
        self.dropout = args.dropout

        self.dim_of_graph_vector = 2 * self.dim_of_node_vector

        self.enc_U = \
            nn.ModuleList([nn.Linear(2 * self.dim_of_node_vector
                                     + self.dim_of_edge_vector
                                     + self.N_conditions,
                                     self.dim_of_node_vector)
                           for _ in range(3)])
        self.enc_C = \
            nn.ModuleList([nn.GRUCell(self.dim_of_node_vector,
                                      self.dim_of_node_vector)
                           for _ in range(3)])

        self.init_scaffold_U = \
            nn.ModuleList([nn.Linear(2 * self.dim_of_node_vector
                                     + self.dim_of_edge_vector
                                     + self.N_conditions,
                                     self.dim_of_node_vector)
                           for _ in range(3)])
        self.init_scaffold_C = \
            nn.ModuleList([nn.GRUCell(self.dim_of_node_vector,
                                      self.dim_of_node_vector)
                           for _ in range(3)])

        self.prop_add_node_U = \
            nn.ModuleList([nn.Linear(3 * self.dim_of_node_vector
                                     + self.dim_of_edge_vector
                                     + self.N_conditions,
                                     self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_add_node_C = \
            nn.ModuleList([nn.GRUCell(self.dim_of_node_vector,
                                      self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_add_node_FC = \
            nn.ModuleList([nn.Linear(self.dim_of_graph_vector
                                     + self.dim_of_node_vector
                                     + self.N_conditions,
                                     self.dim_of_FC),
                           nn.Linear(self.dim_of_FC, self.dim_of_FC),
                           nn.Linear(self.dim_of_FC, self.N_atom_features)])

        self.prop_add_edge_U = \
            nn.ModuleList([nn.Linear(3 * self.dim_of_node_vector
                                     + self.dim_of_edge_vector
                                     + self.N_conditions,
                                     self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_add_edge_C = \
            nn.ModuleList([nn.GRUCell(self.dim_of_node_vector,
                                      self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_add_edge_FC = \
            nn.ModuleList([nn.Linear(self.dim_of_graph_vector
                                     + self.dim_of_node_vector
                                     + self.N_conditions,
                                     self.dim_of_FC),
                           nn.Linear(self.dim_of_FC, self.dim_of_FC),
                           nn.Linear(self.dim_of_FC, self.N_bond_features)])

        self.prop_select_node_U = \
            nn.ModuleList([nn.Linear(3 * self.dim_of_node_vector
                                     + self.dim_of_edge_vector
                                     + self.N_conditions,
                                     self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_select_node_C = \
            nn.ModuleList([nn.GRUCell(self.dim_of_node_vector,
                                      self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_select_node_FC = \
            nn.ModuleList([nn.Linear(3 * self.dim_of_node_vector
                                     + self.N_conditions,
                                     self.dim_of_FC),
                           nn.Linear(self.dim_of_FC, self.dim_of_FC),
                           nn.Linear(self.dim_of_FC, 1)])

        self.prop_select_isomer_U = \
            nn.ModuleList([nn.Linear(3 * self.dim_of_node_vector
                                     + self.dim_of_edge_vector
                                     + self.N_conditions,
                                     self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_select_isomer_C = \
            nn.ModuleList([nn.GRUCell(self.dim_of_node_vector,
                                      self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_select_isomer_FC = \
            nn.ModuleList([nn.Linear(2 * self.dim_of_node_vector
                                     + self.N_conditions,
                                     self.dim_of_FC),
                           nn.Linear(self.dim_of_FC, self.dim_of_FC),
                           nn.Linear(self.dim_of_FC, 1)])

        self.prop_predict_U = \
            nn.ModuleList([nn.Linear(2 * self.dim_of_node_vector
                                     + self.dim_of_edge_vector,
                                     self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_predict_C = \
            nn.ModuleList([nn.GRUCell(self.dim_of_node_vector,
                                      self.dim_of_node_vector)
                           for _ in range(2)])
        self.prop_predict_FC = \
            nn.ModuleList([nn.Linear(self.dim_of_node_vector, self.dim_of_FC),
                           nn.Linear(self.dim_of_FC, self.N_properties)])

        self.predict_property_FC = \
            nn.ModuleList([nn.Linear(2 * self.dim_of_node_vector, 512),
                           nn.Linear(512, 512),
                           nn.Linear(512, 1)])

        self.cal_graph_vector_FC = \
            nn.Linear(self.dim_of_node_vector, self.dim_of_graph_vector)
        self.cal_graph_vector_coef_FC = \
            nn.Linear(self.dim_of_node_vector, self.dim_of_graph_vector)

        self.cal_encoded_vector_FC = \
            nn.Linear(self.dim_of_node_vector, self.dim_of_node_vector)
        self.cal_encoded_vector_coef_FC = \
            nn.Linear(self.dim_of_node_vector, self.dim_of_node_vector)

        self.init_node_state_FC = \
            nn.Linear(self.dim_of_graph_vector + self.dim_of_node_vector,
                      self.dim_of_node_vector)
        self.init_edge_state_FC = \
            nn.Linear(self.dim_of_graph_vector + self.dim_of_edge_vector,
                      self.dim_of_edge_vector)

        self.original_node_embedding_FC = \
            nn.Linear(self.N_atom_features + self.N_extra_atom_features,
                      self.dim_of_node_vector,
                      bias=False)
        self.original_edge_embedding_FC = \
            nn.Linear(self.N_bond_features + self.N_extra_bond_features,
                      self.dim_of_edge_vector,
                      bias=False)

        self.node_embedding_FC = \
            nn.Linear(self.N_atom_features, self.dim_of_node_vector,
                      bias=False)
        self.edge_embedding_FC = \
            nn.Linear(self.N_bond_features, self.dim_of_edge_vector,
                      bias=False)

        self.mean_FC = \
            nn.Linear(self.dim_of_node_vector, self.dim_of_node_vector)
        self.logvar_FC = \
            nn.Linear(self.dim_of_node_vector, self.dim_of_node_vector)

    def forward(self, s1, s2, condition, mask, shuffle=False):
        """\
        Parameters
        ----------
        s1: str
            A whole-molecule SMILES.
        s2: str
            A scaffold SMILES.
        condition1: list[float]
            [ whole_value1, whole_value2, ... ]
            Can be an empty list for unconditional training.
        condition2: list[float]
            [ scaffold_value1, scaffold_value2, ... ]
            Can be an empty list for unconditional training.

        Returns
        -------
        scaffold_g: OrderedDict[int, list[tuple[torch.autograd.Variable, int]]]
            Reconstructed dict of the latent edge vectors and partner-node indices.
        scaffold_h: OrderedDict[int, torch.autograd.Variable]
            Reconstructed dict of the latent node vectors.
        total_loss_rec: torch.autograd.Variable of shape (1,)
            Reconstruction loss.
        total_loss_vae: torch.autograd.Variable of shape (1,)
            VAE loss (weighted by `beta1`).
        total_loss_isomer: torch.autograd.Variable of shape (1,)
            Isomer selection loss.
        total_loss_predict: torch.Tensor of shape (1,)
            Condition prediction loss.
        """
        # Specification of graph variables defined here
        # ---------------------------------------------
        # g, g_save, scaffold_g, scaffold_g_save: edge dict objects
        #    -> OrderedDict[int, list[tuple[torch.autograd.Variable, int]]]
        #    -> { node_idx: [ (edge_vector, partner_node_idx), ... ], ... }
        #
        # h, h_save, scaffold_h, scaffold_h_save: node dict objects
        #    -> OrderedDict[int, torch.autograd.Variable]
        #    -> { node_idx: node_vector, ... }
        #
        # g_save, h_save:
        #     Backup of the whole-graph one-hots w/o extra features.
        #     These are not changed further.
        # g, h:
        #     become a latent vector for VAE.
        # scaffold_g_save, scaffold_h_save:
        #     The scaffold one-hots w/o extra features,
        #     to which new one-hots will be added
        #     to check later if the reconstruction is successful.
        # scaffold_g, scaffold_h:
        #     Scaffold dicts of latent edge/node vectors
        #     to which new initialized state vectors will be added.

        # Make graph of molecule and scaffold WITHOUT extra atom/bond features.
        g_save, h_save, scaffold_g_save, scaffold_h_save = \
            util.make_graphs(s1, s2)
        if g_save is None and h_save is None:
            return None

        # Make graph of molecule and scaffold WITH extra atom/bond features.
        g, h, scaffold_g, scaffold_h = \
            util.make_graphs(s1, s2, extra_atom_feature=True,
                             extra_bond_feature=True)
        g_predict, h_predict, scaffold_g_predict, scaffold_h_predict = \
            util.make_graphs(s1, s2, extra_atom_feature=True,
                             extra_bond_feature=True)

        # collect losses
        add_node_losses = []
        add_edge_losses = []
        select_node_losses = []

        # embed node state of graph
        self.embed_graph(g, h)
        self.embed_graph(scaffold_g, scaffold_h)

        self.embed_graph(g_predict, h_predict)
        self.embed_graph(scaffold_g_predict, scaffold_h_predict)

        # (N_conditions, )
        condition_truth = \
            util.create_var(torch.FloatTensor(condition))
        condition_whole = self.predict(g_predict, h_predict)
        condition_scaffold = \
            self.predict(scaffold_g_predict, scaffold_h_predict)
        condition_whole = torch.squeeze(condition_whole)
        condition_scaffold = torch.squeeze(condition_scaffold)
        condition = torch.cat((condition_whole, condition_scaffold))
        condition_masked = torch.mul(condition, mask)

        criteria_predict = nn.MSELoss()
        total_loss_predict = criteria_predict(condition_masked, condition_truth)

        # predicting loss of each property
        loss_property = []
        for i in range(self.N_properties):
            condition_prop = \
                torch.stack([condition_whole[i], condition_scaffold[i]], 0)
            condition_prop_truth = \
                torch.stack([condition_truth[i],
                             condition_truth[self.N_properties + i]], 0)
            loss = criteria_predict(condition_prop, condition_prop_truth)
            loss_property.append(loss.item())

        # (N_condition,) -> (1, N_conditions)
        condition = condition.view(1, -1)
        self.encode(g, h, condition)

        # make one vector representing graph using all node vectors
        encoded_vector = self.cal_encoded_vector(h)  # (1, dim_of_node_vector)

        # reparameterization trick. this routine is needed for VAE.
        latent_vector, mu, logvar_FC = self.reparameterize(encoded_vector)
        # -> (1, dim_of_node_vector), same, same
        if condition.shape:
            latent_vector = \
                torch.cat([latent_vector, condition], -1)
            # -> (1, dim_of_node_vector + N_conditions)

        # encode node state of scaffold graph
        self.init_scaffold_state(scaffold_g, scaffold_h, condition)

        # check which node is included in scaffold and which node is not
        leaves = [i for i in h_save.keys() if i not in scaffold_h.keys()]
        if shuffle: random.shuffle(leaves)

        for idx in leaves:
            # determine which node type should be added and calculate the loss
            new_node = self.add_node(scaffold_g, scaffold_h, latent_vector)
            # -> (1, N_atom_features)
            add_node_losses.append(
                (-h_save[idx] * torch.log(new_node + 1e-6)).sum())

            # add new node to the graph and initialize the new node state
            scaffold_h_save[idx] = h_save[idx]
            scaffold_h[idx] = self.init_node_state(scaffold_h,
                                                   scaffold_h_save[idx])

            # find the edges connected to the new node
            edge_list = [e for e in g_save[idx] if
                         e[1] in list(scaffold_h.keys())]
            if shuffle: random.shuffle(edge_list)

            for edge in edge_list:
                # determin which edge type is added and calculate the corresponding loss
                new_edge = self.add_edge(scaffold_g, scaffold_h, latent_vector)
                # -> (1, N_bond_features)
                add_edge_losses.append(
                    (-edge[0] * torch.log(new_edge + 1e-6)).sum())

                # determine which node is connected through selected edge and calculate the corresponding loss
                # The answer one-hot whose nonzero index is the partner-atom index:
                target = \
                    util.create_var(util.one_hot(
                        torch.FloatTensor([list(scaffold_h.keys()).index(edge[1])]),
                        len(scaffold_h) - 1))
                # -> (1, len(scaffold_h)-1)
                selected_node = self.select_node(scaffold_g, scaffold_h,
                                                 latent_vector).view(
                    target.size())
                # -> (1, len(scaffold_h)-1)
                select_node_losses.append(
                    (-target * torch.log(1e-6 + selected_node)).sum())

                # add edge to the graph and initialize the new node state
                if idx not in scaffold_g_save:
                    scaffold_g_save[idx] = []
                    scaffold_g[idx] = []
                scaffold_g_save[idx].append(edge)
                scaffold_g[idx].append(
                    (self.init_edge_state(scaffold_h, edge[0]), edge[1]))

                if edge[1] not in scaffold_g_save:
                    scaffold_g_save[edge[1]] = []
                    scaffold_g[edge[1]] = []
                scaffold_g_save[edge[1]].append((edge[0], idx))
                scaffold_g[edge[1]].append(
                    (self.init_edge_state(scaffold_h, edge[0]), idx))

            # the edge should not be added more. calculate the corresponding loss
            new_edge = self.add_edge(scaffold_g, scaffold_h, latent_vector)
            # Force the termination vector to be [0, 0, ..., 0, 1].
            end_add_edge = util.create_var(
                util.one_hot(torch.FloatTensor([self.N_bond_features - 1]),
                                   self.N_bond_features))
            add_edge_losses.append(
                (-end_add_edge * torch.log(1e-6 + new_edge)).sum())

        # the node should not be added more. calculate the corresponding loss
        new_node = self.add_node(scaffold_g, scaffold_h, latent_vector)
        # Force the termination vector to be [0, 0, ..., 0, 1].
        end_add_node = util.create_var(
            util.one_hot(torch.FloatTensor([self.N_atom_features - 1]),
                               self.N_atom_features))
        add_node_losses.append(
            (-end_add_node * torch.log(1e-6 + new_node)).sum())

        # convert list to the torch tensor
        total_add_node_loss = torch.stack(add_node_losses).mean()
        if len(add_edge_losses) > 0:
            total_add_edge_loss = torch.stack(add_edge_losses).mean()
            total_select_node_loss = torch.stack(select_node_losses).mean()
        else:
            total_add_edge_loss = 0.0
            total_select_node_loss = 0.0

        # check whether reconstructed graph is same as the input graph
        if not util.is_equal_node_type(scaffold_h_save, h_save):
            print('node miss match')
            print(s1)
            print(s2)
        if not util.is_equal_edge_type(scaffold_g_save, g_save):
            print('edge miss match')
            print(s1)
            print(s2)

        # reconstruction loss
        total_loss_rec = \
            total_add_node_loss + total_add_edge_loss + total_select_node_loss

        # VAE loss (AEVB 2013)
        total_loss_vae = -0.5 * torch.sum(1 + logvar_FC - mu.pow(2) - logvar_FC.exp())

        # select isomer
        isomers = util.enumerate_molecule(s1)  # ??
        selected_isomer, target = self.select_isomer(s1, latent_vector)

        # isomer loss
        total_loss_isomer = (selected_isomer - target).pow(2).sum()

        return scaffold_g, \
               scaffold_h, \
               total_loss_rec, \
               total_loss_vae, \
               total_loss_isomer, \
               total_loss_predict, \
               loss_property

    def sample(self, s1=None, s2=None, latent_vector=None, condition=None, stochastic=False):

        """\
        Parameters
        ----------
        s1: whole SMILES str
            If given, its graph becomes a latent vector to be decoded.
        s2: scaffold SMILES str
            Must be given other than None.
        latent_vector: None | torch.autograd.Variable
            A latent vector to be decoded.
            Not used if `s1` is given.
            If both `latent_vector` and `s1` are None,
            a latent vector is sampled from the standard normal.
        condition1: list[float] | None
            [ target_value1, target_value2, ... ]
            If None, target values are sampled from uniform [0, 1].
            Can be an empty list for unconditional sampling.
        condition2: list[float] | None
            [ scaffold_value1, scaffold_value2, ... ]
            If None, scaffold values are sampled from uniform [0, 1].
            Can be an empty list for unconditional sampling.
        stochastic: bool
            See `utils.probability_to_one_hot`.

        Returns
        -------
        scaffold_g_save: OrderedDict[int, list[tuple[torch.autograd.Variable, int]]]
            A new dict of edge one-hot vectors and partner-node indices
            generated from the given scaffold `s2`.
        scaffold_h_save: OrderedDict[int, torch.autograd.Variable]
            A new dict of node one-hot vectors generated from the given scaffold `s2`.
        """

        # Specification of graph variables defined here
        # ---------------------------------------------
        # g, g_save, scaffold_g, scaffold_g_save: edge dict objects
        #    -> OrderedDict[int, list[tuple[torch.autograd.Variable, int]]]
        #    -> { node_idx: [ (edge_vector, partner_node_idx), ... ], ... }
        #
        # h, h_save, scaffold_h, scaffold_h_save: node dict objects
        #    -> OrderedDict[int, torch.autograd.Variable]
        #    -> { node_idx: node_vector, ... }
        #
        # g_save, h_save:
        #     Backup of the whole-graph one-hots w/o extra features.
        #     These are not changed further.
        # g, h:
        #     become a latent vector for VAE.
        # scaffold_g_save, scaffold_h_save:
        #     The scaffold one-hots w/o extra features,
        #     to which new one-hots will be added
        #     to check later if the reconstruction is successful.
        # scaffold_g, scaffold_h:
        #     Scaffold dicts of latent edge/node vectors
        #     to which new initialized state vectors will be added.

        max_add_nodes = 100
        max_add_edges = 5

        if s2 in None:
            print('when you sample, you must give scaffold')
            return None

        # Embed the scaffold edge/node vectors.
        # If `s1` is given, convert its graph to a latent vector.
        if s1 is not None:
            g_save, h_save, scaffold_g_save, scaffold_h_save = util.make_graphs(s1, s2)
            if g_save is None and h_save is None:
                return None
            g, h, scaffold_g, scaffold_h = util.make_graphs(s1, s2, extra_atom_feature= True, extra_bond_feature= True)

            self.embed_graph(g, h)
            self.embed_graph(scaffold_g, scaffold_h)

            self.encode(g, h)
            encoded_vector = self.cal_encoded_vector(h)
            latent_vector, mu, logvar_FC = self.reparameterize(encoded_vector)
            # `mu` and `logvar` are not used further.

        # If `s1` is None, sample a latent vector from the standard normal.
        elif s1 is None:
            scaffold_g_save, scaffold_h_save = util.make_graph(s2)
            if scaffold_g_save is None and scaffold_h_save is None:
                return None
            scaffold_g, scaffold_h = util.make_graph(s2, extra_atom_feature= True, extra_bond_feature= True)

            self.embed_graph((scaffold_g, scaffold_h))
            if latent_vector is None:
                latent_vector = util.create_var(torch.randn(1, self.dim_of_node_vector))

        # Sample condition values if not given.
        if condition[:self.N_properties] is None or condition[self.N_properties:] is None:
            assert not self.N_properties
            whole_condition = np.random.rand(self.N_properties)
            scaffold_condition = np.random.rand(self.N_properties)
            condition = whole_condition + scaffold_condition

        # A condition torch.FloatTensor of shape (1, N_conditions):
        condition = util.create_var(torch.Tensor(condition))
        if condition.shape:
            condition = condition.unsqueeze(0)
        latent_vector = torch.cat([latent_vector, condition], -1)
        # -> (1, dim_of_node_vector + N_conditions)
        self.init_scaffold_state(scaffold_g, scaffold_h, condition)

        for _ in range(max_add_nodes):
            new_node = self.add_node(scaffold_g, scaffold_h, latent_vector)
            new_node = util.probability_to_one_hot(new_node, stochastic)

            # Recall our definition of the termination vector:
            if np.argmax(new_node.data.cpu().numpy().ravel()) == self.N_bond_features - 1:
                break

            idx = len(scaffold_h)
            scaffold_h_save[idx] = new_node
            scaffold_h[idx] = self.init_node_state_FC(scaffold_h, new_node)

            for _ in range(max_add_edges):
                new_edge = self.add_edge(scaffold_g, scaffold_h, latent_vector)  # (1, N_bond_features)
                new_edge = util.probability_to_one_hot(new_edge, stochastic)

                # Recall our definition of the termination vector:
                if np.argmax(new_edge.data.cpu().numpy().ravel()) == self.N_bond_features - 1:
                    break

                selected_node = self.select_node(scaffold_g, scaffold_h, latent_vector).view(1, -1)
                selected_node = self.select_node(scaffold_g,scaffold_h,latent_vector).view(1, -1)
                # -> (1, len(scaffold_h)-1)
                # Index of the selected node (int)
                selected_node = list(scaffold_h.keys())[
                    np.argmax(util.probability_to_one_hot(selected_node, stochastic).data.cpu().numpy().ravel())]
                if idx not in scaffold_g_save:
                    scaffold_g_save[idx] = []
                    scaffold_g[idx] = []
                scaffold_g_save[idx].append((new_edge, selected_node))
                scaffold_g[idx].append((self.init_edge_state(scaffold_h, new_edge), selected_node))

                # Add the same edge in the opposite direction.
                if selected_node not in scaffold_g_save:
                    scaffold_g_save[selected_node] = []
                    scaffold_g[selected_node] = []
                scaffold_g_save[selected_node].append((new_edge, idx))
                scaffold_g[selected_node].append((self.init_edge_state(scaffold_h, new_edge), idx))

        return scaffold_g_save, scaffold_h_save

    def embed_graph(self, g, h):
        """\
        Embede one-hot edge and node vectors.

        The resulting shapes are:
            vectors in g.values() -> (1, self.dim_of_edge_vector)
            vectors in h.values() -> (1, self.dim_of_node_vector)
        """
        for i in h:
            h[i] = self.original_node_embedding_FC(h[i])

        for i in g:
            for j in range(len(g[i])):
                g[i][j] = \
                    (self.original_edge_embedding_FC(g[i][j][0]), g[i][j][1])

    def encode(self, g, h, condition):
        for k in range(len(self.enc_U)):
            self.mpnn(g, h, self.enc_U[k], self.enc_C[k], condition)

    def predict(self, g, h):
        for k in range(len(self.prop_predict_U)):
            self.mpnn(g, h, self.prop_predict_U[k], self.prop_predict_C[k])
        encoded_vector = self.cal_encoded_vector(h)
        condition = \
            self.linear(encoded_vector, self.prop_predict_FC, act=nn.ReLU(),
                        dropout=self.dropout)
        return condition

    def predict_properties(self, smiles):
        mol = Chem.Mol(Chem.MolFromSmiles(smiles))
        g, h = util.make_graph(mol, extra_atom_feature=True,
                               extra_bond_feature=True)
        self.embed_graph(g, h)
        properties = self.predict(g, h)
        return properties

    def cal_graph_vector(self, h):
        """Return a graph-representation vector of shape (1, dim_of_graph_vector).
        See Eq. (4) of Yujia Li et al. 2018."""
        # h_sum = utils.average_node_state(h)
        if len(h) == 0:
            return util.create_var(torch.zeros(1, self.dim_of_graph_vector))
        inputs = torch.cat([h[i] for i in h.keys()], 0)
        h1 = self.cal_graph_vector_FC(inputs)  # cf. cal_encoded_vector
        h2 = torch.sigmoid(self.cal_graph_vector_coef_FC(inputs))  # cf.
        # cal_encoded_vector
        retval = (h1 * h2).mean(0, keepdim=True)
        # print (retval.size())
        return retval

    def cal_encoded_vector(self, h):
        """Return a graph-representation vector of shape (1, dim_of_node_vector)
                See Eq. (4) of Yujia Li et al. 2018."""
        if len(h) == 0:
            return util.create_var(torch.zeros(1, self.dim_of_node_vector))
        inputs = torch.cat([h[i] for i in h.keys()], 0)
        h1 = self.cal_encoded_vector_FC(inputs)  # cf. cal_graph_vector
        h2 = torch.sigmoid(self.cal_encoded_vector_coef_FC(inputs))  # cf.
        # cal_graph_vector
        retval = (h1 * h2).mean(0, keepdim=True)
        # print (retval.size())
        return retval

    def add_node(self, g, h, latent_vector):
        """Return a node vector of shape (1, N_atom_features)."""
        # propagation
        for k in range(len(self.prop_add_node_U)):
            self.mpnn(g, h, self.prop_add_node_U[k], self.prop_add_node_C[k],
                      latent_vector)
        # calculate graph vector
        graph_vector = self.cal_graph_vector(h)  # (1, dim_of_graph_vector)
        retval = torch.cat([graph_vector, latent_vector], -1)
        # -> (1, dim_of_graph_vector + dim_of_node_vector + N_conditions)
        # FC layer
        retval = self.linear(retval, self.prop_add_node_FC, nn.ReLU())
        retval = F.softmax(retval, -1)
        return retval

    def add_edge(self, g, h, latent_vector):
        """Return an edge vector of shape (1, N_bond_features)."""
        # propagation
        for k in range(len(self.prop_add_edge_U)):
            self.mpnn(g, h, self.prop_add_edge_U[k], self.prop_add_edge_C[k],
                      latent_vector)

        # calculate graph vector
        graph_vector = self.cal_graph_vector(h)  # (1, dim_of_graph_vector)
        retval = torch.cat([graph_vector, latent_vector], -1)
        # -> (1, dim_of_graph_vector + dim_of_node_vector + N_conditions)

        # FC layer
        retval = self.linear(retval, self.prop_add_edge_FC, nn.ReLU())
        retval = F.softmax(retval, -1)
        return retval

    def select_node(self, g, h, latent_vector):
        """\
        Return a node-selection vector of shape:
            (len(h)-1, self.select_node3.out_features) == (len(h)-1, 1)

        '-1' in 'len(h)-1' is due to the exclusion of the last node,
        which is to be a newly added one.
        """
        # propagation
        for k in range(len(self.prop_select_node_U)):
            self.mpnn(g, h, self.prop_select_node_U[k],
                      self.prop_select_node_C[k], latent_vector)

        # collect node state
        vs = util.collect_node_state(h, except_last=True)
        # (len(h)-1, dim_of_node_vector)
        size = vs.size()
        us = h[list(h.keys())[-1]].repeat(list(size)[0], 1)  # vs.shape
        latent_vectors = latent_vector.repeat(list(size)[0], 1)
        # -> (len(h)-1, dim_of_node_vector + N_conditions)
        retval = torch.cat([vs, us, latent_vectors], -1)
        # -> (len(h)-1, 3*dim_of_node_vector + N_conditions)

        # FC layer
        retval = self.linear(retval, self.prop_select_node_FC, nn.ReLU())
        retval = F.softmax(retval, 0)
        # print (h.size())
        # h = h.view(-1)
        return retval

    def select_isomer(self, mother, latent_vector):
        """\
        Return an isomer-selection vector and the answer one-hot.

        Returns
        -------
        retval: isomer-selection latent vector of shape (len(isomers),)
        target: answer one-hot of shape (len(isomers),)
            where `isomers` are the isomers of `mother`.
        """
        # sample possible isomer
        # list of isomer SMILESs
        isomers = util.enumerate_molecule(mother)
        graph_vectors = []

        # make graph for each isomer
        for s in isomers:
            g, h = \
                util.make_graph(s,
                                      extra_atom_feature=True,
                                      extra_bond_feature=True)
            self.embed_graph(g, h)
            for k in range(len(self.prop_select_isomer_U)):
                self.mpnn(g, h, self.prop_select_isomer_U[k],
                          self.prop_select_isomer_C[k], latent_vector)
            graph_vectors.append(util.average_node_state(h))
        graph_vectors = torch.cat(graph_vectors, 0)
        # -> (len(isomers), dim_of_node_vector)
        latent_vectors = latent_vector.repeat(len(isomers), 1)
        # -> (len(isomers), dim_of_node_vector + N_conditions)
        retval = torch.cat([graph_vectors, latent_vectors], -1)
        # -> (len(isomers), 2*dim_of_node_vector + N_conditions)

        # FC layer
        retval = self.linear(retval, self.prop_select_isomer_FC, nn.ReLU())
        retval = retval.view(-1)  # (len(isomers),)
        retval = F.softmax(retval, 0)
        target = []
        m = Chem.MolFromSmiles(mother)

        # check which isomer is same as mother
        for s in isomers:
            if m.HasSubstructMatch(Chem.MolFromSmiles(s), useChirality=True):
                target.append(1)
            else:
                target.append(0)
        target = util.create_var(torch.Tensor(target))  # (len(isomers),)

        return retval, target

    def init_node_state(self, h, atom_feature):
        """Return an initialized node-state vector of shape (1, dim_of_node_vector).
        See Eq. (11) of Yujia Li et al. 2018."""
        graph_vector = self.cal_graph_vector(h)
        return self.init_node_state_FC(
            torch.cat([graph_vector, self.node_embedding_FC(atom_feature)], -1))

    def init_edge_state(self, h, edge_feature):
        """Return an initialized edge-state vector of shape (1, dim_of_edge_vector).
        See Eq. (11) of Yujia Li et al. 2018."""
        graph_vector = self.cal_graph_vector(h)
        return self.init_edge_state_FC(
            torch.cat([graph_vector, self.edge_embedding_FC(edge_feature)], -1))

    def init_scaffold_state(self, scaffold_g, scaffold_h, condition):
        """Initial encoding of the node-state vectors of a scaffold graph"""
        for k in range(len(self.init_scaffold_U)):
            self.mpnn(scaffold_g,
                      scaffold_h,
                      self.init_scaffold_U[k],
                      self.init_scaffold_C[k],
                      condition)

    def reparameterize(self, latent_vector):
        mu = self.mean_FC(latent_vector)
        logvar_FC = self.logvar_FC(latent_vector)
        std = torch.exp(0.5 * logvar_FC)
        eps = util.create_var(torch.randn(std.size()))
        return eps.mul(std).add_(mu), mu, logvar_FC

    @staticmethod
    def mpnn(g, h, U, C, condition_vector=None):
        if len(g) == 0:
            return
        # collect node and edge
        node_list1 = []
        node_list2 = []
        edge_list = []

        # make set of node vectors to matrix
        hs = torch.cat([h[v] for v in g.keys()],
                       0)  # (len(h), dim_of_node_vector)

        for v in g.keys():
            message = 0.0  # ??
            for i in range(len(g[v])):
                # index of connected node
                w = g[v][i][1]
                node_list1.append(h[v])
                node_list2.append(h[w])
                edge_list.append(g[v][i][0])

        # Vectors to matrix
        # N_edges <- sum(len(edge) for edge in g.values())
        node_list1 = torch.cat(node_list1,
                               0)  # (N_edges, dim_of_node_vector)
        node_list2 = torch.cat(node_list2,
                               0)  # (N_edges, dim_of_node_vector)
        edge_list = torch.cat(edge_list, 0)  # (N_edges, dim_of_edge_vector)

        # calculate message
        if condition_vector is None or not condition_vector.shape:
            messages = F.relu(
                U(torch.cat([node_list1, node_list2, edge_list], -1)))
        else:
            ls = torch.cat(
                [condition_vector for i in
                 range(list(node_list1.size())[0])],
                0)
            messages = F.relu(
                U(torch.cat([node_list1, node_list2, edge_list, ls], -1)))
        # messages shape -> (N_edges, dim_of_node_vector)

        # summing messages
        index = 0
        messages_summed = []
        for v in g.keys():
            message = 0.0  # ??
            i1 = index
            for i in range(len(g[v])):
                index += 1
            i2 = index
            messages_summed.append(messages[i1:i2].sum(0))
        messages_summed = torch.stack(messages_summed,
                                      0)  # (len(h), dim_of_node_vector)

        # update node state
        hs = C(messages_summed, hs)  # (len(h), dim_of_node_vector)

        # split matrix of node states to vectors
        hs = torch.chunk(hs, len(g), dim=0)
        for idx, v in enumerate(g.keys()):
            h[v] = hs[idx]

    @staticmethod
    def linear(vec, FC, act=None, dropout=0):
        for i, layer in enumerate(FC):
            vec = layer(vec)
            if not (dropout == 0):
                dropout_layer = nn.Dropout(p=dropout)
                vec = dropout_layer(vec)
            if (act is not None) and (i != len(FC)-1):
                vec = act(vec)
        return vec










