from collections import OrderedDict
from operator import itemgetter
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import torch
from torch.autograd import Variable
import torch.nn as nn

#import deepchem as dc

def create_var(tensor, requires_grad=None): 
    if requires_grad is None: 
        #return Variable(tensor)
        return Variable(tensor)
    else: 
        return Variable(tensor,requires_grad=requires_grad)

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        #x = allowable_set[-2]
        return None
    return list(map(lambda s: int(x == s), allowable_set))

def atom_features(atom, include_extra = False):
    retval  = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [ 'C',
        'N',
        'O',
        'S',
        'F',
        'P',
        'Cl',
        'Br',
        'D'
      ])
    if include_extra:
        retval += [atom.GetDegree(),
            atom.GetFormalCharge()
           ]
    return retval

def bond_features(bond, include_extra = False):
    bt = bond.GetBondType()
    retval = [
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      0 # no bond
      #bond.GetIsConjugated(),
      #bond.IsInRing()
      ]
    if include_extra:
        bs = bond.GetStereo()
        retval += [bs == Chem.rdchem.BondStereo.STEREONONE,
                   bs == Chem.rdchem.BondStereo.STEREOANY,
                   bs == Chem.rdchem.BondStereo.STEREOZ,
                   bs == Chem.rdchem.BondStereo.STEREOE,
                   bs == Chem.rdchem.BondStereo.STEREOCIS,
                   bs == Chem.rdchem.BondStereo.STEREOTRANS
                  ]
    return np.array(retval)


def make_graph(smiles, extra_atom_feature = False, extra_bond_feature = False):
    g = OrderedDict({})
    h = OrderedDict({})
    if type(smiles) is str or type(smiles) is np.str_:
        molecule = Chem.MolFromSmiles(smiles)
    else:
        molecule = smiles
    chiral_list = Chem.FindMolChiralCenters(molecule)
    chiral_index = [c[0] for c in chiral_list]
    chiral_tag = [c[1] for c in chiral_list]
    #Chem.Kekulize(molecule)
    #Chem.Kekulize(molecule, clearAromaticFlags=False)
    for i in range(0, molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        if atom_i.GetSymbol() not in ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'D']:
            return None, None
        atom_i = atom_features(atom_i, extra_atom_feature)
        if extra_atom_feature:
            if i in chiral_index:
                if chiral_tag[chiral_index.index(i)]=='R':
                    atom_i += [0, 1, 0]
                else:
                    atom_i += [0, 0, 1]
            else:
                atom_i += [1, 0, 0]
        h[i] = create_var(torch.FloatTensor(atom_i), False).view(1, -1)
        for j in range(0, molecule.GetNumAtoms()):
            e_ij = molecule.GetBondBetweenAtoms(i, j)
            if e_ij != None:
                e_ij =   list(map(lambda x: 1 if x == True else 0, bond_features(e_ij, extra_bond_feature))) # ADDED edge feat
                e_ij = create_var(torch.FloatTensor(e_ij).view(1, -1), False)
                atom_j = molecule.GetAtomWithIdx(j)
                if i not in g:
                    g[i] = []
                g[i].append( (e_ij, j) )
    

    return g, h


def sum_node_state(h):   
    retval = create_var(torch.zeros(h[0].size()))
    for i in list(h.keys()):
        retval+=h[i]
    return retval

def average_node_state(h):   
    #retval = create_var(torch.zeros(1,9))
    retval = create_var(torch.zeros(h[0].size()))
    for i in list(h.keys()):
        retval+=h[i]
    if len(h)>0:
        retval=retval/len(h)
    return retval

def collect_node_state(h, except_last=False):   
    retval = []
    for i in list(h.keys())[:-1]:
        retval.append(h[i])
    if except_last==False:
        retval.append(h[list(h.keys())[-1]])

    return torch.cat(retval, 0)

def cal_formal_charge(atomic_symbol, bonds):
    if atomic_symbol=='N':
        if sorted(bonds, key=lambda x: (x[0], x[1])) == [('C', 1), ('O', 1), ('O', 2)]:
            return 1
        if sum(j for i, j in bonds)==4:
            return 1
    if atomic_symbol=='O':
        if sorted(bonds, key=lambda x: (x[0], x[1])) == [('N', 1)]:
            return -1
    return 0

def graph_to_smiles(g, h):
    atom_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'D']
    atomic_symbols = [None for i in range(len(h))]
    for i in h.keys():
        atomic_symbols[i] = atom_list[np.argmax(h[i].data.cpu().numpy())]

    BO = np.zeros((len(atomic_symbols), len(atomic_symbols)))
    for i in range(len(g)):
        for j in range(len(g[i])):
            BO[i,g[i][j][1]] = np.argmax(g[i][j][0].data.cpu().numpy())+1
    if not np.allclose(BO, BO.T, atol=1e-8):
        print ('BO is not symmetry')
        exit(-1)
    fc_list = []
    for i in range(len(atomic_symbols)):
        bond = [(atomic_symbols[j[1]], BO[i][j[1]]) for j in g[list(g.keys())[i]]]
        fc = cal_formal_charge(atomic_symbols[i], bond)
        if fc!=0:
            fc_list.append([i+1, fc])
    if len(fc_list)==0:
        fc_list = None
    smiles = BO_to_smiles(atomic_symbols, BO, fc_list)
    return smiles

def BO_to_smiles(atomic_symbols, BO, fc_list=None):
    natoms = len(atomic_symbols)
    nbonds = int(np.count_nonzero(BO)/2)
    w = open('GGM_tmp_sdf', 'w') 
    w.write('\n')
    w.write('     GGM\n')
    w.write('\n')
    w.write(str(natoms).rjust(3,' ')+str(nbonds).rjust(3, ' ') + '  0  0  0  0  0  0  0  0999 V2000\n')
    for s in atomic_symbols:
        w.write('    0.0000    0.0000    0.0000 '+s+'   0  0  0  0  0  0  0  0  0  0  0  0\n')
    for i in range (int(natoms)):
        for j in range(0,i):
            if BO[i,j]!=0:
                #if BO[i,j]==4:
                #    BO[i,j]=2
                w.write(str(i+1).rjust(3, ' ') + str(j+1).rjust(3, ' ') + str(int(BO[i,j])).rjust(3, ' ') + '0'.rjust(3, ' ') + '\n')
    if fc_list is not None:
        w.write('M  CHG  '+str(len(fc_list)))
        for fc in fc_list:
            w.write(str(fc[0]).rjust(4, ' ')+str(fc[1]).rjust(4, ' '))
        w.write('\n')
    w.write('M  END\n')

    w.write('$$$$')
    w.close()
    #m = Chem.SDMolSupplier('tmp.sdf')[0]

    os.system('babel -isdf GGM_tmp_sdf -osdf GGM_tmp_sdf 2> /dev/null')
    m = Chem.SDMolSupplier('GGM_tmp_sdf')[0]
    s = Chem.MolToSmiles(m)
    return s

def one_hot(tensor, depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,tensor.long())


def is_equal_node_type(h1, h2):
    if len(h1)!=len(h2):
        return False

    for i in h1:
        if not np.array_equal(h1[i].data.cpu().numpy(), h2[i].data.cpu().numpy()):
            return False
    return True

def is_equal_edge_type(g1, g2):
    if len(g1)!=len(g2):
        return False
    for i in g1:
        if len(g1[i])!=len(g2[i]):
            return False
        sorted1 = sorted(g1[i], key=itemgetter(1))
        sorted2 = sorted(g2[i], key=itemgetter(1))
        for j in range(len(sorted1)):
            if not np.array_equal(sorted1[j][0].data.cpu().numpy(), sorted2[j][0].data.cpu().numpy()):
                print (i, j, sorted1[j][0].data.cpu().numpy())
                print (i, j, sorted2[j][0].data.cpu().numpy())
                return False
            if sorted1[j][1]!=sorted2[j][1]:
                print (i, j, sorted1[j][1])
                print (i, j, sorted2[j][1])
                return False
    return True         


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            if param.grad is None:
                continue
            shared_param._grad = param.grad.cpu()

def probability_to_one_hot(tensor, stochastic = False):
    if stochastic:
        prob = tensor.data.cpu().numpy()[0].astype(np.float64)
        prob = prob/np.sum(prob)
        norm = np.sum(prob)
        prob = [prob[i]/norm for i in range(len(prob))]
        idx = int(np.random.choice(len(prob), 1, p = prob))
    else:
        idx = int(np.argmax(tensor.data.cpu().numpy()))
    return create_var(one_hot(torch.FloatTensor([idx]), list(tensor.size())[-1] ))

def make_graphs(s1, s2, extra_atom_feature = False, extra_bond_feature = False):
    molecule1 = Chem.MolFromSmiles(s1)
    molecule2 = Chem.MolFromSmiles(s2)
    #Chem.Kekulize(molecule1, clearAromaticFlags=False)
    #Chem.Kekulize(molecule2, clearAromaticFlags=False)
    g1, h1 = make_graph(Chem.Mol(molecule1), extra_atom_feature, extra_bond_feature)
    g2, h2 = make_graph(Chem.Mol(molecule2), extra_atom_feature, extra_bond_feature)
    if g1 is None or h1 is None or g2 is None or h2 is None:
        return None, None, None, None
    try:
        g1, h1 = index_rearrange(molecule1, molecule2, g1, h1)
    except:
        return None, None, None, None
    return g1, h1, g2, h2

def index_rearrange(molecule1, molecule2, g, h):
    
    #Chem.Kekulize(molecule1)
    #Chem.Kekulize(molecule2)


    scaffold_index = list(molecule1.GetSubstructMatches(molecule2)[0])
    
    new_index = OrderedDict({})
    for idx,i in enumerate(scaffold_index):
        new_index[i]=idx
    idx = len(scaffold_index)
    for i in range(len(h)):
        if i not in scaffold_index:
            new_index[i] = idx
            idx+=1
    g, h = index_change(g, h, new_index)
    
    return g, h
        
def index_change(g, h, new_index):
    new_h = OrderedDict({})
    new_g = OrderedDict({})
    for i in h.keys():
        new_h[new_index[i]]=h[i]
    for i in g.keys():
        new_g[new_index[i]]=[]
        for j in g[i]:
            new_g[new_index[i]].append((j[0], new_index[j[1]]))
    return new_g, new_h            

def enumerate_molecule(s):
    m = Chem.MolFromSmiles(s) 
    opts = StereoEnumerationOptions(unique=True, onlyUnassigned=False)
    #opts = StereoEnumerationOptions(tryEmbedding=True, unique=True, onlyUnassigned=False)
    isomers = tuple(EnumerateStereoisomers(m, options=opts)) 
    retval = []
    for smi in sorted(Chem.MolToSmiles(x,isomericSmiles=True) for x in isomers):  
        retval.append(smi)
    return retval

def initialize_model(model, load_save_file=False):
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal(param)

    return model    
