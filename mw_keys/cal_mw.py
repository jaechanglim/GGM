from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
with open('smiles_id_scaffold.txt') as f, open('data.txt', 'w') as w:
    lines = f.read().split('\n')[:-1]
    for l in lines[:]:
        l = l.split()
        s1, m_id, s2 = l[0], l[1], l[2]
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None : continue
        c1,c2 = ExactMolWt(m1), ExactMolWt(m2)
        w.write(m_id+'\t'+s1+'\t'+s2+'\t'+str(c1)+'\t'+str(c2)+'\n')
