from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcTPSA

with open('../id_smiles.txt') as f, open('data.txt', 'w') as w:
    for l in f:
        m_id, s1, s2 = l.split()
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None : continue
        c1,c2 = CalcTPSA(m1), CalcTPSA(m2)
        w.write(m_id+'\t'+str(c1)+'\t'+str(c2)+'\n')
