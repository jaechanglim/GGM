from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

with open('../ChEMBL+STOCK1S/id_smiles_test.txt') as f, \
        open('../ChEMBL+STOCK1S/data_logp_test.txt', 'w') as w:
    for l in f:
        m_id, s1, s2 = l.split()
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None : continue
        c1,c2 = MolLogP(m1), MolLogP(m2)
        w.write(m_id+'\t'+str(c1)+'\t'+str(c2)+'\n')
