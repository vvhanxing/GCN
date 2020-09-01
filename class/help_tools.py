import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

max_natoms = 80
def get_mol_feature(smi):

    m = Chem.MolFromSmiles(smi)
    
    natoms = m.GetNumAtoms()    

    def one_of_k_encoding( x, allowable_set):

        if x not in allowable_set:

            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

        #print list((map(lambda s: x == s, allowable_set)))

        return list(map(lambda s: x == s, allowable_set))

 

    def one_of_k_encoding_unk( x, allowable_set):

        """Maps inputs not in the allowable set to the last element."""

        if x not in allowable_set:

            x = allowable_set[-1]

        return list(map(lambda s: x == s, allowable_set))

 

    def atom_feature( smi, atom_i):

        atom = m.GetAtomWithIdx(atom_i)


        return np.array(
                        one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +

                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +

                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +

                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +

                        [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28

    X = [atom_feature(m,i) for i in range(natoms)]

    for i in range(natoms, max_natoms):

        X.append(np.zeros(28))

    X = np.array(X)
                   
    return X



def get_mol_A_(s):
    
    m = Chem.MolFromSmiles(s)
    natoms = m.GetNumAtoms()
    A = GetAdjacencyMatrix(m) + np.eye(natoms)
    A_padding = np.zeros((max_natoms, max_natoms))        
    A_padding[:natoms,:natoms] = A
    
    return A_padding














def plot_mol_with_index(mol,save_name):
    '''
    '''
    atoms = mol.GetNumAtoms()
    for i in range( atoms ):
        mol.GetAtomWithIdx(i).SetProp(
            'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))
    draw = Draw.MolToImage(mol,size=(800, 800),)
    draw.show()      
    return mol







def plot_mol_with_color(smi,x_with_w_norm,save_name):
    mol = Chem.MolFromSmiles(smi)
    atoms = mol.GetNumAtoms()
    #for i in range( atoms ):
        #mol.GetAtomWithIdx(i).SetProp(
            #'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))

    d = rdMolDraw2D.MolDraw2DCairo(500, 500)

    atom_list = list(range(atoms))
    
    atom_cols = {}
    for index ,i in enumerate(list(x_with_w_norm)):
        atom_cols[index] = (0,0,i)
        

    rdMolDraw2D.PrepareAndDrawMolecule(d,
                                       mol,
                                       highlightAtoms=atom_list,
                                       highlightAtomColors=atom_cols,
                                       )
    
    with open(save_name, 'wb') as f:
        f.write(d.GetDrawingText())
    
      
    return mol










