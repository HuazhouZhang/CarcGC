import os
import pandas as pd
import deepchem as dc
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, PyMol, rdFMCS
from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase
from deepchem import metrics
from IPython.display import Image, display
from rdkit.Chem.Draw import SimilarityMaps
import tensorflow as tf

def dataarrange():
	fdata = pd.read_csv('CarcGc/Fragment.csv',index_col=0,header=0)
	mdata = pd.read_csv('CarcGc/Molecule.csv',index_col=0,header=0)
	for chemf in fdata.index:
		chemfs = chemf.split('_')
		chem = chemfs[0]
		mc = mdata.loc[chem,'Molecule']
		fdata.loc[chemf,'Molecule'] = mc
	fdata.to_csv('CarcGc/Fragment_Molecule.csv',index=True,header=True)

def calweight():
	df = pd.read_csv('CarcGc/Fragment_Molecule.csv',index_col=0,header=0)
	df['Contrib']=df['Molecule']-df['Fragment']
	drug_smiles_file=r'Dataset/chemicals.smiles'
	chemdata = pd.read_csv(drug_smiles_file,index_col=0,header=None)
	chemdata.columns = ['SMILES']
	posidata = pd.read_csv('Dataset/Carcinogenicity_1043.csv',index_col=0,header=0)
	posidata = posidata[posidata['Carcinogenicity']=='+']
	chemlist = posidata.index.tolist()
	for i,chem in enumerate(chemlist):
		print(i)
		wt = {}
		smi = chemdata.loc[chem,'SMILES']
		mol = Chem.MolFromSmiles(smi)
		for n,atom in enumerate(Chem.rdmolfiles.CanonicalRankAtoms(mol)):
			indexx = chem+'_'+str(n)
			wt[atom] = df.loc[indexx,'Contrib']
		fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,wt,size=(280,280))
		fig.savefig('CarcGc/Molecule_Weight_figure/%s.png'%chem, bbox_inches='tight')

def main():
	dataarrange()
	calweight()
if __name__ == '__main__':
	main()