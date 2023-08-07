#get drug features using Deepchem library
import os
import deepchem as dc
from rdkit import Chem
import numpy as np
import hickle as hkl
import random
import os

def get_chem_feat():
    drug_smiles_file=r'Dataset/chemicals.smiles'
    save_dir=r'CarcGC\drug_graph_feat'
    pubchemid2smile = {item.strip().split(',')[0]:item.split(',')[1].strip() for item in open(drug_smiles_file, encoding='utf-8').readlines()}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for each in pubchemid2smile.keys():
        print(pubchemid2smile[each])
        mol = Chem.MolFromSmiles(pubchemid2smile[each])
        featurizer = dc.feat.graph_features.ConvMolFeaturizer(use_chirality=True)
        mol_object = featurizer.featurize(mol)
        features = mol_object[0].atom_features
        degree_list = mol_object[0].deg_list
        adj_list = mol_object[0].canon_adj_list
        atoms_feature_all = []
        indexlist = [a for a in range(0,78) if a not in [9,10,11,12,14,17,18,19,20,21,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]]
        for atomarray in features:
            atoms_feature_all.append(atomarray[indexlist])
        features = np.array(atoms_feature_all)
        hkl.dump([features,adj_list,degree_list],save_dir+r'\%s.hkl'%each, mode='w')

def get_haps_feat():
    drug_smiles_file=r'Dataset/chemicals_haps.smiles'
    save_dir=r'CarcGC\drug_graph_feat_haps'
    pubchemid2smile = {item.strip().split(',')[0]:item.split(',')[1].strip() for item in open(drug_smiles_file, encoding='utf-8').readlines()}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for each in pubchemid2smile.keys():
        print(pubchemid2smile[each])
        mol = Chem.MolFromSmiles(pubchemid2smile[each])
        featurizer = dc.feat.graph_features.ConvMolFeaturizer(use_chirality=True)
        mol_object = featurizer.featurize(mol)
        features = mol_object[0].atom_features
        degree_list = mol_object[0].deg_list
        adj_list = mol_object[0].canon_adj_list
        atoms_feature_all = []
        indexlist = [a for a in range(0,78) if a not in [9,10,11,12,14,17,18,19,20,21,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]]
        for atomarray in features:
            atoms_feature_all.append(atomarray[indexlist])
        features = np.array(atoms_feature_all)
        hkl.dump([features,adj_list,degree_list],save_dir+r'\%s.hkl'%each, mode='w')

def get_eachmole_feat():
    drug_smiles_file=r'Dataset/chemicals.smiles'
    save_dir=r'CarcGC\drug_graph_feat_fragment'
    pubchemid2smile = {item.strip().split(',')[0]:item.split(',')[1].strip() for item in open(drug_smiles_file, encoding='utf-8').readlines()}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for each in pubchemid2smile.keys():
        print(pubchemid2smile[each])
        mol = Chem.MolFromSmiles(pubchemid2smile[each])
        featurizer = dc.feat.graph_features.ConvMolFeaturizer(use_chirality=True, per_atom_fragmentation=True)
        mol_object = featurizer.featurize(mol)
        featureslist = mol_object[0]
        for i,featuress in enumerate(featureslist):
            features = featuress.atom_features
            degree_list = featuress.deg_list
            adj_list = featuress.canon_adj_list
            atoms_feature_all = []
            indexlist = [a for a in range(0,78) if a not in [9,10,11,12,14,17,18,19,20,21,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]]
            for atomarray in features:
                atoms_feature_all.append(atomarray[indexlist])
            features = np.array(atoms_feature_all)
            hkl.dump([features,adj_list,degree_list],save_dir+r'\%s_%s.hkl'%(each,i), mode='w')

def main():
    get_chem_feat()
    get_haps_feat()
    get_eachmole_feat()
if __name__ == '__main__':
    main()