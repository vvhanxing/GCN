
from rdkit import Chem

from rdkit.Chem.Crippen import MolLogP

import numpy as np

import torch

import time

import matplotlib.pyplot as plt

import random
import os
import collections
import imageio
from csv_process_tools_class_2 import smi_strength_info

max_natoms = 80
epoch_num = 25
def get_data(label_name,smiles_name):
  with open(label_name) as f:

    labels = f.readlines()[:]

    labels = [s.strip() for s in labels]
  
    labels = [int(s) for s in labels[:] ]
  
    print(">",len(labels))

  with open(smiles_name) as f:

    smiles = f.readlines()[:]

    smiles = [s.strip() for s in smiles]

    data_zip = list(zip(smiles,labels))

    data_dict = collections.OrderedDict(data_zip)
  
    smiles = [s for s in smiles[:] if Chem.MolFromSmiles(s).GetNumAtoms()<80]
  
    print(">",len(smiles))

    print ('Number of smiles:', len(smiles))

    return smiles,labels,data_dict,data_zip

label_name,smiles_name ='label.txt','smiles.txt'
smiles ,labels ,data_dict,data_zip= get_data(label_name,smiles_name)
 


#---------------------------------
st = time.time()
#Y = []

#num_data = 20000



#for s in smiles[:num_data]:

#  m = Chem.MolFromSmiles(s)

#  logp = MolLogP(m)

#  Y.append(logp)

end = time.time()
#---------------------------------
print(len(smiles) ,len(labels))
#for key,value in data_dict.items():
    #print(key,value)
    #input()
  

data_T = [x for x in data_zip if x[1]==1]
data_F = [x for x in data_zip if x[1]==0]

random.seed(0)
for i in range(5):
    data_F.extend(data_T)

random.shuffle(data_F)

data_added = [(x[0],data_dict[x[0]]) for x in data_F]


#for count ,i in enumerate(data_added):
    #print(i)
    #input()

#input("finish")
#print(data_F[-10:])
#random.shuffle(data_F)
#print(data_F[:10])
#input()
#Y=labels
#for  i in data_added:
    #print(i)
    #input()
smiles,Y = list(zip(*data_added))
print(len(smiles),len(Y) )
print(smiles[:4])
print(Y[:4])
print (f'Time:{(end-st):.3f}')

#Dataset

from torch.utils.data import Dataset, DataLoader

from rdkit.Chem.rdmolops import GetAdjacencyMatrix

class MolDataset(Dataset):

    def __init__(self, smiles, properties, max_natoms, normalize_A=False):

      self.smiles = smiles

      self.properties = properties

      self.max_natoms = max_natoms

      

    def __len__(self):

        return len(self.smiles)

 

    def __getitem__(self, idx):

        s = self.smiles[idx]

 

        m = Chem.MolFromSmiles(s)

        natoms = m.GetNumAtoms()

 

        #adjacency matrix

        A = GetAdjacencyMatrix(m) + np.eye(natoms)

        A_padding = np.zeros((self.max_natoms, self.max_natoms))        

        A_padding[:natoms,:natoms] = A

        

        #atom feature

        X = [self.atom_feature(m,i) for i in range(natoms)]

        for i in range(natoms, max_natoms):

          X.append(np.zeros(28))

        X = np.array(X)

 

        sample = dict()

        sample['X'] = torch.from_numpy(X)

        sample['A'] = torch.from_numpy(A_padding)

        sample['Y'] = self.properties[idx]

 

        return sample

 

    def normalize_A(A):

      D = dfadfa(A)

      A = D*DD

      return A

 

    def one_of_k_encoding(self, x, allowable_set):

        if x not in allowable_set:

            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

        #print list((map(lambda s: x == s, allowable_set)))

        return list(map(lambda s: x == s, allowable_set))

 

    def one_of_k_encoding_unk(self, x, allowable_set):

        """Maps inputs not in the allowable set to the last element."""

        if x not in allowable_set:

            x = allowable_set[-1]

        return list(map(lambda s: x == s, allowable_set))

 

    def atom_feature(self, m, atom_i):

 

        atom = m.GetAtomWithIdx(atom_i)

        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),

                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +

                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +

                        self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +

                        self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +

                        [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28

#Model

import torch

import torch.nn as nn

import torch.nn.functional as F

 

class GConvRegressor(torch.nn.Module):

  def __init__(self, n_feature=128, n_layer = 10):

    super(GConvRegressor, self).__init__()

    self.W = nn.ModuleList([nn.Linear(n_feature, n_feature) for _ in range(n_layer)])

    self.embedding = nn.Linear(28, n_feature)

    self.fc = nn.Linear(n_feature, 2)

 

  def forward(self, x, A):

 

    x = self.embedding(x)

    for l in self.W:

      x = l(x)

      x = torch.einsum('ijk,ikl->ijl', (A.clone(), x))

      x = F.relu(x)

    x = x.mean(1)

 

    retval = self.fc(x)

 

    return retval



#Train model

import time

lr = 1e-4

model = GConvRegressor(128, 5) 

 

#model initialize

for param in model.parameters():

    if param.dim() == 1:

        continue

        nn.init.constant(param, 0)

    else:

        nn.init.xavier_normal_(param)

 

#Dataset 40387

#train_smiles = smiles[:40000]

#train_logp = Y[:40000]

#train_dataset = MolDataset(train_smiles, train_logp, max_natoms)


test_smiles = smiles[40000:]#

test_logp = Y[40000:]#

test_dataset = MolDataset(test_smiles, test_logp, max_natoms)



#label_name,smiles_name ='label_test.txt','smiles_test.txt'

#smiles ,Y ,_,_= get_data(label_name,smiles_name)

#test_smiles = smiles[:100]#

#test_logp = Y[:100]#

#print(test_smiles[:5])
#print(">>",test_logp)
#print("len(test_logp)",len(test_logp))

#test_dataset = MolDataset(test_smiles, test_logp, max_natoms)

 
 

#Dataloader

#train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=0)

test_dataloader = DataLoader(test_dataset, batch_size=387, num_workers=0)

 

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()

loss_list = []

def train():



  st = time.time()

 

  for epoch in range(epoch_num):

    epoch_loss = []

    for i_batch, batch in enumerate(train_dataloader):

      x, y, A = batch['X'] .float(), batch['Y'] .long(), batch['A'] .float()

      pred = model(x, A).squeeze(-1)

      loss = loss_fn(pred, y)

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

 

      optimizer.step()

      loss_list.append(loss.data.cpu().numpy())

      epoch_loss.append(loss.data.cpu().numpy())

 

    if True: print (epoch, np.mean(np.array(epoch_loss)))





    #Save model

    fn = 'save-'+str(epoch)+'.pt'

    torch.save(model.state_dict(), fn)


  #Load model

  model.load_state_dict(torch.load(fn))





  plt.plot(loss_list)

  plt.xlabel('Num iteration')

  plt.ylabel('Loss')

  plt.show()
  
  end = time.time()

  print ('Time:', end-st)
  
#train()
#----------------------------------------------------------------
#Test model
def test(model_name,fig_save_name):
    fn =model_name
    model.load_state_dict(torch.load(fn))
    model.eval()

    with torch.no_grad():

        y_pred_train, y_pred_test = [], []

        loss_train, loss_test = [], []

        pred_train, pred_test = [], []

        true_train, true_test = [], []


  #for batch in train_dataloader:

    #x, y, A = batch['X'] .float(), batch['Y'] .long(), batch['A'] .float()

    #pred = model(x, A).squeeze(-1)

    #pred_train.append(pred.data.cpu().numpy())

    #true_train.append(y.data.cpu().numpy())

    #loss_train.append(loss_fn(y, pred).data.cpu().numpy())

  

    for batch in test_dataloader:

        x, y, A = batch['X'] .float(), batch['Y'] .long(), batch['A'] .float()

        pred = model(x, A).squeeze(-1)

        pred_test.append(pred.data.cpu().numpy())

        true_test.append(y.data.cpu().numpy())

        #loss_test.append(loss_fn(y, pred).data.cpu().numpy())


    #print(len(pred_test))
    #print(true_test[0])

    print(pred_test[0])
    x_,y_ = list(zip(*list(pred_test[0])))
    x_t,y_t = [],[]
    x_f,y_f = [],[]
    #pred_train = np.concatenate(pred_train, -1)

    #pred_test = np.concatenate(pred_test, -1)

    #true_train = np.concatenate(true_train, -1)

    #true_test = np.concatenate(true_test, -1)

    c = []
    c_t = []
    c_f = []
    
    for count, smile in enumerate(test_smiles):
      
        #print(smi_strength_info[smile])
        c.append( smi_strength_info[smile] )
        
    

    print("---",len(c),len(true_test[0]))
    print(c)
    print(true_test[0])
    
    
    for count,i in  enumerate(true_test[0]):
        if i==1:
            x_t.append(x_[count])
            y_t.append(y_[count])
            c_t.append(c[count])
        else:
            x_f.append(x_[count])
            y_f.append(y_[count])
            c_f.append(c[count])
        
   

    #print ('Train loss:', np.mean(np.array(loss_train)))

    #print ('Test loss:', np.mean(np.array(loss_test)))


    #plt.scatter(true_train, pred_train, s=1)
    print(pred_test[0].max(axis = 0))
    x_max,y_max = list( pred_test[0].max(axis = 0))
    x_min,y_min = list( pred_test[0].min(axis = 0))
    
      
    theta = np.arange(x_min, x_max, 0.01)

    plt.plot(theta,theta)

    plt.scatter(x_t, y_t, marker='^',c = [-x for x in c_t]  )

    #c_c = 0
    #for x,y in zip(x_t, y_t):
      #plt.text(x,y+0.02,str(c_t[c_c]),ha='center', va='bottom', fontsize=5)
      #c_c+=1
      

    plt.scatter(x_f, y_f, marker='o',color="w",edgecolors =  [ (1.0,1.0,1-x*0.4-0.4) for x in  c_f  ] )
    
    #c_c = 0
    #for x,y in zip(x_f, y_f):
      #plt.text(x,y+0.02,str(c_f[c_c]),ha='center', va='bottom', fontsize=5)
      #c_c+=1
    

    #plt.plot([-8,12], [-8,12])

    #plt.xlabel('True')

    #plt.ylabel('Pred')
    
    #plt.show()
    plt.savefig(fig_save_name+".png", dpi=750)
    plt.clf()



def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return




folder = "model/"
fig_folder = "p/"
#model_name = 'save-24.pt'
#test(folder+model_name,"r")
#input("Finsh")

for model_name in os.listdir(folder):
    if ".pt" in  model_name:
        print(model_name)
        test(folder+model_name,folder+fig_folder+model_name)

image_list = os.listdir(folder+fig_folder)
create_gif([ folder+fig_folder+x for x in  image_list], '1.gif', duration=0.1)
