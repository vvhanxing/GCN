
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
from sklearn import metrics
max_natoms = 80
epoch_num = 50
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
        #from plot_mol import  plot_mol_with_index(m)

 

        #adjacency matrix

        A = GetAdjacencyMatrix(m) + np.eye(natoms)
        #print(A)
        #print(A.shape)
########################################################
        #D = np.array(np.sum(A, axis=0))
        #print(D)
        #print(D.shape)
        
        #D = np.matrix(np.diag(D))
        #print(D.shape)
        
        #A = D**-1*A
        #print(A.shape)
        #input()
########################################################
        A_padding = np.zeros((self.max_natoms, self.max_natoms))        

        A_padding[:natoms,:natoms] = A

        A_padding = torch.from_numpy(A_padding)
        


        #atom feature

        X = [self.atom_feature(m,i) for i in range(natoms)]
        #print("X")
        #print(len(X))


        for i in range(natoms, max_natoms):

            X.append(np.zeros(28))

        X = np.array(X)



        #from   help_tools import get_mol_feature
        #print(X)
        #print(get_mol_feature(s).all()  ==X.all())





        sample = dict()

        sample['X'] = torch.from_numpy(X)

        sample['A'] = A_padding

        sample['Y'] = self.properties[idx]

        sample["smi"] = s

 

        return sample

 

    def normalize_A(A):

      #D = dfadfa(A)

      #A = D*DD
      

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

        return np.array(
                        self.one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +

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

    self.dropout = nn.Dropout2d(0.5)

    self.fc = nn.Linear(n_feature, 2)

 

  def forward(self, x, A):

 

    x = self.embedding(x)


    for l in self.W:

      x = l(x)

      x = torch.einsum('ijk,ikl->ijl', (A.clone(), x))

      x = F.relu(x)
    
    x = x.mean(1)
    #print(x)
    #x = torch.max(x,1)[0]

 
    #print(x.size())
    #input()
    x= self.dropout(x)
    
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

train_smiles = smiles[:40000]

train_y = Y[:40000]

train_dataset = MolDataset(train_smiles, train_y, max_natoms)

train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=0)

        
test_smiles = smiles[40000:]#

test_y = Y[40000:]#

test_dataset = MolDataset(test_smiles, test_y, max_natoms)



#label_name,smiles_name ='label_test.txt','smiles_test.txt'

#smiles ,Y ,_,_= get_data(label_name,smiles_name)

#test_smiles = smiles[:100]#

#test_y = Y[:100]#

#test_dataset = MolDataset(test_smiles, test_y, max_natoms)




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
  
train()
#----------------------------------------------------------------
#Test model
from sklearn.cluster import KMeans

def roc_plot(GTlist ,Problist ,show=False):


    fpr, tpr, thresholds = metrics.roc_curve(GTlist, Problist, pos_label=1)

    roc_auc = metrics.auc(fpr, tpr)  #auc为Roc曲线下的面积

 
    if show==True:
        

        plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)

        plt.legend(loc='lower right')

        plt.xlim([-0.1, 1.1])

        plt.ylim([-0.1, 1.1])

        plt.xlabel('False Positive Rate') #横坐标是fpr

        plt.ylabel('True Positive Rate')  #纵坐标是tpr

        plt.title('Receiver operating characteristic example')

        plt.show()
        
    return roc_auc


def k_mean(X,show=False):
    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    if show:
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.show()
    return y_pred
  
  
  
def test(model_name,fig_save_name):
    fn =model_name
    model.load_state_dict(torch.load(fn))
    model.eval()

    with torch.no_grad():

        y_pred_train, y_pred_test = [], []

        loss_train, loss_test = [], []

        pred_train, pred_test = [], []

        true_train, true_test = [], []

  

        for batch in test_dataloader:

            x, y, A, SMI = batch['X'] .float(), batch['Y'] .long(), batch['A'] .float(),batch['smi']

            pred = model(x, A).squeeze(-1)

            pred_test.append(pred.data.cpu().numpy())

            true_test.append(y.data.cpu().numpy())

        #loss_test.append(loss_fn(y, pred).data.cpu().numpy())

    #print(SMI[:10])
    #input()
    #print(len(pred_test))
    #print(true_test[0])

    #print(pred_test[0])input
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
    c_t_index = []
    c_f_index = []
    
    for count, smile in enumerate(test_smiles):
      
        #print(smi_strength_info[smile])
        c.append( smi_strength_info[smile] )
        
    

    print("---",len(c),len(true_test[0]))


   
    for count,i in  enumerate(true_test[0]):
        if i==1:
            x_t.append(x_[count])
            y_t.append(y_[count])
            c_t.append(c[count])
            c_t_index.append(count)
        else:
            x_f.append(x_[count])
            y_f.append(y_[count])
            c_f.append(c[count])
            c_f_index.append(count)
        
   

    #print ('Train loss:', np.mean(np.array(loss_train)))

    #print ('Test loss:', np.mean(np.array(loss_test)))


    #plt.scatter(true_train, pred_train, s=1)
    print(pred_test[0].max(axis = 0))
    x_max,y_max = list( pred_test[0].max(axis = 0))
    x_min,y_min = list( pred_test[0].min(axis = 0))
    
      
    theta = np.arange(x_min, x_max, 0.01)

    plt.plot(theta,theta)


    plt.scatter(x_f, y_f, marker='o',color="w",edgecolors =  [ (1.0,1.0,1-x*0.4-0.4) for x in  c_f  ] )


    plt.scatter(x_t, y_t, marker='^',c = [-x for x in c_t]  )

    c_c = 0
    for x,y in zip(x_t, y_t):
      #plt.text(x,y+0.02,str(c_t[c_c]),ha='center', va='bottom', fontsize=5)
      plt.text(x,y+0.02,str(c_t_index[c_c]),ha='center', va='bottom', fontsize=2)
      c_c+=1
    
    c_c = 0
    for x,y in zip(x_f, y_f):
      #plt.text(x,y+0.02,str(c_f[c_c]),ha='center', va='bottom', fontsize=5)
      plt.text(x,y+0.02,str(c_f_index[c_c]),ha='center', va='bottom', fontsize=2)
      c_c+=1
    
    #plt.plot([-8,12], [-8,12])
    plt.title(fig_save_name)

    #plt.xlabel('True')

    #plt.ylabel('Pred')
    
    #plt.show()
    plt.savefig(fig_save_name+".png",dpi = 800)
    plt.clf()
    #return SMI




def valid(model_name,fig_save_name):
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
            #print(A[0,:,:])
            #print(x)
            #print(x.shape)
            #input(A.shape)
            pred = model(x, A).squeeze(-1)

            pred_test.append(pred.data.cpu().numpy())

            true_test.append(y.data.cpu().numpy())

            #loss_test.append(loss_fn(y, pred).data.cpu().numpy())

    
    print(len(pred_test))
    #print(true_test[0])
    
    
    print('pred_test[0]',pred_test[0])
    print("y",y.numpy())
    k_mean_pred = k_mean(pred_test[0],False)
    print(k_mean_pred)

    ROCvalue = roc_plot(y.numpy(),np.array(k_mean_pred))
    if ROCvalue<0.5:
        ROCvalue = 1-ROCvalue
    print("ROC",ROCvalue)
    
    
    print()
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
    

        
    


    mark_t = []
    mark_f = []
    for count,i in  enumerate(true_test[0]):
        if i==1:
            x_t.append(x_[count])
            y_t.append(y_[count])
            mark_t.append(count)
            
        else:
            x_f.append(x_[count])
            y_f.append(y_[count])
            mark_f.append(count)
            
        
    print("-----",mark_t ,mark_f)

    #print ('Train loss:', np.mean(np.array(loss_train)))

    #print ('Test loss:', np.mean(np.array(loss_test)))

    #plt.scatter(true_train, pred_train, s=1)
    for i in true_test[0]:
        print(i)
    print(pred_test[0].max(axis = 0))
    x_max,y_max = list( pred_test[0].max(axis = 0))
    x_min,y_min = list( pred_test[0].min(axis = 0))
        
    theta = np.arange(x_min, x_max, 0.01)

    plt.plot(theta,theta)

    plt.scatter(x_t, y_t, marker='^')

    plt.scatter(x_f, y_f, marker='o')

    print("mark_t",mark_t)
    print("mark_f",mark_f)

    m =0
    for a,b in zip(x_t, y_t):
      
      plt.text(a,b+0.02,str(mark_t[m]),ha='center', va='bottom', fontsize=5)
      m+=1

    n=0
    for a,b in zip(x_f, y_f):
      
      plt.text(a,b+0.02,str(mark_f[n]),ha='center', va='bottom', fontsize=5)
      n+=1



    plt.title("   ROC  " +str(ROCvalue)[:6]+fig_save_name)

    plt.xlabel('True')

    plt.ylabel('Pred')
    #plt.show()

    x_t.extend(x_f)
    y_t.extend(y_f)
    
    X = np.array( list(zip(x_t,y_t)) )
    
    plt.savefig(fig_save_name+".png",dpi = 800)
    plt.clf()


def predict_heat_map():
  pass

def predict_smi(smi,model_name,save_fig_name):

    from  help_tools import get_mol_feature,get_mol_A_
    x = get_mol_feature(smi)
    A = get_mol_A_(smi)


    fn =model_name
    model.load_state_dict(torch.load(fn))
    model.eval()
    #print(model)
    parm = {}
    for name,parameters in model.named_parameters():
        #print(name)
        parm[name] = parameters.detach().numpy()

    #print(parm["W.0.weight"].shape,parm["W.0.bias"].shape,parm["embedding.weight"].shape,x.shape,parm["fc.weight"].shape)
    x = x.dot(parm["embedding.weight"].T)+parm["embedding.bias"]
    x = A.dot(x.dot(parm["W.0.weight"].T)+parm["W.0.bias"])
    x = (abs(x) + x) / 2
    x = A.dot(x.dot(parm["W.1.weight"].T)+parm["W.1.bias"])
    x = (abs(x) + x) / 2
    x = A.dot(x.dot(parm["W.2.weight"].T)+parm["W.2.bias"])
    x = (abs(x) + x) / 2
    x = A.dot(x.dot(parm["W.3.weight"].T)+parm["W.3.bias"])
    x = (abs(x) + x) / 2
    x = A.dot(x.dot(parm["W.4.weight"].T)+parm["W.4.bias"])
    x = (abs(x) + x) / 2

    #x = np.mean(x,0)

    x_f = x*parm["fc.weight"][0]+parm["fc.bias"][0] #80_128*128 + 1
    x_t = x*parm["fc.weight"][1]+parm["fc.bias"][1] #80_128*128 + 1
    x_f_with_w = np.sum(x_f,1) #80
    x_t_with_w = np.sum(x_t,1) #80
    x_with_w = x_t_with_w-x_f_with_w
    x_with_w_norm = (x_with_w-x_with_w.min(0))/(x_with_w.max(0)-x_with_w.min(0))
    #print("x_with_w_norm",x_with_w_norm)
    from help_tools import  plot_mol_with_color
    plot_mol_with_color(smi,x_with_w_norm,"heat_map/"+save_fig_name+".png")
    

    #x = x.dot(parm["fc.weight"].T)+parm["fc.bias"]
    #print(x.shape,x)
    
    with torch.no_grad():
        x = get_mol_feature(smi)
        x = torch.from_numpy(x).float().view(1,80,28)
        A = torch.from_numpy(A).float().view(1,80,80)
      
        score = model(x, A).squeeze(-1).numpy()
        #print(score)
        return score,np.where(score==np.max(score))[1]



        #true_test.append(y.data.cpu().numpy())

      
  
#predict_smi("CNC(=O)C1CN(c2nc(NCCOc3ccccc3)nc(Nc3cc(C)n[nH]3)n2)C1","model/save-39.pt","example.png")
csv_file_name = "demo_test_data.csv"


for batch in test_dataloader:

    x, y, A, SMI = batch['X'] .float(), batch['Y'] .long(), batch['A'] .float(),batch['smi']
    for c, s in enumerate(SMI):
        predict_smi(s,"model/save-49.pt",str(c)+".png")
input("finish")
#import csv
#with open(csv_file_name) as f:
    #reader = csv.reader(f)
    #rows = [row for row in  reader]
    #for row_count,row in enumerate(rows):
        
        #if row_count>0 and row_count!=36:
           #print(row_count,row[8],row[19],predict_smi(row[8],"model/save-49.pt",str(row_count-1)+".png"))
            
#input()    
#input()

def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return True


folder = "model/"
fig_folder = "p/"



for i in range(50):
    model_name = 'save-'+str( i)+'.pt'
    if ".pt" in  model_name:
        print(model_name)
        test(folder+model_name,fig_folder+model_name)

image_list = os.listdir(folder+fig_folder)
create_gif([ fig_folder+'save-'+str(x)+'.pt.png' for x in  range(40)], '2.gif', duration=0.2)
