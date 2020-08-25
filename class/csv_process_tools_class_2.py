######################################
#conda init cmd.exe                  #
#close cmd                           #
#conda activate base                 #
######################################
import csv
#from rdkit import Chem


from collections import OrderedDict


def get_csv_train_data(csv_file_name,strength):
    """
    获取csv训练集数据
    """   
    with open(csv_file_name) as f:
        reader = csv.reader(f)
        rows = [row for row in  reader]
        names_id,smiles, labels, strengths= [],[],[],[]
        count_0 =0
        count_1 =0
        for row_count,row in enumerate(rows):
            if row_count>0:
                
                
                c1 = int(row[19])
                c2 = int(row[20])
            
                if c1>=strength and c2==0:
                    names_id.append( row[0])
                    smiles.append(row[8])
                    count_1 +=1
                    labels.append(1)
                    strengths.append(c1)
                    
                if c2>0:
                    names_id.append( row[0])
                    smiles.append(row[8])
                    count_0 +=1
                    labels.append(0)
                    strengths.append(0)
        print("---------------------------",strength)
        print("0 :",count_0,"   1 :",count_1)
        
        return names_id,smiles, labels,strengths

csv_file_name = "demo_lib002_pivot.csv"
names_id,smiles_list,labels,strengths = get_csv_train_data(csv_file_name,0)
smi_strength_info = OrderedDict(  list(zip(smiles_list,strengths)))     
#if __name__ == "__main__":

    #csv_file_name = "demo_lib002_pivot.csv"
    #names_id,smiles_list,labels,strengths = get_csv_train_data(csv_file_name,strength)
    #smi_strength_info = OrderedDict(  list(zip(smiles_list,strengths))   )
    #for i in range(1,24):
        #strength = i
        #names_id,smiles_list,labels,strengths = get_csv_train_data(csv_file_name,strength)
        #print("smiles_list",len(smiles_list))
        #print("labels",len(labels))
    #print("end")
