from sklearn.svm import LinearSVC
import numpy as np

def my_fit( Z_train ):
  def binaryToDecimal(binary):
    decimal, i = 0, 0
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary//10
        i += 1
    return decimal
	
	#function to return which xorro p and q give
  which_XORRO = lambda np_array : binaryToDecimal(int(str(np_array[0])+str(np_array[1])+str(np_array[2])+str(np_array[3]))) 
	
  Z_train =Z_train.astype('int32')

	#defining models
  M=120
  model=[LinearSVC() for i in range(M)]
 
	#breaking the dataset
  x=Z_train[:,0:64]
  XORRO1=Z_train[:,[64,65,66,67]]
  XORRO2=Z_train[:,[68,69,70,71]]
  y=Z_train[:,72]

	#to store actual crp for each model
  x_data=[[] for i in range(0,120)]
  y_data=[[] for i in range(0,120)]

	#index of model
  ind=[[] for i in range(120)]
  k=0
  for i in range(0,16):
    for j in range(0,16):
      if(i<j):
        ind[k]=16*i + j
        k+=1

  for i in range(0,len(Z_train)):
    number=min(which_XORRO(XORRO1[i])*16+which_XORRO(XORRO2[i]),which_XORRO(XORRO1[i])+which_XORRO(XORRO2[i])*16)
    if (which_XORRO(XORRO1[i]) < which_XORRO(XORRO2[i])):
        y_data[ind.index(number)].append(1-y[i])
    else:
        y_data[ind.index(number)].append(y[i])
    x_data[ind.index(number)].append(x[i])

  for i in range(0,len(x_data)):
    x_data[i]=np.array(x_data[i])
    y_data[i]=np.array(y_data[i])

  x_data=np.array(x_data,dtype=object)
  y_data=np.array(y_data,dtype=object)
  #preprocessing

  #fitting
  for i in range(0,120):
    # print(y_data[i])
    if (x_data[i].ndim!=1):
      model[i].fit(x_data[i],y_data[i])

  return model					# Return the trained model



def my_predict( X_tst ,model):
  def binaryToDecimal(binary):
    decimal, i = 0, 0
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary//10
        i += 1
    return decimal
	
	#function to return which xorro p and q give
  which_XORRO = lambda np_array : binaryToDecimal(int(str(np_array[0])+str(np_array[1])+str(np_array[2])+str(np_array[3])))
	
  X_tst =X_tst.astype('int32')
 
	#breaking the dataset
  x=X_tst[:,[i for i in range(0,64)]]
  XORRO1=X_tst[:,[64,65,66,67]]
  XORRO2=X_tst[:,[68,69,70,71]]
  x_data=[[] for i in range(0,120)]
  y_predicted=[[] for i in range(0,120)]
  y1=[[]for i in range(0,len(X_tst))]
  k=np.zeros((120))

	#index of model
  ind=[[] for i in range(120)]
  k_=0
  for i in range(0,16):
    for j in range(0,16):
      if(i<j):
        ind[k_]=16*i + j
        k_+=1

  for i in range(0,len(X_tst)):
    number=min(which_XORRO(XORRO1[i])*16+which_XORRO(XORRO2[i]),which_XORRO(XORRO1[i])+which_XORRO(XORRO2[i])*16)
    x_data[ind.index(number)].append(x[i])
    y_predicted[ind.index(number)].append(0)

  for i in range(0,len(x_data)):
    x_data[i]=np.array(x_data[i])

  x_data=np.array(x_data,dtype=object)
  for i in range(0,120):
    if (x_data[i].ndim!=1):
      y_predicted[i]=model[i].predict(x_data[i])

  for i in range(0,len(X_tst)):
     number=min(which_XORRO(XORRO1[i])*16+which_XORRO(XORRO2[i]),which_XORRO(XORRO1[i])+which_XORRO(XORRO2[i])*16)
     if (k[ind.index(number)]<=len(y_predicted[ind.index(number)])):
      if (which_XORRO(XORRO1[i]) < which_XORRO(XORRO2[i])):
        y1[i]=1-y_predicted[ind.index(number)][int(k[ind.index(number)])]
      else:
         y1[i]=y_predicted[ind.index(number)][int(k[ind.index(number)])]
      k[ind.index(number)]+=1

  return y1