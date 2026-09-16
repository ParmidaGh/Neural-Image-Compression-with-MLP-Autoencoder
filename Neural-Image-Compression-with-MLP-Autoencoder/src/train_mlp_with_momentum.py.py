import numpy as np
import cv2
import time
import os
import copy
import random
import math
import matplotlib.pyplot as plt
import time
import tensorflow as tf
with tf.device('/device:GPU:0'):

  def open_file(length):
      input_vector=[]
      block_num=256**2/length**2
      block_num=int(block_num)
      for i in range (91):
          addr='/content/drive/MyDrive/Q1/TrainSet/train ('+str(i)+').jpg'
          img=cv2.imread(addr,cv2.COLOR_BGR2GRAY)
          for j in range(block_num):
              vec=[]
              vec.append(1)
              start_x=(j//(256/length))*(length)
              start_x=int(start_x)
              start_y=(j%(256/length))*(length)
              start_y=int(start_y)
              for h in range(start_x,start_x+length):
                  for k in range(start_y,start_y+length):
                      a=float(img[h,k])
                      a/=255
                      vec.append(a)
              input_vector.append(vec)
      return(input_vector)

  def error_func(output,target,trainset_num):
      error=0
      for i in range(len(output)):
          if output[i]!=target[i]:
              error+=1
      error=error/trainset_num*100
      return error

  #تابع فعال سازی
  def sigmoid(y):
      return 1/(1+np.exp(-y))


  #مشتق تابع فعال سازی
  def sigmoid_deriv(y):
      return sigmoid(y)*(1-sigmoid(y))


  #آموزش شبکه عصبی
  def Train (input_vector,target_vector_list,w_start_range,w_end_range,learning_rate,neuron_num,epoch_limit):
      #تعیین مقادیر اولیه
      trainset_num=len(input_vector) #تعداد داده آموزش
      n=len(input_vector[0]) #تعداد نورون ورودی
      p=neuron_num #تعداد نورون لایه مخفی
      m=len(target_vector_list[0]) #تعداد نورون خروجی
      v=np.zeros((n,p)) #bias added before
      w=np.zeros((p+1,m)) #+1 for bias
      plot_x=[]
      plot_y=[]

      #مقدار دهی وزن های اولیه
      for i in range(n):
          for j in range(p):
              v[i,j]=random.uniform(w_start_range,w_end_range)

      for i in range(p+1):
          for j in range(m):
              w[i,j]=random.uniform(w_start_range,w_end_range)


      #Train part
      stop=False
      epoch=0

      start_time = time.time()
      while(stop==False and epoch<epoch_limit):

          #ساخت مقادیر ورودی و خروجی لایه مخفی و لایه خروجی
          not_updated=0 #time of weight not updated for each train_set
          output=[]
          train_data_num=0
          while (train_data_num < trainset_num and epoch<epoch_limit):
              update_flag=True
              x=input_vector[train_data_num]
              target_vector=target_vector_list[train_data_num]
              z_in_list=[]
              z_out_list=[]


              # feedforward
              for j in range(p):
                  z_in=0
                  for i in range(n):
                      z_in+=float(v[i,j])*x[i]
                  z_in_list.append(z_in)
                  z_out_list.append(sigmoid(z_in))
              z_out_list.insert(0,1) # for bias
              y_in_list=[]
              y_out_list=[]
              for k in range(m):
                  y_in=0
                  for j in range(p+1):
                      y_in+=float(w[j,k])*z_out_list[j]
                  y_in_list.append(y_in)
                  y_out_list.append(sigmoid(y_in))
              output.append(y_out_list) #ذخیره خروجی شبکه به ازای داده آموزش


              #backpropagation
              y_err=[]
              for k in range(m):
                  y_err.append((target_vector[k]-y_out_list[k])*sigmoid_deriv(y_in_list[k]))

              if y_out_list==target_vector:
                  not_updated+=1
                  update_flag=False

              delta_w=np.zeros((p+1,m))
              for k in range(m):
                  for j in range(p+1):
                      delta_w[j,k]=learning_rate*y_err[k]*z_out_list[j]


              z_in_err=[]
              for j in range(p):
                  sum=0
                  for k in range(m):
                      sum+=y_err[k]*float(w[j,k])
                  z_in_err.append(sum)

              z_err=[0]*p
              for j in range(p):
                  z_err[j]=z_in_err[j]*sigmoid_deriv(z_in_list[j])

              delta_v=np.zeros((n,p))
              for i in range(n):
                  for j in range(p):
                      delta_v[i,j]=learning_rate*z_err[j]*x[i]

              # momentum
              if update_flag==True:
                  #update weight
                  epoch+=1
                  delta_w_t1=delta_w
                  delta_v_t1=delta_v
                  w=w+delta_w+0.5*delta_w_t1
                  v=v+delta_v+0.5*delta_v_t1

              train_data_num += 1 # number of train data


          error=error_func(output,target_vector_list,trainset_num)
          plot_x.append(epoch)
          plot_y.append(error)
          if not_updated==trainset_num or error<=10:
              stop=True
      false_answer=0 #c is counter of false answer of network for train data
      for q in range(len(output)):
        if output[q]!=target_vector_list[q]:
          false_answer+=1

      end_time = time.time()
      convergence_time = end_time - start_time 
      print(f"Number of epochs: {epoch}")
      print(f"Error rate: {error/train_data_num}")
      print(f"Convergence Time: {convergence_time} seconds")

      print(plot_y[-1])
      plot_x=np.array(plot_x)
      plot_y=np.array(plot_y)
      plt.plot(plot_x,plot_y)
      plt.xlabel('epoch')
      plt.ylabel('error')
      result=[]
      result.append(v)
      result.append(w)
      return result


  def save_weights(v,w,hidden_layer_num):
      v=v.tolist()
      w=w.tolist()
      path='/content/drive/MyDrive/Q1/weights_for_'+str(hidden_layer_num)+'_momentum'
      os.mkdir(path)
      v_file=open(path+'/v_'+str(hidden_layer_num)+'_hidden neuron.txt','w')
      w_file=open(path+'/w_'+str(hidden_layer_num)+'_hidden neuron.txt','w')
      for i in range(len(v)):
          string=str(v[i])
          string=string[1:-2]
          if i<len(v)-1:
              string=string+'\n'
          v_file.writelines(string)
      v_file.close

      for i in range(len(w)):
          string=str(w[i])
          string=string[1:-2]
          if i<len(w)-1:
              string=string+'\n'
          w_file.writelines(string)
      w_file.close

  def main():
      learning_rate=0.5
      w_range = 0.1
      epoch_limit=500000
      p= 16 #تعداد نورون لایه مخفی

      input_vector=open_file(8)
      target_vector_list=copy.deepcopy(input_vector)
      for item in target_vector_list:
          item.pop(0)
      result=Train(input_vector,target_vector_list,-w_range,w_range,learning_rate,p,epoch_limit)
      n=len(input_vector[0]) #تعداد نورون ورودی
      m=len(target_vector_list[0]) #تعداد نورون خروجی
      v=np.zeros((n,p))
      w=np.zeros((p+1,m)) #+1 for bias
      v=result[0]
      w=result[1]

      #ذخیره وزن ها
      save_weights(v,w,p)
      print("end")

  main()