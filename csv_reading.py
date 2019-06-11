# importing csv module 
import csv 
import pandas as pd
import cv2
import tensorflow as tf
import numpy as np

# csv file name 
filename = r'MURA-v1.1/train_image_paths.csv'
i = 0
j = 0
k=11

# initializing the titles and rows list 
fields = [] 
rows = [] 
# reading csv file 
with open(filename, 'r') as csvfile: 
# creating a csv reader object 
    csvreader = csv.reader(csvfile) 

# extracting field names through first row 
    fields = next(csvreader) 

# extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 

# get total number of rows 
    print("Total no. of rows: %d"%(csvreader.line_num)) 

# printing the field names 
print('Field names are:' + ', '.join(field for field in fields)) 

# printing first 5 rows 
print('\nFirst 5 rows are:\n') 
for row in rows[:30000]: 
# parsing each column of a row 
    for col in row: 
       img = cv2.imread(col, 0)
       clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
       cl = clahe.apply(img)
       

       if (col.find('study1_positive') != -1): 
            path = r'positivev/'
        #cv2.imwrite(path+'image'+'.png',img)
            cv2.imwrite(path+'pic{:}.png'.format(i), cl)
            
            i = i+1
       elif (col.find('study2_positive') != -1): 
            path = r'positivev/'
        #cv2.imwrite(path+'image'+'.png',img)
            cv2.imwrite(path+'pic{:}.png'.format(i), cl)
            i = i+1    
       elif (col.find('study3_positive') != -1): 
            path = r'positivev/'
        #cv2.imwrite(path+'image'+'.png',img)
            cv2.imwrite(path+'pic{:}.png'.format(i), cl)
            i = i+1 
       elif (col.find('study4_positive') != -1): 
            path = r'positivev/'
        #cv2.imwrite(path+'image'+'.png',img)
            cv2.imwrite(path+'pic{:}.png'.format(i), cl)
            i = i+1 
       else:
            path = r'negativev/'
            cv2.imwrite(path+'pic{:}.png'.format(j), cl)
            j = j+1
       print(col) 
    
        

        
         
