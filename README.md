## The codes for the work "Multiscale Subtraction Attention Network Based on Bayesian Loss for Medical Image Segmentation
 
 
### Prepare data
 
1、You can go to https://paperswithcode.com/dataset/kvasir to acquire the Kvasir-SEG dataset.
2、You can go to https://drive.google.com/file/d/1t3cyyTbA0mikL8L2rWRtREdWTLfmA3qL/view to acquire the BUSI dataset.

 
### Environment
 
Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.
 
###  Train/Test
 
Run the train script on the Kvasir-SEG and the BUSI dataset. The batch size we used is 4. If you do not have enough GPU memory, the bacth size can be reduced to 2 save memory. For more information, contact 1154692412@qq.com.


#### Train
##### python main_MSAByNet.py 

#### Test
##### python infer.py 
