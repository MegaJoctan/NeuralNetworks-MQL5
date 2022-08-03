//+------------------------------------------------------------------+
//|                                                   TestScript.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/job/new?prefered=omegajoctan"
#property description "hire me in a Machine learning project"
#property version   "1.00"
#property script_show_inputs
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "NeuralNets.mqh";
CNeuralNets *nn;

input int  batch_size =10;
input int  hidden_layers =2;
input bool use_softmax=false;

data data_blobs[];
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---     
     nn = new CNeuralNets(SIGMOID,use_softmax);
           
     MakeBlobs(batch_size);
     
     ArrayPrint(data_blobs);
       
     double Inputs[],OutPuts[];
     
     ArrayResize(Inputs,2);     ArrayResize(OutPuts,2);
     
     double weights[], biases[];
     generate_weights(weights,ArraySize(Inputs));
     generate_bias(biases);
     
     Print("weights"); ArrayPrint(weights);
     Print("biases"); ArrayPrint(biases);
     
     for (int i=0; i<batch_size; i++)
       {
         Print("Dataset Iteration ",i);
         Inputs[0] = data_blobs[i].sample_1; Inputs[1]= data_blobs[i].sample_2;    
         nn.FeedForwardMLP(hidden_layers,Inputs,weights,biases,OutPuts);
       }
       
     delete(nn);    
  }
//+------------------------------------------------------------------+
void MakeBlobs(int size=10)
 { 
     ArrayResize(data_blobs,size);
     for (int i=0; i<size; i++) 
       {   
         data_blobs[i].sample_1 = (i+1)*(2); 
         
         data_blobs[i].sample_2 = (i+1)*(5); 
         
         data_blobs[i].class_ = (int)round(nn.MathRandom(0,1));
       }  
 }
//+------------------------------------------------------------------+

void generate_weights(double &weights[],int inputs)
 {
   int size = (int)MathPow(inputs,2)*hidden_layers;
   ArrayResize(weights,size);
   
      for (int i=0; i<size; i++) weights[i] = nn.MathRandom(-1,1);
 }
//+------------------------------------------------------------------+
void generate_bias(double &bias[])
 {
   ArrayResize(bias,hidden_layers);
   for (int i=0; i<hidden_layers; i++) bias[i] = nn.MathRandom(-1,1);
 }
//+------------------------------------------------------------------+
