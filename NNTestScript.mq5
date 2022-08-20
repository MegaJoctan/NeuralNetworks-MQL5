//+------------------------------------------------------------------+
//|                                                 NNTestScript.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+

#include "NeuralNets.mqh";
CNeuralNets *neuralnet; 

//+------------------------------------------------------------------+
void OnStart()
  {
     
     int hlnodes[3] = {4,6,1};
     int outputs = 1;
     int inputs_=2;
    
     double XMatrix[]; int rows,cols;
     
     CSVToMatrix(XMatrix,rows,cols,"NASDAQ_DATA.csv");
     
     //MatrixPrint(XMatrix,cols,3);
     /*
     neuralnet = new CNeuralNets(SIGMOID,RELU,cols,hlnodes,outputs);     
     
     neuralnet.train_feedforwardMLP(XMatrix);
     
     delete(neuralnet);
     */
     
     double weights[], bias[];
     
     int handlew = FileOpen("NNTestScript\\model_w.bin",FILE_READ|FILE_BIN);
     FileReadArray(handlew,weights);
     FileClose(handlew);
     
     int handleb = FileOpen("NNTestScript\\model_b.bin",FILE_READ|FILE_BIN);
     FileReadArray(handleb,bias);
     FileClose(handleb);
     
     double Inputs[]; ArrayCopy(Inputs,XMatrix,0,0,cols); //copy the four first columns from this matrix
     double Output[];
     
     neuralnet = new CNeuralNets(RELU,RELU,cols,hlnodes,outputs);     
     
     neuralnet.FeedForwardMLP(Inputs,Output,weights,bias);
     
     Print("Outputs");
     ArrayPrint(Output);
     
     delete(neuralnet);
     
     
     
    
  }
//+------------------------------------------------------------------+
