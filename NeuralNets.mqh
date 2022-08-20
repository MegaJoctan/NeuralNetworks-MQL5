//+------------------------------------------------------------------+
//|                                                   NeuralNets.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+

#import "The_Matrix.ex5" //source code here >>> https://www.mql5.com/en/market/product/81533
   void MatrixMultiply(double &A[],double &B[],double &AxBMatrix[], int colsA,int rowsB,int &new_rows,int &new_cols);
   void CSVToMatrix(double &Matrix[],int &mat_rows,int &mat_cols,string csv_file,string sep=",");
   void MatrixPrint(double &Matrix[],int cols,int digits=5);
#import

bool m_debug = true;

//+------------------------------------------------------------------+

enum fx
  {
     RELU,
     SIGMOID,
     TANH
  } A_fx, last_AFx;


//+------------------------------------------------------------------+

class CNeuralNets
  {
   private:
                
               void     setmodelParams(double &w[],double &b[]);
               bool     WriteBin(double &w[],double &b[]); //store the model to binary files
               
   protected:
               double   e; 
               bool     use_softmax; 
               int      m_inputs;
               int      m_hiddenLayers;
               int      m_outputLayers;
               int      m_hiddenLayerNodes[];               
               
               double   Sigmoid(double z);
               double   tanh(double z);
               double   Relu(double z);
               void     SoftMax(double &Nodes[]);               
               double   ActivationFx(double Q);
               double   MathRandom(double mini, double maxi);
               
               
   public:
                CNeuralNets(fx HActivationFx,fx OActivationFx,int inputs,int &NodesHL[],int outputs=NULL, bool SoftMax=false);
               ~CNeuralNets(void);
               
                void     train_feedforwardMLP(double &XMatrix[],int epochs=1);
                void     FeedForwardMLP(double &MLPInputs[],double &MLPOutput[],double &Weights[],double &bias[]);
                
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuralNets::CNeuralNets(fx HActivationFx,fx OActivationFx,int inputs,int &NodesHL[],int outputs=NULL, bool SoftMax=false)
 {
   e = 2.718281828;
   use_softmax = SoftMax;
   
   A_fx = HActivationFx; 
   last_AFx = OActivationFx;
   
   m_inputs = inputs;
   m_hiddenLayers = ArraySize(NodesHL);
   ArrayCopy(m_hiddenLayerNodes,NodesHL);
   m_outputLayers = outputs;
      
   
   if (m_debug) printf("CNeural Nets Initialized Hidden Layer Activation = %s Output Activation %s UseSoftMax = %s",EnumToString(HActivationFx),EnumToString(OActivationFx),SoftMax?"Yes":"No");
   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuralNets::~CNeuralNets(void)
 {
    ArrayFree(m_hiddenLayerNodes); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CNeuralNets::Sigmoid(double z)
 { 
   return(1.0/(1.0+MathPow(e,-z)));
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CNeuralNets::tanh(double z)
 {
   return((MathPow(e,z) - MathPow(e,-z))/(MathPow(e,z) + MathPow(e,-z)));
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CNeuralNets::Relu(double z)
 {
   if (z < 0) return(0);
   else return(z);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuralNets::SoftMax(double &Nodes[])
 {
   double TempArr[];
   ArrayCopy(TempArr,Nodes); 
   
   double proba = 0, sum=0;
    
   for (int j=0; j<ArraySize(TempArr); j++)    sum += MathPow(e,TempArr[j]);
    
    for (int i=0; i<ArraySize(TempArr); i++)
      {
         proba = MathPow(e,TempArr[i])/sum;
         Nodes[i] = proba;
     } 
    ArrayFree(TempArr);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CNeuralNets::ActivationFx(double Q)
 {
   double out = 0;
   switch(A_fx)
     {
      case  SIGMOID:
        out = Sigmoid(Q);
        break;
      case TANH:
         out = (tanh(Q));
         break;
      case RELU:
         out = (Relu(Q));
         break;
      default:
         Print("Unknown Activation Function");
        break;
     }
   return(out);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuralNets::setmodelParams(double &w[],double &b[])
 {
    m_hiddenLayers = m_hiddenLayers+1;
    
    ArrayResize(m_hiddenLayerNodes,m_hiddenLayers);    
    m_hiddenLayerNodes[m_hiddenLayers-1] = m_outputLayers;
    
    ArrayResize(b,m_hiddenLayers);
    
    //Generate random bias
    for(int i=0; i<m_hiddenLayers; i++)         b[i] = MathRandom(0,1);
    
    if (m_debug)
     {
       Print("biases");
       ArrayPrint(b,4);
     }
     
    //generate weights 
    int sum_weights=0, L_inputs=m_inputs; 
    
    for (int i=0; i<m_hiddenLayers; i++)
      {
         sum_weights += L_inputs * m_hiddenLayerNodes[i];
         ArrayResize(w,sum_weights);
         L_inputs = m_hiddenLayerNodes[i];         
      }
    
    for (int j=0; j<sum_weights; j++) w[j] = MathRandom(0,1);
     
    if (m_debug)
      {       
        Print(" weights ",ArraySize(w));
        ArrayPrint(w,3);
      }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void   CNeuralNets::FeedForwardMLP(
                    double &MLPInputs[],
                    double &MLPOutput[],
                    double &Weights[],
                    double &bias[])
 { 
    if (m_outputLayers == NULL)
      {
        if (A_fx == RELU)     m_outputLayers = 1;
        else                  m_outputLayers = ArraySize(MLPInputs);
      }
   
   if (ArraySize(Weights) == 0 || ArraySize(bias) == 0) Print("There is an empty model weights or bias, Train the Model first before using it");
   
//---
    if (m_debug)
      {
        Print("Hidden layer nodes plus the output");
        ArrayPrint(m_hiddenLayerNodes);
      }
    
    int HLnodes = ArraySize(MLPInputs); 
    int weight_start = 0;
    
//---

    double L_weights[];
    
    int inputs=ArraySize(MLPInputs);
    int w_size = 0; //size of weights
    int cols = inputs, rows=1;
    
    double IxWMatrix[]; //dot product matrix 
    
    int start = 0; //starting to copy weights    
    for (int i=0; i<m_hiddenLayers; i++)
      {
          
          w_size = (inputs*m_hiddenLayerNodes[i]);
          ArrayResize(L_weights,w_size);
                    
            ArrayCopy(L_weights,Weights,0,start,w_size);
            
            start += w_size;
                            
              if (m_debug) {
                    printf("Hidden Layer %d | Nodes %d | Bias %.4f",i+1,m_hiddenLayerNodes[i],bias[i]);
                    
                    printf("Inputs %d Weights %d",ArraySize(MLPInputs),ArraySize(L_weights));
                    ArrayPrint(MLPInputs,5);  ArrayPrint(L_weights,3);
              }
              
              MatrixMultiply(MLPInputs,L_weights,IxWMatrix,cols,cols,rows,cols);
              
              //printf("before rows %d cols %d",rows,cols);
              
              
              if (m_debug) { Print("\nIxWMatrix"); MatrixPrint(IxWMatrix,cols); }
              
              ArrayFree(MLPInputs); ArrayResize(MLPInputs,m_hiddenLayerNodes[i]);
              inputs = ArraySize(MLPInputs);
              
              if (i == m_hiddenLayers-1) A_fx = last_AFx; //last layer Activation Function
              
              if (m_debug) Print("Activation Function ",EnumToString(A_fx));
              
              for(int k=0; k<ArraySize(IxWMatrix); k++) MLPInputs[k] = ActivationFx(IxWMatrix[k]+bias[i]); 
              
              if (m_debug) 
                {
                  Print("AFter Activation Function new inputs");
                  MatrixPrint(MLPInputs,cols);
                }
      }
      
     if (use_softmax) SoftMax(MLPInputs);     
     ArrayCopy(MLPOutput,MLPInputs);
     
     if (m_debug) { Print("MLP Final Output"); ArrayPrint(MLPOutput,3); }
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CNeuralNets:: MathRandom(double mini, double maxi)
  {
     double f   = (MathRand() / 32768.0);
     return mini + (f * (maxi - mini));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+         
void CNeuralNets::train_feedforwardMLP(double &XMatrix[],int epochs=1)
   {         
      double MLPInputs[]; ArrayResize(MLPInputs,m_inputs);
      double MLPOutputs[]; ArrayResize(MLPOutputs,m_outputLayers);
      
      double Weights[], bias[];
      
      setmodelParams(Weights,bias); //Generating random weights and bias
      
      for (int i=0; i<epochs; i++)
         {
           int start = 0;
           int rows = ArraySize(XMatrix)/m_inputs;
           
             for (int j=0; j<rows; j++) //iterate the entire dataset in a single epoch
               {
                 if (m_debug) printf("<<<< %d >>>",j+1);
                 ArrayCopy(MLPInputs,XMatrix,0,start,m_inputs);
             
                 FeedForwardMLP(MLPInputs,MLPOutputs,Weights,bias);
             
                 start+=m_inputs;
               }
         }
       
       WriteBin(Weights,bias);
   }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuralNets::WriteBin(double &w[], double &b[])
 {
      string file_name_w = NULL, file_name_b=  NULL;
      int handle_w, handle_b;
      
      file_name_w = MQLInfoString(MQL_PROGRAM_NAME)+"\\"+"model_w.bin";
      file_name_b =  MQLInfoString(MQL_PROGRAM_NAME)+"\\"+"model_b.bin"; 
      
      FileDelete(file_name_w); FileDelete(file_name_b);
      
       handle_w = FileOpen(file_name_w,FILE_WRITE|FILE_BIN);       
       if (handle_w == INVALID_HANDLE)   {  printf("Invalid %s Handle err %d",file_name_w,GetLastError());  }
       else                                 FileWriteArray(handle_w,w);
      
       FileClose(handle_w);    
       
       handle_b = FileOpen(file_name_b,FILE_WRITE|FILE_BIN);
       if (handle_b == INVALID_HANDLE)   {  printf("Invalid %s Handle err %d",file_name_b,GetLastError());  }
       else                                 FileWriteArray(handle_b,b);
     
       FileClose(handle_b);
     
     return(true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

