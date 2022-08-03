//+------------------------------------------------------------------+
//|                                                   NeuralNets.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum fx
  {
     RELU,
     SIGMOID,
     TANH
  } A_fx;

struct data
  {
    double sample_1;
    double sample_2;
    int    class_;
  };
  
class CNeuralNets
  {
   protected:
               double   e;
               bool     m_debug;
               bool     use_softmax;
                     
               double   Sigmoid(double z);
               double   tanh(double z);
               double   Relu(double z);
               void     SoftMax(double &Nodes[]);
               
               double   ActivationFx(double Q);
               void     Neuron(int HLnodes, double bias, double &Weights[], double &Inputs[], double &Outputs[]);
  
                
               
   public:
                CNeuralNets(fx ActivationFx, bool SoftMax=false);
               ~CNeuralNets(void);
               
                void     FeedForwardMLP(int HiddenLayers, double &MLPInputs[], double &MLPWeights[],double &bias[], double &MLPOutput[]);
               
                double   MathRandom(double mini, double maxi);
                int      MathRandInt(int mini, int maxi);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuralNets::CNeuralNets(fx ActivationFx,bool SoftMax=false)
 {
   e = 2.718281828;
   use_softmax = SoftMax;
   m_debug = true;
   A_fx = ActivationFx;
   
   printf("CNeural Nets Initialized activation = %s UseSoftMax = %s",EnumToString(ActivationFx),SoftMax?"Yes":"No");
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuralNets::~CNeuralNets(void)
 {
 
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
   ArrayCopy(TempArr,Nodes);  ArrayFree(Nodes);
   
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
void CNeuralNets::Neuron(int HLnodes,
                        double bias,
                        double &Weights[],
                        double &Inputs[],
                        double &Outputs[]
                       )
 {
   ArrayResize(Outputs,HLnodes);
   
   for (int i=0, w=0; i<HLnodes; i++)
    {
      if (m_debug) Print("\n HLNode ",i+1);
      double dot_prod = 0;
      for(int j=0; j<ArraySize(Inputs); j++, w++)
        {
            if (m_debug) printf("i %d  w %d = input %.5f x weight %.5f",i,w,Inputs[j],Weights[w]);
            dot_prod += Inputs[j]*Weights[w];
        }
      if (m_debug) printf("dot_Product %.5f + bias %.3f = %.5f",dot_prod,bias,dot_prod+bias);
      Outputs[i] = ActivationFx(dot_prod+bias);
      if (m_debug) printf("Activation function Output =%.5f",Outputs[i]);
    }     
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void   CNeuralNets::FeedForwardMLP(int HiddenLayers,
           double &MLPInputs[],
           double &MLPWeights[],
           double &bias[],
           double &MLPOutput[])
 {
    
    double L_weights[], L_inputs[], L_Out[];
    
    ArrayCopy(L_inputs,MLPInputs);
    
    int HLnodes = ArraySize(MLPInputs);
    int no_weights = HLnodes*ArraySize(L_inputs);
    int weight_start = 0;
    
    for (int i=0; i<HiddenLayers; i++)
      {
        
        if (m_debug) printf("<< Hidden Layer %d >>",i+1);
        ArrayCopy(L_weights,MLPWeights,0,weight_start,no_weights);

        Neuron(HLnodes,bias[i],L_weights,L_inputs,L_Out);
        
        ArrayCopy(L_inputs,L_Out);
        ArrayFree(L_Out);
        
        ArrayFree(L_weights);
        
        weight_start += no_weights;
      }
     
    if (use_softmax)  SoftMax(L_inputs);
    ArrayCopy(MLPOutput,L_inputs);
    if (m_debug)
      {
       Print("\nFinal MLP output(s)");
       ArrayPrint(MLPOutput,5);
      }
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
int CNeuralNets:: MathRandInt(int mini, int maxi)
  {
   double f   = (MathRand() / 32768.0);
   return mini + (int)(f * (maxi - mini));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+