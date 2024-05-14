"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[1665],{5648:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>o,contentTitle:()=>l,default:()=>a,frontMatter:()=>d,metadata:()=>c,toc:()=>t});var i=r(4848),s=r(8453);const d={sidebar_label:"rnn_model",title:"vame.model.rnn_model"},l=void 0,c={id:"reference/vame/model/rnn_model",title:"vame.model.rnn_model",description:"Variational Animal Motion Embedding 0.1 Toolbox",source:"@site/docs/reference/vame/model/rnn_model.md",sourceDirName:"reference/vame/model",slug:"/reference/vame/model/rnn_model",permalink:"/VAME/docs/reference/vame/model/rnn_model",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"rnn_model",title:"vame.model.rnn_model"},sidebar:"docsSidebar",previous:{title:"evaluate",permalink:"/VAME/docs/reference/vame/model/evaluate"},next:{title:"rnn_vae",permalink:"/VAME/docs/reference/vame/model/rnn_vae"}},o={},t=[{value:"Encoder Objects",id:"encoder-objects",level:2},{value:"__init__",id:"__init__",level:4},{value:"forward",id:"forward",level:4},{value:"Lambda Objects",id:"lambda-objects",level:2},{value:"__init__",id:"__init__-1",level:4},{value:"forward",id:"forward-1",level:4},{value:"Decoder Objects",id:"decoder-objects",level:2},{value:"__init__",id:"__init__-2",level:4},{value:"forward",id:"forward-2",level:4},{value:"Decoder_Future Objects",id:"decoder_future-objects",level:2},{value:"__init__",id:"__init__-3",level:4},{value:"forward",id:"forward-3",level:4},{value:"RNN_VAE Objects",id:"rnn_vae-objects",level:2},{value:"__init__",id:"__init__-4",level:4},{value:"forward",id:"forward-4",level:4},{value:"Encoder_LEGACY Objects",id:"encoder_legacy-objects",level:2},{value:"__init__",id:"__init__-5",level:4},{value:"forward",id:"forward-5",level:4},{value:"Lambda_LEGACY Objects",id:"lambda_legacy-objects",level:2},{value:"__init__",id:"__init__-6",level:4},{value:"forward",id:"forward-6",level:4},{value:"Decoder_LEGACY Objects",id:"decoder_legacy-objects",level:2},{value:"__init__",id:"__init__-7",level:4},{value:"forward",id:"forward-7",level:4},{value:"Decoder_Future_LEGACY Objects",id:"decoder_future_legacy-objects",level:2},{value:"__init__",id:"__init__-8",level:4},{value:"forward",id:"forward-8",level:4},{value:"RNN_VAE_LEGACY Objects",id:"rnn_vae_legacy-objects",level:2},{value:"__init__",id:"__init__-9",level:4},{value:"forward",id:"forward-9",level:4}];function h(e){const n={a:"a",code:"code",em:"em",h2:"h2",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.p,{children:"Variational Animal Motion Embedding 0.1 Toolbox\n\xa9 K. Luxem & P. Bauer, Department of Cellular Neuroscience\nLeibniz Institute for Neurobiology, Magdeburg, Germany"}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.a,{href:"https://github.com/LINCellularNeuroscience/VAME",children:"https://github.com/LINCellularNeuroscience/VAME"}),"\nLicensed under GNU General Public License v3.0"]}),"\n",(0,i.jsxs)(n.p,{children:["The Model is partially adapted from the Timeseries Clustering repository developed by Tejas Lodaya:\n",(0,i.jsx)(n.a,{href:"https://github.com/tejaslodaya/timeseries-clustering-vae/blob/master/vrae/vrae.py",children:"https://github.com/tejaslodaya/timeseries-clustering-vae/blob/master/vrae/vrae.py"})]}),"\n",(0,i.jsx)(n.h2,{id:"encoder-objects",children:"Encoder Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class Encoder(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Encoder module of the Variational Autoencoder."}),"\n",(0,i.jsx)(n.h4,{id:"__init__",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(NUM_FEATURES: int, hidden_size_layer_1: int,\n             hidden_size_layer_2: int, dropout_encoder: float)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Initialize the Encoder module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"NUM_FEATURES"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of input features."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_1"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the first hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_2"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the second hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"dropout_encoder"})," ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for regularization."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(inputs: torch.Tensor) -> torch.Tensor\n"})}),"\n",(0,i.jsx)(n.p,{children:"Forward pass of the Encoder module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"inputs"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Input tensor of shape (batch_size, sequence_length, num_features)."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"torch.Tensor"})," - Encoded representation tensor of shape (batch_size, hidden_size_layer_1 * 4)."]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"lambda-objects",children:"Lambda Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class Lambda(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Lambda module for computing the latent space parameters."}),"\n",(0,i.jsx)(n.h4,{id:"__init__-1",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(ZDIMS: int, hidden_size_layer_1: int, hidden_size_layer_2: int,\n             softplus: bool)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Initialize the Lambda module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the latent space."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_1"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the first hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_2"})," ",(0,i.jsx)(n.em,{children:"int, deprecated"})," - Size of the second hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"softplus"})," ",(0,i.jsx)(n.em,{children:"bool"})," - Whether to use softplus activation for logvar."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward-1",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(\n        hidden: torch.Tensor\n) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]\n"})}),"\n",(0,i.jsx)(n.p,{children:"Forward pass of the Lambda module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Hidden representation tensor of shape (batch_size, hidden_size_layer_1 * 4)."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsx)(n.p,{children:"tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Latent space tensor, mean tensor, logvar tensor."}),"\n",(0,i.jsx)(n.h2,{id:"decoder-objects",children:"Decoder Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class Decoder(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Decoder module of the Variational Autoencoder."}),"\n",(0,i.jsx)(n.h4,{id:"__init__-2",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int,\n             hidden_size_rec: int, dropout_rec: float)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Initialize the Decoder module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"TEMPORAL_WINDOW"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the temporal window."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the latent space."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"NUM_FEATURES"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of input features."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_rec"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the recurrent hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"dropout_rec"})," ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for regularization."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward-2",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(inputs: torch.Tensor, z: torch.Tensor) -> torch.Tensor\n"})}),"\n",(0,i.jsx)(n.p,{children:"Forward pass of the Decoder module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"inputs"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Input tensor of shape (batch_size, seq_len, ZDIMS)."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"z"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Latent space tensor of shape (batch_size, ZDIMS)."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"torch.Tensor"})," - Decoded output tensor of shape (batch_size, seq_len, NUM_FEATURES)."]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"decoder_future-objects",children:"Decoder_Future Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class Decoder_Future(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Decoder module for predicting future sequences."}),"\n",(0,i.jsx)(n.h4,{id:"__init__-3",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int,\n             FUTURE_STEPS: int, hidden_size_pred: int, dropout_pred: float)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Initialize the Decoder_Future module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"TEMPORAL_WINDOW"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the temporal window."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the latent space."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"NUM_FEATURES"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of input features."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"FUTURE_STEPS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of future steps to predict."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_pred"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the prediction hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"dropout_pred"})," ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for regularization."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward-3",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(inputs: torch.Tensor, z: torch.Tensor) -> torch.Tensor\n"})}),"\n",(0,i.jsx)(n.p,{children:"Forward pass of the Decoder_Future module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"inputs"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Input tensor of shape (batch_size, seq_len, ZDIMS)."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"z"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Latent space tensor of shape (batch_size, ZDIMS)."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"torch.Tensor"})," - Predicted future tensor of shape (batch_size, FUTURE_STEPS, NUM_FEATURES)."]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"rnn_vae-objects",children:"RNN_VAE Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class RNN_VAE(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Variational Autoencoder module."}),"\n",(0,i.jsx)(n.h4,{id:"__init__-4",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int,\n             FUTURE_DECODER: bool, FUTURE_STEPS: int, hidden_size_layer_1: int,\n             hidden_size_layer_2: int, hidden_size_rec: int,\n             hidden_size_pred: int, dropout_encoder: float, dropout_rec: float,\n             dropout_pred: float, softplus: bool)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Initialize the VAE module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"TEMPORAL_WINDOW"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the temporal window."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the latent space."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"NUM_FEATURES"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of input features."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"FUTURE_DECODER"})," ",(0,i.jsx)(n.em,{children:"bool"})," - Whether to include a future decoder."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"FUTURE_STEPS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of future steps to predict."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_1"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the first hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_2"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the second hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_rec"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the recurrent hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_pred"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the prediction hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"dropout_encoder"})," ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for encoder."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward-4",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(seq: torch.Tensor) -> tuple\n"})}),"\n",(0,i.jsx)(n.p,{children:"Forward pass of the VAE."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"seq"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Input sequence tensor of shape (batch_size, seq_len, NUM_FEATURES)."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsx)(n.p,{children:"Tuple containing:"}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsx)(n.li,{children:"If FUTURE_DECODER is True:"}),"\n",(0,i.jsx)(n.li,{children:"prediction (torch.Tensor): Reconstructed input sequence tensor."}),"\n",(0,i.jsx)(n.li,{children:"future (torch.Tensor): Predicted future sequence tensor."}),"\n",(0,i.jsx)(n.li,{children:"z (torch.Tensor): Latent representation tensor."}),"\n",(0,i.jsx)(n.li,{children:"mu (torch.Tensor): Mean of the latent distribution tensor."}),"\n",(0,i.jsx)(n.li,{children:"logvar (torch.Tensor): Log variance of the latent distribution tensor."}),"\n",(0,i.jsx)(n.li,{children:"If FUTURE_DECODER is False:"}),"\n",(0,i.jsx)(n.li,{children:"prediction (torch.Tensor): Reconstructed input sequence tensor."}),"\n",(0,i.jsx)(n.li,{children:"z (torch.Tensor): Latent representation tensor."}),"\n",(0,i.jsx)(n.li,{children:"mu (torch.Tensor): Mean of the latent distribution tensor."}),"\n",(0,i.jsx)(n.li,{children:"logvar (torch.Tensor): Log variance of the latent distribution tensor."}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"encoder_legacy-objects",children:"Encoder_LEGACY Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class Encoder_LEGACY(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"LEGACY Encoder module of the Variational Autoencoder."}),"\n",(0,i.jsx)(n.h4,{id:"__init__-5",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(NUM_FEATURES: int, hidden_size_layer_1: int,\n             hidden_size_layer_2: int, dropout_encoder: float)\n"})}),"\n",(0,i.jsx)(n.p,{children:"(LEGACY) Initialize the Encoder_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"NUM_FEATURES"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of input features."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_1"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the first hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_2"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the second hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"dropout_encoder"})," ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for the encoder."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward-5",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(inputs: torch.Tensor) -> torch.Tensor\n"})}),"\n",(0,i.jsx)(n.p,{children:"(LEGACY) Forward pass of the Encoder_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"inputs"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Input tensor of shape (batch_size, seq_len, NUM_FEATURES)."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"torch.Tensor"})," - Encoded tensor."]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"lambda_legacy-objects",children:"Lambda_LEGACY Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class Lambda_LEGACY(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"LEGACY Lambda module for computing the latent space parameters."}),"\n",(0,i.jsx)(n.h4,{id:"__init__-6",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(ZDIMS: int, hidden_size_layer_1: int, hidden_size_layer_2: int)\n"})}),"\n",(0,i.jsx)(n.p,{children:"(LEGACY) Initialize the Lambda_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the latent space."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_1"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the first hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_2"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the second hidden layer."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward-6",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(\n    cell_output: torch.Tensor\n) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]\n"})}),"\n",(0,i.jsx)(n.p,{children:"(LEGACY) Forward pass of the Lambda_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cell_output"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Output tensor of the encoder."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsx)(n.p,{children:"Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:"}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsx)(n.li,{children:"torch.Tensor: Sampled latent tensor."}),"\n",(0,i.jsx)(n.li,{children:"torch.Tensor: Mean of the latent distribution."}),"\n",(0,i.jsx)(n.li,{children:"torch.Tensor: Log variance of the latent distribution."}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"decoder_legacy-objects",children:"Decoder_LEGACY Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class Decoder_LEGACY(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"LEGACY Decoder module of the Variational Autoencoder."}),"\n",(0,i.jsx)(n.h4,{id:"__init__-7",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int,\n             hidden_size_rec: int, dropout_rec: float)\n"})}),"\n",(0,i.jsx)(n.p,{children:"(LEGACY) Initialize the Decoder_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"TEMPORAL_WINDOW"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the temporal window."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the latent space."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"NUM_FEATURES"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of input features."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_rec"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the recurrent hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"dropout_rec"})," ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for the decoder."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward-7",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(inputs: torch.Tensor) -> torch.Tensor\n"})}),"\n",(0,i.jsx)(n.p,{children:"(LEGACY) Forward pass of the Decoder_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"inputs"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Input tensor."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"torch.Tensor"})," - Reconstructed tensor."]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"decoder_future_legacy-objects",children:"Decoder_Future_LEGACY Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class Decoder_Future_LEGACY(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"LEGACY Decoder module for predicting future sequences."}),"\n",(0,i.jsx)(n.h4,{id:"__init__-8",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int,\n             FUTURE_STEPS: int, hidden_size_pred: int, dropout_pred: float)\n"})}),"\n",(0,i.jsx)(n.p,{children:"(LEGACY) Initialize the Decoder_Future_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"TEMPORAL_WINDOW"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the temporal window."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the latent space."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"NUM_FEATURES"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of input features."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"FUTURE_STEPS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of future steps to predict."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_pred"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the prediction hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"dropout_pred"})," ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for the prediction."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward-8",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(inputs: torch.Tensor) -> torch.Tensor\n"})}),"\n",(0,i.jsx)(n.p,{children:"(LEGACY) Forward pass of the Decoder_Future_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"inputs"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Input tensor."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"torch.Tensor"})," - Predicted future tensor."]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"rnn_vae_legacy-objects",children:"RNN_VAE_LEGACY Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class RNN_VAE_LEGACY(nn.Module)\n"})}),"\n",(0,i.jsx)(n.p,{children:"LEGACY Variational Autoencoder module."}),"\n",(0,i.jsx)(n.h4,{id:"__init__-9",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def __init__(TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int,\n             FUTURE_DECODER: bool, FUTURE_STEPS: int, hidden_size_layer_1: int,\n             hidden_size_layer_2: int, hidden_size_rec: int,\n             hidden_size_pred: int, dropout_encoder: float, dropout_rec: float,\n             dropout_pred: float, softplus: bool)\n"})}),"\n",(0,i.jsx)(n.p,{children:"(LEGACY) Initialize the RNN_VAE_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"TEMPORAL_WINDOW"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the temporal window."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the latent space."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"NUM_FEATURES"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of input features."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"FUTURE_DECODER"})," ",(0,i.jsx)(n.em,{children:"bool"})," - Whether to include a future decoder."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"FUTURE_STEPS"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of future steps to predict."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_1"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the first hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_layer_2"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the second hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_rec"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the recurrent hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"hidden_size_pred"})," ",(0,i.jsx)(n.em,{children:"int"})," - Size of the prediction hidden layer."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"dropout_encoder"})," ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for the encoder."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"}),"0 ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for the decoder."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"}),"1 ",(0,i.jsx)(n.em,{children:"float"})," - Dropout rate for the prediction."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ZDIMS"}),"2 ",(0,i.jsx)(n.em,{children:"bool, deprecated"})," - Whether to use softplus activation."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"forward-9",children:"forward"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def forward(seq: torch.Tensor) -> Tuple\n"})}),"\n",(0,i.jsx)(n.p,{children:"Forward pass of the RNN_VAE_LEGACY module."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"seq"})," ",(0,i.jsx)(n.em,{children:"torch.Tensor"})," - Input sequence tensor of shape (batch_size, seq_len, NUM_FEATURES)."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"Tuple"})," - Tuple containing:","\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsx)(n.li,{children:"torch.Tensor: Predicted tensor."}),"\n",(0,i.jsx)(n.li,{children:"torch.Tensor: Future prediction tensor if FUTURE_DECODER is True, else nothing."}),"\n",(0,i.jsx)(n.li,{children:"torch.Tensor: Latent tensor."}),"\n",(0,i.jsx)(n.li,{children:"torch.Tensor: Mean of the latent distribution."}),"\n",(0,i.jsx)(n.li,{children:"torch.Tensor: Log variance of the latent distribution."}),"\n"]}),"\n"]}),"\n"]})]})}function a(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(h,{...e})}):h(e)}},8453:(e,n,r)=>{r.d(n,{R:()=>l,x:()=>c});var i=r(6540);const s={},d=i.createContext(s);function l(e){const n=i.useContext(d);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:l(e.components),i.createElement(d.Provider,{value:n},e.children)}}}]);