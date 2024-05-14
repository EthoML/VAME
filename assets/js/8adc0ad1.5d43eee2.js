"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[8061],{3101:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>o,contentTitle:()=>t,default:()=>a,frontMatter:()=>i,metadata:()=>c,toc:()=>d});var r=s(4848),l=s(8453);const i={sidebar_label:"rnn_vae",title:"vame.model.rnn_vae"},t=void 0,c={id:"reference/vame/model/rnn_vae",title:"vame.model.rnn_vae",description:"Variational Animal Motion Embedding 0.1 Toolbox",source:"@site/docs/reference/vame/model/rnn_vae.md",sourceDirName:"reference/vame/model",slug:"/reference/vame/model/rnn_vae",permalink:"/docs/reference/vame/model/rnn_vae",draft:!1,unlisted:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/reference/vame/model/rnn_vae.md",tags:[],version:"current",frontMatter:{sidebar_label:"rnn_vae",title:"vame.model.rnn_vae"},sidebar:"docsSidebar",previous:{title:"rnn_model",permalink:"/docs/reference/vame/model/rnn_model"},next:{title:"align_egocentrical",permalink:"/docs/reference/vame/util/align_egocentrical"}},o={},d=[{value:"reconstruction_loss",id:"reconstruction_loss",level:4},{value:"future_reconstruction_loss",id:"future_reconstruction_loss",level:4},{value:"cluster_loss",id:"cluster_loss",level:4},{value:"kullback_leibler_loss",id:"kullback_leibler_loss",level:4},{value:"kl_annealing",id:"kl_annealing",level:4},{value:"gaussian",id:"gaussian",level:4},{value:"train",id:"train",level:4},{value:"test",id:"test",level:4},{value:"train_model",id:"train_model",level:4}];function h(e){const n={a:"a",code:"code",em:"em",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,l.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.p,{children:"Variational Animal Motion Embedding 0.1 Toolbox\n\xa9 K. Luxem & P. Bauer, Department of Cellular Neuroscience\nLeibniz Institute for Neurobiology, Magdeburg, Germany"}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.a,{href:"https://github.com/LINCellularNeuroscience/VAME",children:"https://github.com/LINCellularNeuroscience/VAME"}),"\nLicensed under GNU General Public License v3.0"]}),"\n",(0,r.jsx)(n.h4,{id:"reconstruction_loss",children:"reconstruction_loss"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def reconstruction_loss(x: torch.Tensor, x_tilde: torch.Tensor,\n                        reduction: str) -> torch.Tensor\n"})}),"\n",(0,r.jsx)(n.p,{children:"Compute the reconstruction loss between input and reconstructed data."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"x"})," ",(0,r.jsx)(n.em,{children:"torch.Tensor"})," - Input data tensor."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"x_tilde"})," ",(0,r.jsx)(n.em,{children:"torch.Tensor"})," - Reconstructed data tensor."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"reduction"})," ",(0,r.jsx)(n.em,{children:"str"})," - Type of reduction for the loss."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"torch.Tensor"})," - Reconstruction loss."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"future_reconstruction_loss",children:"future_reconstruction_loss"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def future_reconstruction_loss(x: torch.Tensor, x_tilde: torch.Tensor,\n                               reduction: str) -> torch.Tensor\n"})}),"\n",(0,r.jsx)(n.p,{children:"Compute the future reconstruction loss between input and predicted future data."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"x"})," ",(0,r.jsx)(n.em,{children:"torch.Tensor"})," - Input future data tensor."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"x_tilde"})," ",(0,r.jsx)(n.em,{children:"torch.Tensor"})," - Reconstructed future data tensor."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"reduction"})," ",(0,r.jsx)(n.em,{children:"str"})," - Type of reduction for the loss."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"torch.Tensor"})," - Future reconstruction loss."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"cluster_loss",children:"cluster_loss"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def cluster_loss(H: torch.Tensor, kloss: int, lmbda: float,\n                 batch_size: int) -> torch.Tensor\n"})}),"\n",(0,r.jsx)(n.p,{children:"Compute the cluster loss."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"H"})," ",(0,r.jsx)(n.em,{children:"torch.Tensor"})," - Latent representation tensor."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"kloss"})," ",(0,r.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"lmbda"})," ",(0,r.jsx)(n.em,{children:"float"})," - Lambda value for the loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"batch_size"})," ",(0,r.jsx)(n.em,{children:"int"})," - Size of the batch."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"torch.Tensor"})," - Cluster loss."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"kullback_leibler_loss",children:"kullback_leibler_loss"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def kullback_leibler_loss(mu: torch.Tensor,\n                          logvar: torch.Tensor) -> torch.Tensor\n"})}),"\n",(0,r.jsxs)(n.p,{children:["Compute the Kullback-Leibler divergence loss.\nsee Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 - ",(0,r.jsx)(n.a,{href:"https://arxiv.org/abs/1312.6114",children:"https://arxiv.org/abs/1312.6114"})]}),"\n",(0,r.jsx)(n.p,{children:"Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)"}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"mu"})," ",(0,r.jsx)(n.em,{children:"torch.Tensor"})," - Mean of the latent distribution."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"logvar"})," ",(0,r.jsx)(n.em,{children:"torch.Tensor"})," - Log variance of the latent distribution."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"torch.Tensor"})," - Kullback-Leibler divergence loss."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"kl_annealing",children:"kl_annealing"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def kl_annealing(epoch: int, kl_start: int, annealtime: int,\n                 function: str) -> float\n"})}),"\n",(0,r.jsx)(n.p,{children:"Anneal the Kullback-Leibler loss to let the model learn first the reconstruction of the data\nbefore the KL loss term gets introduced."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"})," ",(0,r.jsx)(n.em,{children:"int"})," - Current epoch number."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"kl_start"})," ",(0,r.jsx)(n.em,{children:"int"})," - Epoch number to start annealing the loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"annealtime"})," ",(0,r.jsx)(n.em,{children:"int"})," - Annealing time."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"function"})," ",(0,r.jsx)(n.em,{children:"str"})," - Annealing function type."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"float"})," - Annealed weight value for the loss."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"gaussian",children:"gaussian"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def gaussian(ins: torch.Tensor,\n             is_training: bool,\n             seq_len: int,\n             std_n: float = 0.8) -> torch.Tensor\n"})}),"\n",(0,r.jsx)(n.p,{children:"Add Gaussian noise to the input data."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"ins"})," ",(0,r.jsx)(n.em,{children:"torch.Tensor"})," - Input data tensor."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"is_training"})," ",(0,r.jsx)(n.em,{children:"bool"})," - Whether it is training mode."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"seq_len"})," ",(0,r.jsx)(n.em,{children:"int"})," - Length of the sequence."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"std_n"})," ",(0,r.jsx)(n.em,{children:"float"})," - Standard deviation for the Gaussian noise."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"torch.Tensor"})," - Noisy input data tensor."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"train",children:"train"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def train(train_loader: Data.DataLoader, epoch: int, model: nn.Module,\n          optimizer: torch.optim.Optimizer, anneal_function: str, BETA: float,\n          kl_start: int, annealtime: int, seq_len: int, future_decoder: bool,\n          future_steps: int, scheduler: torch.optim.lr_scheduler._LRScheduler,\n          mse_red: str, mse_pred: str, kloss: int, klmbda: float, bsize: int,\n          noise: bool) -> Tuple[float, float, float, float, float, float]\n"})}),"\n",(0,r.jsx)(n.p,{children:"Train the model."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"train_loader"})," ",(0,r.jsx)(n.em,{children:"DataLoader"})," - Training data loader."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"})," ",(0,r.jsx)(n.em,{children:"int"})," - Current epoch number."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"model"})," ",(0,r.jsx)(n.em,{children:"nn.Module"})," - Model to be trained."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"optimizer"})," ",(0,r.jsx)(n.em,{children:"Optimizer"})," - Optimizer for training."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"anneal_function"})," ",(0,r.jsx)(n.em,{children:"str"})," - Annealing function type."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"BETA"})," ",(0,r.jsx)(n.em,{children:"float"})," - Beta value for the loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"kl_start"})," ",(0,r.jsx)(n.em,{children:"int"})," - Epoch number to start annealing the loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"annealtime"})," ",(0,r.jsx)(n.em,{children:"int"})," - Annealing time."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"seq_len"})," ",(0,r.jsx)(n.em,{children:"int"})," - Length of the sequence."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"future_decoder"})," ",(0,r.jsx)(n.em,{children:"bool"})," - Whether a future decoder is used."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"0 ",(0,r.jsx)(n.em,{children:"int"})," - Number of future steps to predict."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"1 _lr_scheduler.",(0,r.jsx)(n.em,{children:"LRScheduler"})," - Learning rate scheduler."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"2 ",(0,r.jsx)(n.em,{children:"str"})," - Reduction type for MSE reconstruction loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"3 ",(0,r.jsx)(n.em,{children:"str"})," - Reduction type for MSE prediction loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"4 ",(0,r.jsx)(n.em,{children:"int"})," - Number of clusters for cluster loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"5 ",(0,r.jsx)(n.em,{children:"float"})," - Lambda value for cluster loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"6 ",(0,r.jsx)(n.em,{children:"int"})," - Size of the batch."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"7 ",(0,r.jsx)(n.em,{children:"bool"})," - Whether to add Gaussian noise to the input."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsx)(n.p,{children:"Tuple[float, float, float, float, float, float]: Kullback-Leibler weight, train loss, K-means loss, KL loss,\nMSE loss, future loss."}),"\n",(0,r.jsx)(n.h4,{id:"test",children:"test"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def test(test_loader: Data.DataLoader, epoch: int, model: nn.Module,\n         optimizer: torch.optim.Optimizer, BETA: float, kl_weight: float,\n         seq_len: int, mse_red: str, kloss: str, klmbda: float,\n         future_decoder: bool, bsize: int) -> Tuple[float, float, float]\n"})}),"\n",(0,r.jsx)(n.p,{children:"Evaluate the model on the test dataset."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"test_loader"})," ",(0,r.jsx)(n.em,{children:"DataLoader"})," - DataLoader for the test dataset."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"})," ",(0,r.jsx)(n.em,{children:"int, deprecated"})," - Current epoch number."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"model"})," ",(0,r.jsx)(n.em,{children:"nn.Module"})," - The trained model."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"optimizer"})," ",(0,r.jsx)(n.em,{children:"Optimizer, deprecated"})," - The optimizer used for training."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"BETA"})," ",(0,r.jsx)(n.em,{children:"float"})," - Beta value for the VAE loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"kl_weight"})," ",(0,r.jsx)(n.em,{children:"float"})," - Weighting factor for the KL divergence loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"seq_len"})," ",(0,r.jsx)(n.em,{children:"int"})," - Length of the sequence."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"mse_red"})," ",(0,r.jsx)(n.em,{children:"str"})," - Reduction method for the MSE loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"kloss"})," ",(0,r.jsx)(n.em,{children:"str"})," - Loss function for K-means clustering."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"klmbda"})," ",(0,r.jsx)(n.em,{children:"float"})," - Lambda value for K-means loss."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"0 ",(0,r.jsx)(n.em,{children:"bool"})," - Flag indicating whether to use a future decoder."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"epoch"}),"1 ",(0,r.jsx)(n.em,{children:"int"})," - Batch size."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsx)(n.p,{children:"Tuple[float, float, float]: Tuple containing MSE loss per item, total test loss per item,\nand K-means loss weighted by the kl_weight."}),"\n",(0,r.jsx)(n.h4,{id:"train_model",children:"train_model"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def train_model(config: str) -> None\n"})}),"\n",(0,r.jsx)(n.p,{children:"Train Variational Autoencoder using the configuration file values."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"config"})," ",(0,r.jsx)(n.em,{children:"str"})," - Path to the configuration file."]}),"\n"]})]})}function a(e={}){const{wrapper:n}={...(0,l.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(h,{...e})}):h(e)}},8453:(e,n,s)=>{s.d(n,{R:()=>t,x:()=>c});var r=s(6540);const l={},i=r.createContext(l);function t(e){const n=r.useContext(i);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(l):e.components||l:t(e.components),r.createElement(i.Provider,{value:n},e.children)}}}]);