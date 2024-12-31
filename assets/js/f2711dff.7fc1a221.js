"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[1824],{7623:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>l,contentTitle:()=>s,default:()=>h,frontMatter:()=>o,metadata:()=>a,toc:()=>c});var i=t(4848),r=t(8453);const o={title:"Installation",sidebar_position:1},s=void 0,a={id:"getting_started/installation",title:"Installation",description:"Installation",source:"@site/docs/getting_started/installation.md",sourceDirName:"getting_started",slug:"/getting_started/installation",permalink:"/VAME/docs/getting_started/installation",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:1,frontMatter:{title:"Installation",sidebar_position:1},sidebar:"docsSidebar",previous:{title:"Getting Started",permalink:"/VAME/docs/category/getting-started"},next:{title:"Running VAME Workflow",permalink:"/VAME/docs/getting_started/running"}},l={},c=[{value:"Installation",id:"installation",level:2},{value:"Install with pip (Recommended)",id:"install-with-pip-recommended",level:3},{value:"Install from Github repository",id:"install-from-github-repository",level:3},{value:"References",id:"references",level:2},{value:"License: GPLv3",id:"license-gplv3",level:2}];function d(e){const n={a:"a",admonition:"admonition",code:"code",h2:"h2",h3:"h3",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",...(0,r.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.h2,{id:"installation",children:"Installation"}),"\n",(0,i.jsxs)(n.p,{children:["To get started we recommend using ",(0,i.jsx)(n.a,{href:"https://www.anaconda.com/distribution/",children:"Anaconda"})," or ",(0,i.jsx)(n.a,{href:"https://docs.python.org/3/library/venv.html",children:"Virtual Environment"})," with Python 3.11 or higher."]}),"\n",(0,i.jsx)(n.h3,{id:"install-with-pip-recommended",children:"Install with pip (Recommended)"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"pip install vame-py\n"})}),"\n",(0,i.jsx)(n.h3,{id:"install-from-github-repository",children:"Install from Github repository"}),"\n",(0,i.jsxs)(n.ol,{children:["\n",(0,i.jsx)(n.li,{children:"Clone the VAME repository to your local machine by running"}),"\n"]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-bash",children:"git clone https://github.com/EthoML/VAME.git\ncd VAME\n"})}),"\n",(0,i.jsxs)(n.ol,{start:"2",children:["\n",(0,i.jsx)(n.li,{children:"Installing VAME from local source"}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Option 1:"})," Using VAME.yaml file to create a conda environment and install VAME in it by running"]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-bash",children:"conda env create -f VAME.yaml\n"})}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Option 2:"}),"  Installing local VAME with pip in your active virtual environment by running"]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-bash",children:"pip install .\n"})}),"\n",(0,i.jsx)(n.admonition,{type:"warning",children:(0,i.jsxs)(n.p,{children:["You should make sure that you have a GPU powerful enough to train deep learning networks. In our original 2022 paper, we were using a single Nvidia GTX 1080 Ti GPU to train our network. A hardware guide can be found ",(0,i.jsx)(n.a,{href:"https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/",children:"here"}),"."]})}),"\n",(0,i.jsx)(n.admonition,{type:"tip",children:(0,i.jsx)(n.p,{children:"VAME can also be trained in Google Colab or on a HPC cluster."})}),"\n",(0,i.jsxs)(n.p,{children:["Once you have your computing setup ready, begin using VAME by following the ",(0,i.jsx)(n.a,{href:"/docs/getting_started/running",children:"demo workflow guide"}),"."]}),"\n",(0,i.jsx)(n.h2,{id:"references",children:"References"}),"\n",(0,i.jsxs)(n.p,{children:["Original VAME publication: ",(0,i.jsx)(n.a,{href:"https://www.biorxiv.org/content/10.1101/2020.05.14.095430v2",children:"Identifying Behavioral Structure from Deep Variational Embeddings of Animal Motion"})," ",(0,i.jsx)("br",{}),"\nKingma & Welling: ",(0,i.jsx)(n.a,{href:"https://arxiv.org/abs/1312.6114",children:"Auto-Encoding Variational Bayes"})," ",(0,i.jsx)("br",{}),"\nPereira & Silveira: ",(0,i.jsx)(n.a,{href:"https://www.joao-pereira.pt/publications/accepted_version_BigComp19.pdf",children:"Learning Representations from Healthcare Time Series Data for Unsupervised Anomaly Detection"})]}),"\n",(0,i.jsx)(n.h2,{id:"license-gplv3",children:"License: GPLv3"}),"\n",(0,i.jsxs)(n.p,{children:["See the ",(0,i.jsx)(n.a,{href:"https://github.com/LINCellularNeuroscience/VAME/blob/master/LICENSE",children:"LICENSE file"})," for the full statement."]})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(d,{...e})}):d(e)}},8453:(e,n,t)=>{t.d(n,{R:()=>s,x:()=>a});var i=t(6540);const r={},o=i.createContext(r);function s(e){const n=i.useContext(o);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:s(e.components),i.createElement(o.Provider,{value:n},e.children)}}}]);