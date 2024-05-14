"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[6184],{3100:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>l,contentTitle:()=>s,default:()=>h,frontMatter:()=>r,metadata:()=>a,toc:()=>d});var t=i(4848),o=i(8453);const r={title:"Running VAME Workflow",sidebar_position:2},s=void 0,a={id:"getting_started/running",title:"Running VAME Workflow",description:"Workflow Overview",source:"@site/docs/getting_started/running.md",sourceDirName:"getting_started",slug:"/getting_started/running",permalink:"/VAME/docs/getting_started/running",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:2,frontMatter:{title:"Running VAME Workflow",sidebar_position:2},sidebar:"docsSidebar",previous:{title:"Installation",permalink:"/VAME/docs/getting_started/installation"},next:{title:"API Reference",permalink:"/VAME/docs/category/api-reference"}},l={},d=[{value:"Workflow Overview",id:"workflow-overview",level:2},{value:"Running a demo workflow",id:"running-a-demo-workflow",level:2},{value:"1. Download the necessary resources:",id:"1-download-the-necessary-resources",level:3},{value:"2. Setting the demo variables",id:"2-setting-the-demo-variables",level:3},{value:"3. Running the demo",id:"3-running-the-demo",level:3}];function c(e){const n={a:"a",admonition:"admonition",code:"code",h2:"h2",h3:"h3",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,o.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h2,{id:"workflow-overview",children:"Workflow Overview"}),"\n",(0,t.jsxs)(n.p,{children:["The below diagram shows the workflow of the VAME application, which consists of four main steps and optional steps to analyse your data.\n",(0,t.jsx)(n.img,{alt:"Workflow Overview",src:i(2416).A+"",width:"753",height:"92"})]}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Initialize project: This is step is responsible by starting the project, getting your data into the right format and creating a training dataset for the VAME deep learning model."}),"\n",(0,t.jsx)(n.li,{children:"Train neural network: Train a variational autoencoder which is parameterized with recurrent neural network to embed behavioural dynamics"}),"\n",(0,t.jsx)(n.li,{children:"Evaluate performance: Evaluate the trained model based on its reconstruction capabilities"}),"\n",(0,t.jsx)(n.li,{children:"Segment behavior: Segment behavioural motifs/poses/states from the input time series"}),"\n",(0,t.jsxs)(n.li,{children:["Quantify behavior:","\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Optional: Create motif videos to get insights about the fine grained poses."}),"\n",(0,t.jsx)(n.li,{children:"Optional: Investigate the hierarchical order of your behavioural states by detecting communities in the resulting markov chain."}),"\n",(0,t.jsx)(n.li,{children:"Optional: Create community videos to get more insights about behaviour on a hierarchical scale."}),"\n",(0,t.jsx)(n.li,{children:"Optional: Visualization and projection of latent vectors onto a 2D plane via UMAP."}),"\n",(0,t.jsx)(n.li,{children:"Optional: Use the generative model (reconstruction decoder) to sample from the learned data distribution, reconstruct random real samples or visualize the cluster centre for validation."}),"\n",(0,t.jsx)(n.li,{children:"Optional: Create a video of an egocentrically aligned animal + path through the community space (similar to our gif on github readme)."}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,t.jsx)(n.admonition,{type:"tip",children:(0,t.jsxs)(n.p,{children:["\u26a0\ufe0f Check out also the published VAME Workflow Guide, including more hands-on recommendations and tricks ",(0,t.jsx)(n.a,{href:"https://www.nature.com/articles/s42003-022-04080-7#Sec8",children:"HERE"}),"."]})}),"\n",(0,t.jsx)(n.h2,{id:"running-a-demo-workflow",children:"Running a demo workflow"}),"\n",(0,t.jsxs)(n.p,{children:["In our github in ",(0,t.jsx)(n.code,{children:"/examples"})," folder there is a demo script called ",(0,t.jsx)(n.code,{children:"demo.py"})," that you can use to run a simple example of the VAME workflow. To run this workflow you will need to do the following:"]}),"\n",(0,t.jsx)(n.h3,{id:"1-download-the-necessary-resources",children:"1. Download the necessary resources:"}),"\n",(0,t.jsx)(n.p,{children:"To run the demo you will need a video and a csv file with the pose estimation results. You can use the following files links:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"video-1.mp4"}),": Video file ",(0,t.jsx)(n.a,{href:"https://drive.google.com/file/d/1w6OW9cN_-S30B7rOANvSaR9c3O5KeF0c/view",children:"link"})]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"video-1.csv"}),": Pose estimation results ",(0,t.jsx)(n.a,{href:"https://github.com/EthoML/VAME/blob/master/examples/video-1.csv",children:"link"})]}),"\n"]}),"\n",(0,t.jsx)(n.h3,{id:"2-setting-the-demo-variables",children:"2. Setting the demo variables"}),"\n",(0,t.jsxs)(n.p,{children:["To start the demo you must define 4 variables in the ",(0,t.jsx)(n.code,{children:"demo.py"})," script. In order to do that, open the ",(0,t.jsx)(n.code,{children:"demo.py"})," file and edit the following:"]}),"\n",(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"The values below are just examples. You must set the variables according to your needs."})}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"# The directory where the project will be saved\nworking_directory = './'\n\n# The name you want for the project\nproject = 'first_vame_project'\n\n# A list of paths to the videos file\nvideos =  ['./video-1.mp4']\n\n# A list of paths to the poses estimations files.\n# Important: The name (without the extension) of the video file and the pose estimation file must be the same. E.g. `video-1.mp4` and `video-1.csv`\nposes_estimations = ['./video-1.csv']\n"})}),"\n",(0,t.jsx)(n.h3,{id:"3-running-the-demo",children:"3. Running the demo"}),"\n",(0,t.jsx)(n.p,{children:"After setting the variables, you can run the demo by running the following code:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"python demo.py\n"})}),"\n",(0,t.jsxs)(n.p,{children:["The demo will create a project folder in the defined working directory with the name you set in the ",(0,t.jsx)(n.code,{children:"project"})," variable and a date suffix, e.g: ",(0,t.jsx)(n.code,{children:"first_name-May-9-2024"}),"."]}),"\n",(0,t.jsxs)(n.p,{children:["In this folder you can find a config file called ",(0,t.jsx)(n.code,{children:"config.yaml"})," where you can set the parameters for the VAME algorithm. The videos and poses estimations files will be copied to the project videos folder. If everything is ok, the workflow will run and the logs will be displayed in your terminal."]})]})}function h(e={}){const{wrapper:n}={...(0,o.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(c,{...e})}):c(e)}},2416:(e,n,i)=>{i.d(n,{A:()=>t});const t=i.p+"assets/images/workflow_overview-eaa7ffc1b70952df84277ded89585b52.png"},8453:(e,n,i)=>{i.d(n,{R:()=>s,x:()=>a});var t=i(6540);const o={},r=t.createContext(o);function s(e){const n=t.useContext(r);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:s(e.components),t.createElement(r.Provider,{value:n},e.children)}}}]);