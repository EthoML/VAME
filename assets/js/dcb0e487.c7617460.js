"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[9350],{9710:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>l,contentTitle:()=>a,default:()=>h,frontMatter:()=>s,metadata:()=>r,toc:()=>d});var t=i(4848),o=i(8453);const s={title:"VAME step-by-step",sidebar_position:2},a=void 0,r={id:"getting_started/step_by_step",title:"VAME step-by-step",description:"Open In Colab",source:"@site/docs/getting_started/step_by_step.mdx",sourceDirName:"getting_started",slug:"/getting_started/step_by_step",permalink:"/VAME/docs/getting_started/step_by_step",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:2,frontMatter:{title:"VAME step-by-step",sidebar_position:2},sidebar:"docsSidebar",previous:{title:"Installation",permalink:"/VAME/docs/getting_started/installation"},next:{title:"VAME Pipeline",permalink:"/VAME/docs/getting_started/pipeline"}},l={},d=[{value:"Input data",id:"input-data",level:2},{value:"Step 1: Initialize your project",id:"step-1-initialize-your-project",level:2},{value:"Step 2: Preprocess the raw pose estimation data",id:"step-2-preprocess-the-raw-pose-estimation-data",level:2},{value:"Cleaning low confidence data points",id:"cleaning-low-confidence-data-points",level:4},{value:"Egocentric alignment using key reference points",id:"egocentric-alignment-using-key-reference-points",level:4},{value:"Outlier cleaning",id:"outlier-cleaning",level:4},{value:"Savitzky-Golay filtering",id:"savitzky-golay-filtering",level:4},{value:"Step 3: Train the VAME model",id:"step-3-train-the-vame-model",level:2},{value:"Step 4: Segment behavior",id:"step-4-segment-behavior",level:2},{value:"Step 5: Vizualization and analysis",id:"step-5-vizualization-and-analysis",level:2},{value:"Create motif and community videos",id:"create-motif-and-community-videos",level:4}];function c(e){const n={a:"a",admonition:"admonition",code:"code",h2:"h2",h4:"h4",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,o.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.p,{children:(0,t.jsx)(n.a,{href:"https://colab.research.google.com/github/EthoML/VAME/blob/main/examples/demo.ipynb",children:(0,t.jsx)(n.img,{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Open In Colab"})})}),"\n",(0,t.jsx)(n.admonition,{type:"tip",children:(0,t.jsxs)(n.p,{children:["Check out also the published VAME Workflow Guide, including more hands-on recommendations ",(0,t.jsx)(n.a,{href:"https://www.nature.com/articles/s42003-022-04080-7#Sec8",children:"HERE"}),"."]})}),"\n",(0,t.jsx)(n.admonition,{type:"tip",children:(0,t.jsxs)(n.p,{children:["You can run an entire VAME workflow with just a few lines, using the ",(0,t.jsx)(n.a,{href:"/docs/getting_started/pipeline",children:"Pipeline method"}),"."]})}),"\n",(0,t.jsx)(n.p,{children:"If you haven't yet, please install VAME:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"pip install vame\n"})}),"\n",(0,t.jsx)(n.p,{children:"The VAME workflow consists of four main steps, plus optional analysis:"}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Initialize project"}),": In this step we will start the project and get your pose estimation data into the ",(0,t.jsx)(n.code,{children:"movement"})," format"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Preprocess"}),": This step will perform cleaning, filtering and alignment of the raw pose estimation data"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Train the VAME model"}),":","\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Split the input data into training and test datasets."}),"\n",(0,t.jsx)(n.li,{children:"Train the VAME model to embed behavioural dynamics."}),"\n",(0,t.jsx)(n.li,{children:"Evaluate the performance of the trained model based on its reconstruction capabilities."}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Segment behavior"}),":","\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Segment pose estimation time series into behavioral motifs, using HMM or K-means."}),"\n",(0,t.jsx)(n.li,{children:"Group similar motifs into communities, using hierarchical clustering."}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Vizualization and analysis [Optional]"}),":","\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Visualization and projection of latent vectors onto a 2D plane via UMAP."}),"\n",(0,t.jsx)(n.li,{children:"Create motif and community videos."}),"\n",(0,t.jsx)(n.li,{children:"Use the generative model (reconstruction decoder) to sample from the learned data distribution."}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,t.jsx)(n.p,{children:"Let's start by importing the necessary libraries:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"import vame\nfrom vame.util.sample_data import download_sample_data\nfrom pathlib import Path\n"})}),"\n",(0,t.jsx)(n.h2,{id:"input-data",children:"Input data"}),"\n",(0,t.jsx)(n.p,{children:"To quickly try VAME, you can download sample data and use it as input. If you want to work with your own data, all you need to do is to provide the paths to the pose estimation files as lists of strings. You can also optionally provide the paths to the corresponding video files."}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'# You can run VAME with data from different sources:\n# "DeepLabCut", "SLEAP" or "LightningPose"\nsource_software = "DeepLabCut"\n\n# Download sample data\nps = download_sample_data(source_software)\nvideos = [ps["video"]]\nposes_estimations = [ps["poses"]]\n\nprint(videos)\nprint(poses_estimations)\n'})}),"\n",(0,t.jsx)(n.h2,{id:"step-1-initialize-your-project",children:"Step 1: Initialize your project"}),"\n",(0,t.jsx)(n.p,{children:"VAME organizes around projects. To start a new project, you need to define some basic things:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"the project's name"}),"\n",(0,t.jsx)(n.li,{children:"the paths to the pose estimation files"}),"\n",(0,t.jsx)(n.li,{children:"the source software used to produce the pose estimation data"}),"\n"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'config_file, config_data = vame.init_new_project(\n    project_name="my_vame_project",\n    poses_estimations=poses_estimations,\n    source_software="DeepLabCut",\n)\n'})}),"\n",(0,t.jsxs)(n.p,{children:["This command will create a project folder in the defined working directory with the project name you defined.\nIn this folder you can find a config file called ",(0,t.jsx)(n.a,{href:"/docs/project-config",children:"config.yaml"})," which holds the main parameters for the VAME workflow."]}),"\n",(0,t.jsx)(n.p,{children:"The videos and pose estimation files will be linked or copied to the project folder."}),"\n",(0,t.jsx)(n.p,{children:"Let's take a look at the project's configuration:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"print(config_data)\n"})}),"\n",(0,t.jsx)(n.p,{children:"Now let's take a look at the formatted input dataset:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'ds_path = Path(config_data["project_path"]) / "data" / "raw" / f"{config_data[\'session_names\'][0]}.nc"\nvame.io.load_poses.load_vame_dataset(ds_path)\n'})}),"\n",(0,t.jsx)(n.h2,{id:"step-2-preprocess-the-raw-pose-estimation-data",children:"Step 2: Preprocess the raw pose estimation data"}),"\n",(0,t.jsx)(n.p,{children:"The preprocessing step includes:"}),"\n",(0,t.jsx)(n.h4,{id:"cleaning-low-confidence-data-points",children:"Cleaning low confidence data points"}),"\n",(0,t.jsx)(n.p,{children:"Pose estimation data points with confidence below the threshold will be cleared and interpolated."}),"\n",(0,t.jsx)(n.h4,{id:"egocentric-alignment-using-key-reference-points",children:"Egocentric alignment using key reference points"}),"\n",(0,t.jsx)(n.p,{children:"Based on two reference keypoints, the data will be aligned to an egocentric coordinate system:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"centered_reference_keypoint"}),": The keypoint that will be centered in the frame."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"orientation_reference_keypoint"}),": The keypoint that will be used to determine the rotation of the frame."]}),"\n"]}),"\n",(0,t.jsxs)(n.p,{children:["By consequence, the ",(0,t.jsx)(n.code,{children:"x"})," and ",(0,t.jsx)(n.code,{children:"y"})," coordinates of the ",(0,t.jsx)(n.code,{children:"centered_reference_keypoint"})," and the ",(0,t.jsx)(n.code,{children:"x"})," coordinate of the ",(0,t.jsx)(n.code,{children:"orientation_reference_keypoint"})," will be set to an array of zeros, and further removed from the dataset."]}),"\n",(0,t.jsx)(n.h4,{id:"outlier-cleaning",children:"Outlier cleaning"}),"\n",(0,t.jsxs)(n.p,{children:["Outliers will be removed based on the interquartile range (IQR) method. This means that data points that are below ",(0,t.jsx)(n.code,{children:"Q1 - iqr_factor * IQR"})," or above ",(0,t.jsx)(n.code,{children:"Q3 + iqr_factor * IQR"})," will be cleared and interpolated."]}),"\n",(0,t.jsx)(n.h4,{id:"savitzky-golay-filtering",children:"Savitzky-Golay filtering"}),"\n",(0,t.jsx)(n.p,{children:"The data will be further smoothed using a Savitzky-Golay filter."}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'vame.preprocessing(\n    config=config_data,\n    centered_reference_keypoint="snout",\n    orientation_reference_keypoint="tailbase",\n)\n'})}),"\n",(0,t.jsx)(n.h2,{id:"step-3-train-the-vame-model",children:"Step 3: Train the VAME model"}),"\n",(0,t.jsx)(n.p,{children:"At this point, we will prepare the data for training the VAME model, run the training and evaluate the model."}),"\n",(0,t.jsx)(n.p,{children:"We start by splitting the input data into train and test sets:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"vame.create_trainset(config=config_data)\n"})}),"\n",(0,t.jsx)(n.p,{children:"Now we can train the VAME model. This migth take a while, depending on dataset size and your hardware."}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"vame.train_model(config=config_data)\n"})}),"\n",(0,t.jsx)(n.p,{children:"The model evaluation produces two plots, one showing the loss of the model during training and the other showing the reconstruction and future prediction of input sequence."}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"vame.evaluate_model(config=config_data)\n"})}),"\n",(0,t.jsx)(n.h2,{id:"step-4-segment-behavior",children:"Step 4: Segment behavior"}),"\n",(0,t.jsx)(n.p,{children:"Behavioral segmentation in VAME is done in two steps:"}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Segmentation of pose estimation data into motifs"}),"\n",(0,t.jsx)(n.li,{children:"Clustering motifs in communities"}),"\n"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"vame.segment_session(config=config_data)\n"})}),"\n",(0,t.jsx)(n.p,{children:"This will perfomr the segmentation using two different algorithms: HMM and K-means. The results will be saved in the project folder."}),"\n",(0,t.jsx)(n.p,{children:"Community detection is done by grouping similar motifs into communities using hierarchical clustering. For that you must choose:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"segmentation_algorithm"}),', which can be either "hmm" or "kmeans"']}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"cut_tree"}),", which is the cut level for the hierarchical clustering"]}),"\n"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'vame.community(\n    config=config_data,\n    segmentation_algorithm="hmm",\n    cut_tree=2,\n)\n'})}),"\n",(0,t.jsx)(n.h2,{id:"step-5-vizualization-and-analysis",children:"Step 5: Vizualization and analysis"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"from vame.visualization.motif import visualize_motif_tree\nfrom vame.visualization.umap import visualize_umap\nfrom vame.visualization.preprocessing import (\n    visualize_preprocessing_scatter,\n    visualize_preprocessing_timeseries,\n)\nfrom vame.visualization.model import plot_loss\n"})}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"visualize_preprocessing_scatter(config=config_data)\n"})}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"visualize_preprocessing_timeseries(config=config_data)\n"})}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'plot_loss(cfg=config_data, model_name="VAME")\n'})}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'visualize_motif_tree(\n    config=config_data,\n    segmentation_algorithm="hmm",\n)\n'})}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'visualize_umap(\n    config=config_data,\n    label="community",\n    segmentation_algorithm="hmm",\n)\n'})}),"\n",(0,t.jsx)(n.h4,{id:"create-motif-and-community-videos",children:"Create motif and community videos"}),"\n",(0,t.jsx)(n.p,{children:"VAME only needs the pose estimation data to generate motifs and communities. But it provides auxiliary functions to split original videos into motifs or communities videos."}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"from vame.video import add_videos_to_project\n\nadd_videos_to_project(config=config_data, videos=videos)\n"})}),"\n",(0,t.jsx)(n.p,{children:"Create motif videos to get insights about the fine grained poses:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"vame.motif_videos(\n    config=config_data,\n    segmentation_algorithm='hmm',\n)\n"})}),"\n",(0,t.jsx)(n.p,{children:"Create community videos to get insights about behavior on a hierarchical scale:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"vame.community_videos(\n    config=config_data,\n    segmentation_algorithm='hmm',\n)\n"})})]})}function h(e={}){const{wrapper:n}={...(0,o.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(c,{...e})}):c(e)}},8453:(e,n,i)=>{i.d(n,{R:()=>a,x:()=>r});var t=i(6540);const o={},s=t.createContext(o);function a(e){const n=t.useContext(s);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:a(e.components),t.createElement(s.Provider,{value:n},e.children)}}}]);