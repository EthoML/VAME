"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[1470],{617:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>a,contentTitle:()=>t,default:()=>h,frontMatter:()=>l,metadata:()=>c,toc:()=>d});var i=s(4848),r=s(8453);const l={sidebar_label:"community_analysis",title:"vame.analysis.community_analysis"},t=void 0,c={id:"reference/vame/analysis/community_analysis",title:"vame.analysis.community_analysis",description:"Variational Animal Motion Embedding 1.0-alpha Toolbox",source:"@site/docs/reference/vame/analysis/community_analysis.md",sourceDirName:"reference/vame/analysis",slug:"/reference/vame/analysis/community_analysis",permalink:"/VAME/docs/reference/vame/analysis/community_analysis",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"community_analysis",title:"vame.analysis.community_analysis"},sidebar:"docsSidebar",previous:{title:"API reference",permalink:"/VAME/docs/category/api-reference"},next:{title:"generative_functions",permalink:"/VAME/docs/reference/vame/analysis/generative_functions"}},a={},d=[{value:"get_adjacency_matrix",id:"get_adjacency_matrix",level:4},{value:"get_transition_matrix",id:"get_transition_matrix",level:4},{value:"consecutive",id:"consecutive",level:4},{value:"find_zero_labels",id:"find_zero_labels",level:4},{value:"augment_motif_timeseries",id:"augment_motif_timeseries",level:4},{value:"get_labels",id:"get_labels",level:4},{value:"get_community_label",id:"get_community_label",level:4},{value:"compute_transition_matrices",id:"compute_transition_matrices",level:4},{value:"create_community_bag",id:"create_community_bag",level:4},{value:"create_cohort_community_bag",id:"create_cohort_community_bag",level:4},{value:"get_community_labels",id:"get_community_labels",level:4},{value:"get_cohort_community_labels",id:"get_cohort_community_labels",level:4},{value:"umap_embedding",id:"umap_embedding",level:4},{value:"umap_vis",id:"umap_vis",level:4},{value:"community",id:"community",level:4}];function o(e){const n={a:"a",code:"code",em:"em",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.p,{children:"Variational Animal Motion Embedding 1.0-alpha Toolbox\n\xc2\xa9 K. Luxem & P. Bauer, Department of Cellular Neuroscience\nLeibniz Institute for Neurobiology, Magdeburg, Germany"}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.a,{href:"https://github.com/LINCellularNeuroscience/VAME",children:"https://github.com/LINCellularNeuroscience/VAME"}),"\nLicensed under GNU General Public License v3.0"]}),"\n",(0,i.jsx)(n.p,{children:"Updated 5/11/2022 with PH edits"}),"\n",(0,i.jsx)(n.h4,{id:"get_adjacency_matrix",children:"get_adjacency_matrix"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_adjacency_matrix(\n        labels: np.ndarray,\n        n_cluster: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]\n"})}),"\n",(0,i.jsx)(n.p,{children:"Calculate the adjacency matrix, transition matrix, and temporal matrix."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"labels"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - Array of cluster labels."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"n_cluster"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsx)(n.p,{children:"Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing adjacency matrix, transition matrix, and temporal matrix."}),"\n",(0,i.jsx)(n.h4,{id:"get_transition_matrix",children:"get_transition_matrix"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_transition_matrix(adjacency_matrix: np.ndarray,\n                          threshold: float = 0.0) -> np.ndarray\n"})}),"\n",(0,i.jsx)(n.p,{children:"Compute the transition matrix from the adjacency matrix."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"adjacency_matrix"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - Adjacency matrix."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"threshold"})," ",(0,i.jsx)(n.em,{children:"float, optional"})," - Threshold for considering transitions. Defaults to 0.0."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"np.ndarray"})," - Transition matrix."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"consecutive",children:"consecutive"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def consecutive(data: np.ndarray, stepsize: int = 1) -> List[np.ndarray]\n"})}),"\n",(0,i.jsx)(n.p,{children:"Identifies location of missing motif finding consecutive elements in an array and returns array(s) at the split."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"data"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - Input array."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"stepsize"})," ",(0,i.jsx)(n.em,{children:"int, optional"})," - Step size. Defaults to 1."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"List[np.ndarray]"})," - List of arrays containing consecutive elements."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"find_zero_labels",children:"find_zero_labels"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def find_zero_labels(motif_usage: Tuple[np.ndarray, np.ndarray],\n                     n_cluster: int) -> np.ndarray\n"})}),"\n",(0,i.jsx)(n.p,{children:"Find zero labels in motif usage and fill them."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"motif_usage"})," ",(0,i.jsx)(n.em,{children:"Tuple[np.ndarray, np.ndarray]"})," - 2D list where the first index is a unique list of motif used and the second index is the motif usage in frames."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"n_cluster"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"np.ndarray"})," - List of motif usage frames with 0's where motifs weren't used (array with zero labels filled)."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"augment_motif_timeseries",children:"augment_motif_timeseries"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def augment_motif_timeseries(label: np.ndarray,\n                             n_cluster: int) -> Tuple[np.ndarray, np.ndarray]\n"})}),"\n",(0,i.jsx)(n.p,{children:"Augment motif time series by filling zero motifs."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"label"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - Original label array."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"n_cluster"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsx)(n.p,{children:"Tuple[np.ndarray, np.ndarray]: Augmented label array and indices of zero motifs."}),"\n",(0,i.jsx)(n.h4,{id:"get_labels",children:"get_labels"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_labels(cfg: dict, files: List[str], model_name: str, n_cluster: int,\n               parametrization: str) -> List[np.ndarray]\n"})}),"\n",(0,i.jsx)(n.p,{children:"Get cluster labels for given videos files."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cfg"})," ",(0,i.jsx)(n.em,{children:"dict"})," - Configuration parameters."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"files"})," ",(0,i.jsx)(n.em,{children:"List[str]"})," - List of video files paths."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"model_name"})," ",(0,i.jsx)(n.em,{children:"str"})," - Model name."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"n_cluster"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"parametrization"})," ",(0,i.jsx)(n.em,{children:"str"})," - parametrization."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"List[np.ndarray]"})," - List of cluster labels for each file."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"get_community_label",children:"get_community_label"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_community_label(cfg: dict, files: List[str], model_name: str,\n                        n_cluster: int, parametrization: str) -> np.ndarray\n"})}),"\n",(0,i.jsx)(n.p,{children:"Get community labels for given files."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cfg"})," ",(0,i.jsx)(n.em,{children:"dict"})," - Configuration parameters."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"files"})," ",(0,i.jsx)(n.em,{children:"List[str]"})," - List of files paths."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"model_name"})," ",(0,i.jsx)(n.em,{children:"str"})," - Model name."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"n_cluster"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"parametrization"})," ",(0,i.jsx)(n.em,{children:"str"})," - parametrization."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"np.ndarray"})," - Array of community labels."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"compute_transition_matrices",children:"compute_transition_matrices"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def compute_transition_matrices(files: List[str], labels: List[np.ndarray],\n                                n_cluster: int) -> List[np.ndarray]\n"})}),"\n",(0,i.jsx)(n.p,{children:"Compute transition matrices for given files and labels."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"files"})," ",(0,i.jsx)(n.em,{children:"List[str]"})," - List of file paths."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"labels"})," ",(0,i.jsx)(n.em,{children:"List[np.ndarray]"})," - List of label arrays."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"n_cluster"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"List[np.ndarray]"})," - List of transition matrices."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"create_community_bag",children:"create_community_bag"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def create_community_bag(files: List[str], labels: List[np.ndarray],\n                         transition_matrices: List[np.ndarray], cut_tree: int,\n                         n_cluster: int) -> Tuple\n"})}),"\n",(0,i.jsx)(n.p,{children:"Create community bag for given files and labels (Markov chain to tree -> community detection)."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"files"})," ",(0,i.jsx)(n.em,{children:"List[str]"})," - List of file paths."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"labels"})," ",(0,i.jsx)(n.em,{children:"List[np.ndarray]"})," - List of label arrays."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"transition_matrices"})," ",(0,i.jsx)(n.em,{children:"List[np.ndarray]"})," - List of transition matrices."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cut_tree"})," ",(0,i.jsx)(n.em,{children:"int"})," - Cut line for tree."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"n_cluster"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"Tuple"})," - Tuple containing list of community bags and list of trees."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"create_cohort_community_bag",children:"create_cohort_community_bag"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def create_cohort_community_bag(files: List[str], labels: List[np.ndarray],\n                                trans_mat_full: np.ndarray, cut_tree: int,\n                                n_cluster: int) -> Tuple\n"})}),"\n",(0,i.jsx)(n.p,{children:"Create cohort community bag for given labels, transition matrix, cut tree, and number of clusters.\n(markov chain to tree -> community detection)"}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"files"})," ",(0,i.jsx)(n.em,{children:"List[str]"})," - List of files paths (deprecated)."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"labels"})," ",(0,i.jsx)(n.em,{children:"List[np.ndarray]"})," - List of label arrays."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"trans_mat_full"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - Full transition matrix."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cut_tree"})," ",(0,i.jsx)(n.em,{children:"int"})," - Cut line for tree."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"n_cluster"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"Tuple"})," - Tuple containing list of community bags and list of trees."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"get_community_labels",children:"get_community_labels"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_community_labels(\n        files: List[str], labels: List[np.ndarray],\n        communities_all: List[List[List[int]]]) -> List[np.ndarray]\n"})}),"\n",(0,i.jsx)(n.p,{children:"Transform kmeans parameterized latent vector into communities. Get community labels for given files and community bags."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"files"})," ",(0,i.jsx)(n.em,{children:"List[str]"})," - List of file paths."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"labels"})," ",(0,i.jsx)(n.em,{children:"List[np.ndarray]"})," - List of label arrays."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"communities_all"})," ",(0,i.jsx)(n.em,{children:"List[List[List[int]]]"})," - List of community bags."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"List[np.ndarray]"})," - List of community labels for each file."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"get_cohort_community_labels",children:"get_cohort_community_labels"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_cohort_community_labels(\n        files: List[str], labels: List[np.ndarray],\n        communities_all: List[List[List[int]]]) -> List[np.ndarray]\n"})}),"\n",(0,i.jsx)(n.p,{children:"Transform kmeans parameterized latent vector into communities. Get cohort community labels for given labels, and community bags."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"files"})," ",(0,i.jsx)(n.em,{children:"List[str], deprecated"})," - List of file paths."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"labels"})," ",(0,i.jsx)(n.em,{children:"List[np.ndarray]"})," - List of label arrays."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"communities_all"})," ",(0,i.jsx)(n.em,{children:"List[List[List[int]]]"})," - List of community bags."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"List[np.ndarray]"})," - List of cohort community labels for each file."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"umap_embedding",children:"umap_embedding"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def umap_embedding(cfg: dict, file: str, model_name: str, n_cluster: int,\n                   parametrization: str) -> np.ndarray\n"})}),"\n",(0,i.jsx)(n.p,{children:"Perform UMAP embedding for given file and parameters."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cfg"})," ",(0,i.jsx)(n.em,{children:"dict"})," - Configuration parameters."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"file"})," ",(0,i.jsx)(n.em,{children:"str"})," - File path."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"model_name"})," ",(0,i.jsx)(n.em,{children:"str"})," - Model name."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"n_cluster"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"parametrization"})," ",(0,i.jsx)(n.em,{children:"str"})," - parametrization."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"np.ndarray"})," - UMAP embedding."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"umap_vis",children:"umap_vis"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def umap_vis(cfg: dict, file: str, embed: np.ndarray,\n             community_labels_all: np.ndarray) -> None\n"})}),"\n",(0,i.jsx)(n.p,{children:"Create plotly visualizaton of UMAP embedding."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cfg"})," ",(0,i.jsx)(n.em,{children:"dict"})," - Configuration parameters."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"file"})," ",(0,i.jsx)(n.em,{children:"str"})," - File path."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"embed"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - UMAP embedding."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"community_labels_all"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - Community labels."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsx)(n.p,{children:"None"}),"\n",(0,i.jsx)(n.h4,{id:"community",children:"community"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def community(config: str,\n              cohort: bool = True,\n              show_umap: bool = False,\n              cut_tree: int = None) -> None\n"})}),"\n",(0,i.jsx)(n.p,{children:"Perform community analysis."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"config"})," ",(0,i.jsx)(n.em,{children:"str"})," - Path to the configuration file."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cohort"})," ",(0,i.jsx)(n.em,{children:"bool, optional"})," - Flag indicating cohort analysis. Defaults to True."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"show_umap"})," ",(0,i.jsx)(n.em,{children:"bool, optional"})," - Flag indicating weather to show UMAP visualization. Defaults to False."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cut_tree"})," ",(0,i.jsx)(n.em,{children:"int, optional"})," - Cut line for tree. Defaults to None."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsx)(n.p,{children:"None"})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(o,{...e})}):o(e)}},8453:(e,n,s)=>{s.d(n,{R:()=>t,x:()=>c});var i=s(6540);const r={},l=i.createContext(r);function t(e){const n=i.useContext(l);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:t(e.components),i.createElement(l.Provider,{value:n},e.children)}}}]);