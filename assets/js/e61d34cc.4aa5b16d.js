"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2578],{1083:(n,e,s)=>{s.r(e),s.d(e,{assets:()=>o,contentTitle:()=>t,default:()=>h,frontMatter:()=>l,metadata:()=>c,toc:()=>a});var i=s(4848),r=s(8453);const l={sidebar_label:"community_analysis",title:"analysis.community_analysis"},t=void 0,c={id:"reference/analysis/community_analysis",title:"analysis.community_analysis",description:"logger\\_config",source:"@site/docs/reference/analysis/community_analysis.md",sourceDirName:"reference/analysis",slug:"/reference/analysis/community_analysis",permalink:"/VAME/docs/reference/analysis/community_analysis",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"community_analysis",title:"analysis.community_analysis"},sidebar:"docsSidebar",previous:{title:"FAQ",permalink:"/VAME/docs/faq"},next:{title:"generative_functions",permalink:"/VAME/docs/reference/analysis/generative_functions"}},o={},a=[{value:"logger_config",id:"logger_config",level:4},{value:"logger",id:"logger",level:4},{value:"get_adjacency_matrix",id:"get_adjacency_matrix",level:4},{value:"get_transition_matrix",id:"get_transition_matrix",level:4},{value:"fill_motifs_with_zero_counts",id:"fill_motifs_with_zero_counts",level:4},{value:"augment_motif_timeseries",id:"augment_motif_timeseries",level:4},{value:"get_motif_labels",id:"get_motif_labels",level:4},{value:"compute_transition_matrices",id:"compute_transition_matrices",level:4},{value:"create_cohort_community_bag",id:"create_cohort_community_bag",level:4},{value:"get_cohort_community_labels",id:"get_cohort_community_labels",level:4},{value:"save_cohort_community_labels_per_file",id:"save_cohort_community_labels_per_file",level:4},{value:"community",id:"community",level:4}];function d(n){const e={code:"code",h4:"h4",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...n.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(e.h4,{id:"logger_config",children:"logger_config"}),"\n",(0,i.jsx)(e.h4,{id:"logger",children:"logger"}),"\n",(0,i.jsx)(e.h4,{id:"get_adjacency_matrix",children:"get_adjacency_matrix"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"def get_adjacency_matrix(\n        labels: np.ndarray,\n        n_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]\n"})}),"\n",(0,i.jsx)(e.p,{children:"Calculate the adjacency matrix, transition matrix, and temporal matrix."}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"labels"})," (",(0,i.jsx)(e.code,{children:"np.ndarray"}),"): Array of cluster labels."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"n_clusters"})," (",(0,i.jsx)(e.code,{children:"int"}),"): Number of clusters."]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.code,{children:"Tuple[np.ndarray, np.ndarray, np.ndarray]"}),": Tuple containing: adjacency matrix, transition matrix, and temporal matrix."]}),"\n"]}),"\n",(0,i.jsx)(e.h4,{id:"get_transition_matrix",children:"get_transition_matrix"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"def get_transition_matrix(adjacency_matrix: np.ndarray,\n                          threshold: float = 0.0) -> np.ndarray\n"})}),"\n",(0,i.jsx)(e.p,{children:"Compute the transition matrix from the adjacency matrix."}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"adjacency_matrix"})," (",(0,i.jsx)(e.code,{children:"np.ndarray"}),"): Adjacency matrix."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"threshold"})," (",(0,i.jsx)(e.code,{children:"float, optional"}),"): Threshold for considering transitions. Defaults to 0.0."]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.code,{children:"np.ndarray"}),": Transition matrix."]}),"\n"]}),"\n",(0,i.jsx)(e.h4,{id:"fill_motifs_with_zero_counts",children:"fill_motifs_with_zero_counts"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"def fill_motifs_with_zero_counts(unique_motif_labels: np.ndarray,\n                                 motif_counts: np.ndarray,\n                                 n_clusters: int) -> np.ndarray\n"})}),"\n",(0,i.jsx)(e.p,{children:"Find motifs that never occur in the dataset, and fill the motif_counts array with zeros for those motifs.\nExample 1:"}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"unique_motif_labels = [0, 1, 3, 4]"}),"\n",(0,i.jsx)(e.li,{children:"motif_counts = [10, 20, 30, 40],"}),"\n",(0,i.jsx)(e.li,{children:"n_clusters = 5"}),"\n",(0,i.jsx)(e.li,{children:"the function will return [10, 20, 0, 30, 40].\nExample 2:"}),"\n",(0,i.jsx)(e.li,{children:"unique_motif_labels = [0, 1, 3, 4]"}),"\n",(0,i.jsx)(e.li,{children:"motif_counts = [10, 20, 30, 40],"}),"\n",(0,i.jsx)(e.li,{children:"n_clusters = 6"}),"\n",(0,i.jsx)(e.li,{children:"the function will return [10, 20, 0, 30, 40, 0]."}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"unique_motif_labels"})," (",(0,i.jsx)(e.code,{children:"np.ndarray"}),"): Array of unique motif labels."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"motif_counts"})," (",(0,i.jsx)(e.code,{children:"np.ndarray"}),"): Array of motif counts (in number of frames)."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"n_clusters"})," (",(0,i.jsx)(e.code,{children:"int"}),"): Number of clusters."]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.code,{children:"np.ndarray"}),": List of motif counts (in number of frame) with 0's for motifs that never happened."]}),"\n"]}),"\n",(0,i.jsx)(e.h4,{id:"augment_motif_timeseries",children:"augment_motif_timeseries"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"def augment_motif_timeseries(labels: np.ndarray,\n                             n_clusters: int) -> Tuple[np.ndarray, np.ndarray]\n"})}),"\n",(0,i.jsx)(e.p,{children:"Augment motif time series by filling zero motifs."}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"labels"})," (",(0,i.jsx)(e.code,{children:"np.ndarray"}),"): Original array of labels."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"n_clusters"})," (",(0,i.jsx)(e.code,{children:"int"}),"): Number of clusters."]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.code,{children:"Tuple[np.ndarray, np.ndarray]"}),": Tuple with:","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"Array of labels augmented with motifs that never occurred, artificially inputed\nat the end of the original labels array"}),"\n",(0,i.jsx)(e.li,{children:"Indices of the motifs that never occurred."}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(e.h4,{id:"get_motif_labels",children:"get_motif_labels"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"def get_motif_labels(config: dict, sessions: List[str], model_name: str,\n                     n_clusters: int,\n                     segmentation_algorithm: str) -> np.ndarray\n"})}),"\n",(0,i.jsx)(e.p,{children:"Get motif labels for given files."}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"config"})," (",(0,i.jsx)(e.code,{children:"dict"}),"): Configuration parameters."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"sessions"})," (",(0,i.jsx)(e.code,{children:"List[str]"}),"): List of session names."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"model_name"})," (",(0,i.jsx)(e.code,{children:"str"}),"): Model name."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"n_clusters"})," (",(0,i.jsx)(e.code,{children:"int"}),"): Number of clusters."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"segmentation_algorithm"})," (",(0,i.jsx)(e.code,{children:"str"}),"): Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'."]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.code,{children:"np.ndarray"}),": Array of community labels (integers)."]}),"\n"]}),"\n",(0,i.jsx)(e.h4,{id:"compute_transition_matrices",children:"compute_transition_matrices"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"def compute_transition_matrices(files: List[str], labels: List[np.ndarray],\n                                n_clusters: int) -> List[np.ndarray]\n"})}),"\n",(0,i.jsx)(e.p,{children:"Compute transition matrices for given files and labels."}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"files"})," (",(0,i.jsx)(e.code,{children:"List[str]"}),"): List of file paths."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"labels"})," (",(0,i.jsx)(e.code,{children:"List[np.ndarray]"}),"): List of label arrays."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"n_clusters"})," (",(0,i.jsx)(e.code,{children:"int"}),"): Number of clusters."]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.code,{children:"List[np.ndarray]:"}),": List of transition matrices."]}),"\n"]}),"\n",(0,i.jsx)(e.h4,{id:"create_cohort_community_bag",children:"create_cohort_community_bag"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"def create_cohort_community_bag(motif_labels: List[np.ndarray],\n                                trans_mat_full: np.ndarray,\n                                cut_tree: int | None, n_clusters: int) -> list\n"})}),"\n",(0,i.jsx)(e.p,{children:"Create cohort community bag for given motif labels, transition matrix,\ncut tree, and number of clusters. (markov chain to tree -> community detection)"}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"motif_labels"})," (",(0,i.jsx)(e.code,{children:"List[np.ndarray]"}),"): List of motif label arrays."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"trans_mat_full"})," (",(0,i.jsx)(e.code,{children:"np.ndarray"}),"): Full transition matrix."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"cut_tree"})," (",(0,i.jsx)(e.code,{children:"int | None"}),"): Cut line for tree."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"n_clusters"})," (",(0,i.jsx)(e.code,{children:"int"}),"): Number of clusters."]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.code,{children:"List"}),": List of community bags."]}),"\n"]}),"\n",(0,i.jsx)(e.h4,{id:"get_cohort_community_labels",children:"get_cohort_community_labels"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"def get_cohort_community_labels(\n        motif_labels: List[np.ndarray],\n        cohort_community_bag: list,\n        median_filter_size: int = 7) -> List[np.ndarray]\n"})}),"\n",(0,i.jsx)(e.p,{children:"Transform kmeans/hmm parameterized latent vector motifs into communities.\nGet cohort community labels for given labels, and community bags."}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"labels"})," (",(0,i.jsx)(e.code,{children:"List[np.ndarray]"}),"): List of label arrays."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"cohort_community_bag"})," (",(0,i.jsx)(e.code,{children:"np.ndarray"}),"): List of community bags. Dimensions: (n_communities, n_clusters_in_community)"]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"median_filter_size"})," (",(0,i.jsx)(e.code,{children:"int, optional"}),"): Size of the median filter, in number of frames. Defaults to 7."]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.code,{children:"List[np.ndarray]"}),": List of cohort community labels for each file."]}),"\n"]}),"\n",(0,i.jsx)(e.h4,{id:"save_cohort_community_labels_per_file",children:"save_cohort_community_labels_per_file"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"def save_cohort_community_labels_per_file(config: dict, sessions: List[str],\n                                          model_name: str, n_clusters: int,\n                                          segmentation_algorithm: str,\n                                          cohort_community_bag: list) -> None\n"})}),"\n",(0,i.jsx)(e.h4,{id:"community",children:"community"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"@save_state(model=CommunityFunctionSchema)\ndef community(config: dict,\n              segmentation_algorithm: SegmentationAlgorithms,\n              cohort: bool = True,\n              cut_tree: int | None = None,\n              save_logs: bool = False) -> None\n"})}),"\n",(0,i.jsx)(e.p,{children:'Perform community analysis.\nFills in the values in the "community" key of the states.json file.\nSaves results files at:'}),"\n",(0,i.jsxs)(e.ol,{children:["\n",(0,i.jsx)(e.li,{children:"If cohort is True:"}),"\n"]}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["project_name/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["results/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["community_cohort/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["segmentation_algorithm-n_clusters/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"cohort_community_bag.npy"}),"\n",(0,i.jsx)(e.li,{children:"cohort_community_label.npy"}),"\n",(0,i.jsx)(e.li,{children:"cohort_segmentation_algorithm_label.npy"}),"\n",(0,i.jsx)(e.li,{children:"cohort_transition_matrix.npy"}),"\n",(0,i.jsx)(e.li,{children:"hierarchy.pkl"}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(e.li,{children:["file_name/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["model_name/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["segmentation_algorithm-n_clusters/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["community/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"cohort_community_label_file_name.npy"}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(e.ol,{start:"2",children:["\n",(0,i.jsx)(e.li,{children:"If cohort is False:"}),"\n"]}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["project_name/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["results/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["file_name/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["model_name/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["segmentation_algorithm-n_clusters/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["community/","\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"transition_matrix_file_name.npy"}),"\n",(0,i.jsx)(e.li,{children:"community_label_file_name.npy"}),"\n",(0,i.jsx)(e.li,{children:"hierarchy_file_name.pkl"}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"config"})," (",(0,i.jsx)(e.code,{children:"dict"}),"): Configuration parameters."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"segmentation_algorithm"})," (",(0,i.jsx)(e.code,{children:"SegmentationAlgorithms"}),"): Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"cohort"})," (",(0,i.jsx)(e.code,{children:"bool, optional"}),"): Flag indicating cohort analysis. Defaults to True."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"cut_tree"})," (",(0,i.jsx)(e.code,{children:"int, optional"}),"): Cut line for tree. Defaults to None."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"save_logs"})," (",(0,i.jsx)(e.code,{children:"bool, optional"}),"): Flag indicating whether to save logs. Defaults to False."]}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:(0,i.jsx)(e.code,{children:"None"})}),"\n"]})]})}function h(n={}){const{wrapper:e}={...(0,r.R)(),...n.components};return e?(0,i.jsx)(e,{...n,children:(0,i.jsx)(d,{...n})}):d(n)}},8453:(n,e,s)=>{s.d(e,{R:()=>t,x:()=>c});var i=s(6540);const r={},l=i.createContext(r);function t(n){const e=i.useContext(l);return i.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function c(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(r):n.components||r:t(n.components),i.createElement(l.Provider,{value:e},n.children)}}}]);