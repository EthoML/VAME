"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2977],{2547:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>c,contentTitle:()=>l,default:()=>h,frontMatter:()=>t,metadata:()=>o,toc:()=>d});var i=s(4848),r=s(8453);const t={sidebar_label:"pose_segmentation",title:"analysis.pose_segmentation"},l=void 0,o={id:"reference/analysis/pose_segmentation",title:"analysis.pose_segmentation",description:"logger\\_config",source:"@site/docs/reference/analysis/pose_segmentation.md",sourceDirName:"reference/analysis",slug:"/reference/analysis/pose_segmentation",permalink:"/VAME/docs/reference/analysis/pose_segmentation",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"pose_segmentation",title:"analysis.pose_segmentation"},sidebar:"docsSidebar",previous:{title:"gif_creator",permalink:"/VAME/docs/reference/analysis/gif_creator"},next:{title:"tree_hierarchy",permalink:"/VAME/docs/reference/analysis/tree_hierarchy"}},c={},d=[{value:"logger_config",id:"logger_config",level:4},{value:"logger",id:"logger",level:4},{value:"embedd_latent_vectors",id:"embedd_latent_vectors",level:4},{value:"get_latent_vectors",id:"get_latent_vectors",level:4},{value:"get_motif_usage",id:"get_motif_usage",level:4},{value:"save_session_data",id:"save_session_data",level:4},{value:"same_segmentation",id:"same_segmentation",level:4},{value:"individual_segmentation",id:"individual_segmentation",level:4},{value:"segment_session",id:"segment_session",level:4}];function a(e){const n={code:"code",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.h4,{id:"logger_config",children:"logger_config"}),"\n",(0,i.jsx)(n.h4,{id:"logger",children:"logger"}),"\n",(0,i.jsx)(n.h4,{id:"embedd_latent_vectors",children:"embedd_latent_vectors"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'def embedd_latent_vectors(\n        cfg: dict,\n        sessions: List[str],\n        model: RNN_VAE,\n        fixed: bool,\n        read_from_variable: str = "position_processed",\n        tqdm_stream: Union[TqdmToLogger, None] = None) -> List[np.ndarray]\n'})}),"\n",(0,i.jsx)(n.p,{children:"Embed latent vectors for the given files using the VAME model."}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"cfg"})," (",(0,i.jsx)(n.code,{children:"dict"}),"): Configuration dictionary."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"sessions"})," (",(0,i.jsx)(n.code,{children:"List[str]"}),"): List of session names."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"model"})," (",(0,i.jsx)(n.code,{children:"RNN_VAE"}),"): VAME model."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"fixed"})," (",(0,i.jsx)(n.code,{children:"bool"}),"): Whether the model is fixed."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"tqdm_stream"})," (",(0,i.jsx)(n.code,{children:"TqdmToLogger, optional"}),"): TQDM Stream to redirect the tqdm output to logger."]}),"\n"]}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"List[np.ndarray]"}),": List of latent vectors for each file."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"get_latent_vectors",children:"get_latent_vectors"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_latent_vectors(project_path: str, sessions: list, model_name: str, seg,\n                       n_clusters: int) -> List\n"})}),"\n",(0,i.jsx)(n.p,{children:"Gets all the latent vectors from each session into one list"}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"project_path: str"}),": Path to vame project folder"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"session: list"}),": List of sessions"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"model_name: str"}),": Name of model"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"seg: str"}),": Type of segmentation algorithm"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"n_clusters"})," (",(0,i.jsx)(n.code,{children:"int"}),"): Number of clusters."]}),"\n"]}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"List"}),": List of session latent vectors"]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"get_motif_usage",children:"get_motif_usage"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_motif_usage(session_labels: np.ndarray, n_clusters: int) -> np.ndarray\n"})}),"\n",(0,i.jsx)(n.p,{children:"Count motif usage from session label array."}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"session_labels"})," (",(0,i.jsx)(n.code,{children:"np.ndarray"}),"): Array of session labels."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"n_clusters"})," (",(0,i.jsx)(n.code,{children:"int"}),"): Number of clusters."]}),"\n"]}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"np.ndarray"}),": Array of motif usage counts."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"save_session_data",children:"save_session_data"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def save_session_data(project_path: str, session: int, model_name: str,\n                      label: np.ndarray, cluster_center: np.ndarray,\n                      latent_vector: np.ndarray, motif_usage: np.ndarray,\n                      n_clusters: int, segmentation_algorithm: str)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Saves pose segmentation data for given session."}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"project_path: str"}),": Path to the vame project folder."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"session: int"}),": Session of interest to segment."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"model_name: str"}),": Name of model"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"label: np.ndarray"}),": Array of the session's motif labels."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"cluster_center: np.ndarray"}),": Array of the session's kmeans cluster centers location in the latent space."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"latent_vector: np.ndarray,"}),": Array of the session's latent vectors."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"motif_usage: np.ndarray"}),": Array of the session's motif usage counts."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"n_clusters"})," (",(0,i.jsx)(n.code,{children:"int"}),"): Number of clusters."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"segmentation_algorithm: str"}),": Type of segmentation method, either 'kmeans or 'hmm'."]}),"\n"]}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsx)(n.li,{children:(0,i.jsx)(n.code,{children:"None"})}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"same_segmentation",children:"same_segmentation"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def same_segmentation(\n    cfg: dict, sessions: List[str], latent_vectors: List[np.ndarray],\n    n_clusters: int, segmentation_algorithm: str\n) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]\n"})}),"\n",(0,i.jsx)(n.p,{children:"Apply the same segmentation to all animals."}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"cfg"})," (",(0,i.jsx)(n.code,{children:"dict"}),"): Configuration dictionary."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"sessions"})," (",(0,i.jsx)(n.code,{children:"List[str]"}),"): List of session names."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"latent_vectors"})," (",(0,i.jsx)(n.code,{children:"List[np.ndarray]"}),"): List of latent vector arrays."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"n_clusters"})," (",(0,i.jsx)(n.code,{children:"int"}),"): Number of clusters."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"segmentation_algorithm"})," (",(0,i.jsx)(n.code,{children:"str"}),"): Segmentation algorithm."]}),"\n"]}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"Tuple"}),": Tuple of labels, cluster centers, and motif usages."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"individual_segmentation",children:"individual_segmentation"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def individual_segmentation(cfg: dict, sessions: List[str],\n                            latent_vectors: List[np.ndarray],\n                            n_clusters: int) -> Tuple\n"})}),"\n",(0,i.jsx)(n.p,{children:"Apply individual segmentation to each session."}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"cfg"})," (",(0,i.jsx)(n.code,{children:"dict"}),"): Configuration dictionary."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"sessions"})," (",(0,i.jsx)(n.code,{children:"List[str]"}),"): List of session names."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"latent_vectors"})," (",(0,i.jsx)(n.code,{children:"List[np.ndarray]"}),"): List of latent vector arrays."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"n_clusters"})," (",(0,i.jsx)(n.code,{children:"int"}),"): Number of clusters."]}),"\n"]}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"Tuple"}),": Tuple of labels, cluster centers, and motif usages."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"segment_session",children:"segment_session"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"@save_state(model=SegmentSessionFunctionSchema)\ndef segment_session(config: dict, save_logs: bool = False) -> None\n"})}),"\n",(0,i.jsx)(n.p,{children:'Perform pose segmentation using the VAME model.\nFills in the values in the "segment_session" key of the states.json file.\nCreates files at:'}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["project_name/","\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["results/","\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsx)(n.li,{children:"hmm_trained.pkl"}),"\n",(0,i.jsxs)(n.li,{children:["session/","\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["model_name/","\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["hmm-n_clusters/","\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsx)(n.li,{children:"latent_vector_session.npy"}),"\n",(0,i.jsx)(n.li,{children:"motif_usage_session.npy"}),"\n",(0,i.jsx)(n.li,{children:"n_cluster_label_session.npy"}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["kmeans-n_clusters/","\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsx)(n.li,{children:"latent_vector_session.npy"}),"\n",(0,i.jsx)(n.li,{children:"motif_usage_session.npy"}),"\n",(0,i.jsx)(n.li,{children:"n_cluster_label_session.npy"}),"\n",(0,i.jsx)(n.li,{children:"cluster_center_session.npy"}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.p,{children:"latent_vector_session.npy contains the projection of the data into the latent space,\nfor each frame of the video. Dimmentions: (n_frames, n_latent_features)"}),"\n",(0,i.jsx)(n.p,{children:"motif_usage_session.npy contains the number of times each motif was used in the video.\nDimmentions: (n_motifs,)"}),"\n",(0,i.jsx)(n.p,{children:"n_cluster_label_session.npy contains the label of the cluster assigned to each frame.\nDimmentions: (n_frames,)"}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"config"})," (",(0,i.jsx)(n.code,{children:"dict"}),"): Configuration dictionary."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"save_logs"})," (",(0,i.jsx)(n.code,{children:"bool, optional"}),"): Whether to save logs, by default False."]}),"\n"]}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsx)(n.li,{children:(0,i.jsx)(n.code,{children:"None"})}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(a,{...e})}):a(e)}},8453:(e,n,s)=>{s.d(n,{R:()=>l,x:()=>o});var i=s(6540);const r={},t=i.createContext(r);function l(e){const n=i.useContext(t);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:l(e.components),i.createElement(t.Provider,{value:n},e.children)}}}]);