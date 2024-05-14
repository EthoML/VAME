"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[4077],{689:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>c,contentTitle:()=>o,default:()=>h,frontMatter:()=>t,metadata:()=>l,toc:()=>d});var s=i(4848),r=i(8453);const t={sidebar_label:"videowriter",title:"vame.analysis.videowriter"},o=void 0,l={id:"reference/vame/analysis/videowriter",title:"vame.analysis.videowriter",description:"Variational Animal Motion Embedding 1.0-alpha Toolbox",source:"@site/docs/reference/vame/analysis/videowriter.md",sourceDirName:"reference/vame/analysis",slug:"/reference/vame/analysis/videowriter",permalink:"/VAME/docs/reference/vame/analysis/videowriter",draft:!1,unlisted:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/reference/vame/analysis/videowriter.md",tags:[],version:"current",frontMatter:{sidebar_label:"videowriter",title:"vame.analysis.videowriter"},sidebar:"docsSidebar",previous:{title:"umap_visualization",permalink:"/VAME/docs/reference/vame/analysis/umap_visualization"},next:{title:"new",permalink:"/VAME/docs/reference/vame/initialize_project/new"}},c={},d=[{value:"get_cluster_vid",id:"get_cluster_vid",level:4},{value:"motif_videos",id:"motif_videos",level:4},{value:"community_videos",id:"community_videos",level:4}];function a(e){const n={a:"a",code:"code",em:"em",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.p,{children:"Variational Animal Motion Embedding 1.0-alpha Toolbox\n\xa9 K. Luxem & P. Bauer, Department of Cellular Neuroscience\nLeibniz Institute for Neurobiology, Magdeburg, Germany"}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.a,{href:"https://github.com/LINCellularNeuroscience/VAME",children:"https://github.com/LINCellularNeuroscience/VAME"}),"\nLicensed under GNU General Public License v3.0"]}),"\n",(0,s.jsx)(n.h4,{id:"get_cluster_vid",children:"get_cluster_vid"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def get_cluster_vid(cfg: dict, path_to_file: str, file: str, n_cluster: int,\n                    videoType: str, flag: str) -> None\n"})}),"\n",(0,s.jsx)(n.p,{children:"Generate cluster videos."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"cfg"})," ",(0,s.jsx)(n.em,{children:"dict"})," - Configuration parameters."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"path_to_file"})," ",(0,s.jsx)(n.em,{children:"str"})," - Path to the file."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"file"})," ",(0,s.jsx)(n.em,{children:"str"})," - Name of the file."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"n_cluster"})," ",(0,s.jsx)(n.em,{children:"int"})," - Number of clusters."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"videoType"})," ",(0,s.jsx)(n.em,{children:"str"})," - Type of video."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"flag"})," ",(0,s.jsx)(n.em,{children:"str"})," - Flag indicating the type of video (motif or community)."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:"None - Generate cluster videos and save them to fs on project folder."}),"\n",(0,s.jsx)(n.h4,{id:"motif_videos",children:"motif_videos"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def motif_videos(config: Union[str, Path], videoType: str = '.mp4') -> None\n"})}),"\n",(0,s.jsx)(n.p,{children:"Generate motif videos."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"config"})," ",(0,s.jsx)(n.em,{children:"Union[str, Path]"})," - Path to the configuration file."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"videoType"})," ",(0,s.jsx)(n.em,{children:"str, optional"})," - Type of video. Default is '.mp4'."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:"None - Generate motif videos and save them to filesystem on project cluster_videos folder."}),"\n",(0,s.jsx)(n.h4,{id:"community_videos",children:"community_videos"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def community_videos(config: Union[str, Path],\n                     videoType: str = '.mp4') -> None\n"})}),"\n",(0,s.jsx)(n.p,{children:"Generate community videos."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"config"})," ",(0,s.jsx)(n.em,{children:"Union[str, Path]"})," - Path to the configuration file."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"videoType"})," ",(0,s.jsx)(n.em,{children:"str, optional"})," - Type of video. Default is '.mp4'."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:"None - Generate community videos and save them to filesystem on project community_videos folder."})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(a,{...e})}):a(e)}},8453:(e,n,i)=>{i.d(n,{R:()=>o,x:()=>l});var s=i(6540);const r={},t=s.createContext(r);function o(e){const n=s.useContext(t);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:o(e.components),s.createElement(t.Provider,{value:n},e.children)}}}]);