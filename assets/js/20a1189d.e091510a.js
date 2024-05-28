"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[3567],{3072:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>o,contentTitle:()=>l,default:()=>h,frontMatter:()=>t,metadata:()=>c,toc:()=>a});var r=i(4848),s=i(8453);const t={sidebar_label:"gif_creator",title:"vame.analysis.gif_creator"},l=void 0,c={id:"reference/vame/analysis/gif_creator",title:"vame.analysis.gif_creator",description:"Variational Animal Motion Embedding 1.0-alpha Toolbox",source:"@site/docs/reference/vame/analysis/gif_creator.md",sourceDirName:"reference/vame/analysis",slug:"/reference/vame/analysis/gif_creator",permalink:"/VAME/docs/reference/vame/analysis/gif_creator",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"gif_creator",title:"vame.analysis.gif_creator"},sidebar:"docsSidebar",previous:{title:"generative_functions",permalink:"/VAME/docs/reference/vame/analysis/generative_functions"},next:{title:"pose_segmentation",permalink:"/VAME/docs/reference/vame/analysis/pose_segmentation"}},o={},a=[{value:"create_video",id:"create_video",level:4},{value:"gif",id:"gif",level:4}];function d(e){const n={a:"a",code:"code",em:"em",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.p,{children:"Variational Animal Motion Embedding 1.0-alpha Toolbox\n\xa9 K. Luxem & P. Bauer, Department of Cellular Neuroscience\nLeibniz Institute for Neurobiology, Magdeburg, Germany"}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.a,{href:"https://github.com/LINCellularNeuroscience/VAME",children:"https://github.com/LINCellularNeuroscience/VAME"}),"\nLicensed under GNU General Public License v3.0"]}),"\n",(0,r.jsx)(n.h4,{id:"create_video",children:"create_video"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def create_video(path_to_file: str, file: str, embed: np.ndarray,\n                 clabel: np.ndarray, frames: List[np.ndarray], start: int,\n                 length: int, max_lag: int, num_points: int) -> None\n"})}),"\n",(0,r.jsx)(n.p,{children:"Create video frames for the given embedding."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"path_to_file"})," ",(0,r.jsx)(n.em,{children:"str"})," - Path to the file."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"file"})," ",(0,r.jsx)(n.em,{children:"str"})," - File name."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"embed"})," ",(0,r.jsx)(n.em,{children:"np.ndarray"})," - Embedding array."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"clabel"})," ",(0,r.jsx)(n.em,{children:"np.ndarray"})," - Cluster labels."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"frames"})," ",(0,r.jsx)(n.em,{children:"List[np.ndarray]"})," - List of frames."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"start"})," ",(0,r.jsx)(n.em,{children:"int"})," - Starting index."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"length"})," ",(0,r.jsx)(n.em,{children:"int"})," - Length of the video."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"max_lag"})," ",(0,r.jsx)(n.em,{children:"int"})," - Maximum lag."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"num_points"})," ",(0,r.jsx)(n.em,{children:"int"})," - Number of points."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsx)(n.p,{children:"None"}),"\n",(0,r.jsx)(n.h4,{id:"gif",children:"gif"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def gif(\n    config: str,\n    pose_ref_index: int,\n    subtract_background: bool = True,\n    start: int = None,\n    length: int = 500,\n    max_lag: int = 30,\n    label: str = 'community',\n    file_format: str = '.mp4',\n    crop_size: Tuple[int, int] = (300, 300)) -> None\n"})}),"\n",(0,r.jsx)(n.p,{children:"Create a GIF from the given configuration."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"config"})," ",(0,r.jsx)(n.em,{children:"str"})," - Path to the configuration file."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"pose_ref_index"})," ",(0,r.jsx)(n.em,{children:"int"})," - Pose reference index."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"subtract_background"})," ",(0,r.jsx)(n.em,{children:"bool, optional"})," - Whether to subtract background. Defaults to True."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"start"})," ",(0,r.jsx)(n.em,{children:"int, optional"})," - Starting index. Defaults to None."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"length"})," ",(0,r.jsx)(n.em,{children:"int, optional"})," - Length of the video. Defaults to 500."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"max_lag"})," ",(0,r.jsx)(n.em,{children:"int, optional"})," - Maximum lag. Defaults to 30."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"label"})," ",(0,r.jsx)(n.em,{children:"str, optional"})," - Label type. Defaults to 'community'."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"file_format"})," ",(0,r.jsx)(n.em,{children:"str, optional"})," - File format. Defaults to '.mp4'."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"crop_size"})," ",(0,r.jsx)(n.em,{children:"Tuple[int, int], optional"})," - Crop size. Defaults to (300,300)."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsx)(n.p,{children:"None"})]})}function h(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(d,{...e})}):d(e)}},8453:(e,n,i)=>{i.d(n,{R:()=>l,x:()=>c});var r=i(6540);const s={},t=r.createContext(s);function l(e){const n=r.useContext(t);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:l(e.components),r.createElement(t.Provider,{value:n},e.children)}}}]);