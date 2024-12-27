"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[9888],{1632:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>a,contentTitle:()=>s,default:()=>m,frontMatter:()=>l,metadata:()=>o,toc:()=>c});var r=i(4848),t=i(8453);const l={sidebar_label:"gif_pose_helper",title:"vame.util.gif_pose_helper"},s=void 0,o={id:"reference/vame/util/gif_pose_helper",title:"vame.util.gif_pose_helper",description:"Variational Animal Motion Embedding 1.0-alpha Toolbox",source:"@site/docs/reference/vame/util/gif_pose_helper.md",sourceDirName:"reference/vame/util",slug:"/reference/vame/util/gif_pose_helper",permalink:"/VAME/docs/reference/vame/util/gif_pose_helper",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"gif_pose_helper",title:"vame.util.gif_pose_helper"},sidebar:"docsSidebar",previous:{title:"data_manipulation",permalink:"/VAME/docs/reference/vame/util/data_manipulation"},next:{title:"model_util",permalink:"/VAME/docs/reference/vame/util/model_util"}},a={},c=[{value:"get_animal_frames",id:"get_animal_frames",level:4}];function d(e){const n={a:"a",code:"code",em:"em",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,t.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.p,{children:"Variational Animal Motion Embedding 1.0-alpha Toolbox\n\xa9 K. Luxem & P. Bauer, Department of Cellular Neuroscience\nLeibniz Institute for Neurobiology, Magdeburg, Germany"}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.a,{href:"https://github.com/LINCellularNeuroscience/VAME",children:"https://github.com/LINCellularNeuroscience/VAME"}),"\nLicensed under GNU General Public License v3.0"]}),"\n",(0,r.jsx)(n.h4,{id:"get_animal_frames",children:"get_animal_frames"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def get_animal_frames(\n    cfg: dict,\n    filename: str,\n    pose_ref_index: list,\n    start: int,\n    length: int,\n    subtract_background: bool,\n    file_format: str = '.mp4',\n    crop_size: tuple = (300, 300)) -> list\n"})}),"\n",(0,r.jsx)(n.p,{children:"Extracts frames of an animal from a video file and returns them as a list."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"cfg"})," ",(0,r.jsx)(n.em,{children:"dict"})," - Configuration dictionary containing project information."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"filename"})," ",(0,r.jsx)(n.em,{children:"str"})," - Name of the video file."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"pose_ref_index"})," ",(0,r.jsx)(n.em,{children:"list"})," - List of reference coordinate indices for alignment."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"start"})," ",(0,r.jsx)(n.em,{children:"int"})," - Starting frame index."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"length"})," ",(0,r.jsx)(n.em,{children:"int"})," - Number of frames to extract."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"subtract_background"})," ",(0,r.jsx)(n.em,{children:"bool"})," - Whether to subtract background or not."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"file_format"})," ",(0,r.jsx)(n.em,{children:"str, optional"})," - Format of the video file. Defaults to '.mp4'."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"crop_size"})," ",(0,r.jsx)(n.em,{children:"tuple, optional"})," - Size of the cropped area. Defaults to (300, 300)."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"list"})," - List of extracted frames."]}),"\n"]})]})}function m(e={}){const{wrapper:n}={...(0,t.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(d,{...e})}):d(e)}},8453:(e,n,i)=>{i.d(n,{R:()=>s,x:()=>o});var r=i(6540);const t={},l=r.createContext(t);function s(e){const n=r.useContext(l);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:s(e.components),r.createElement(l.Provider,{value:n},e.children)}}}]);