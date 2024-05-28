"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[9888],{1632:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>d,contentTitle:()=>t,default:()=>h,frontMatter:()=>l,metadata:()=>c,toc:()=>o});var i=r(4848),s=r(8453);const l={sidebar_label:"gif_pose_helper",title:"vame.util.gif_pose_helper"},t=void 0,c={id:"reference/vame/util/gif_pose_helper",title:"vame.util.gif_pose_helper",description:"Variational Animal Motion Embedding 1.0-alpha Toolbox",source:"@site/docs/reference/vame/util/gif_pose_helper.md",sourceDirName:"reference/vame/util",slug:"/reference/vame/util/gif_pose_helper",permalink:"/VAME/docs/reference/vame/util/gif_pose_helper",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"gif_pose_helper",title:"vame.util.gif_pose_helper"},sidebar:"docsSidebar",previous:{title:"csv_to_npy",permalink:"/VAME/docs/reference/vame/util/csv_to_npy"},next:{title:"FAQ",permalink:"/VAME/docs/faq"}},d={},o=[{value:"crop_and_flip",id:"crop_and_flip",level:4},{value:"background",id:"background",level:4},{value:"get_rotation_matrix",id:"get_rotation_matrix",level:4},{value:"nan_helper",id:"nan_helper",level:4},{value:"interpol",id:"interpol",level:4},{value:"get_animal_frames",id:"get_animal_frames",level:4}];function a(e){const n={a:"a",code:"code",em:"em",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.p,{children:"Variational Animal Motion Embedding 1.0-alpha Toolbox\n\xa9 K. Luxem & P. Bauer, Department of Cellular Neuroscience\nLeibniz Institute for Neurobiology, Magdeburg, Germany"}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.a,{href:"https://github.com/LINCellularNeuroscience/VAME",children:"https://github.com/LINCellularNeuroscience/VAME"}),"\nLicensed under GNU General Public License v3.0"]}),"\n",(0,i.jsx)(n.h4,{id:"crop_and_flip",children:"crop_and_flip"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def crop_and_flip(rect: tuple, src: np.ndarray, points: list,\n                  ref_index: list) -> tuple\n"})}),"\n",(0,i.jsx)(n.p,{children:"Crop and flip an image based on a rectangle and reference points."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"rect"})," ",(0,i.jsx)(n.em,{children:"tuple"})," - Tuple containing rectangle information (center, size, angle)."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"src"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - Source image to crop and flip."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"points"})," ",(0,i.jsx)(n.em,{children:"list"})," - List of points to be aligned."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"ref_index"})," ",(0,i.jsx)(n.em,{children:"list"})," - Reference indices for alignment."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"tuple"})," - Cropped and flipped image, shifted points."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"background",children:"background"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def background(path_to_file: str,\n               filename: str,\n               file_format: str = '.mp4',\n               num_frames: int = 1000) -> np.ndarray\n"})}),"\n",(0,i.jsx)(n.p,{children:"Compute background image from fixed camera."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"path_to_file"})," ",(0,i.jsx)(n.em,{children:"str"})," - Path to the directory containing the video files."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"filename"})," ",(0,i.jsx)(n.em,{children:"str"})," - Name of the video file."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"file_format"})," ",(0,i.jsx)(n.em,{children:"str, optional"})," - Format of the video file. Defaults to '.mp4'."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"num_frames"})," ",(0,i.jsx)(n.em,{children:"int, optional"})," - Number of frames to use for background computation. Defaults to 1000."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"np.ndarray"})," - Background image."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"get_rotation_matrix",children:"get_rotation_matrix"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_rotation_matrix(\n    adjacent: float, opposite: float,\n    crop_size: tuple = (300, 300)) -> np.ndarray\n"})}),"\n",(0,i.jsx)(n.p,{children:"Compute the rotation matrix based on the adjacent and opposite sides."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"adjacent"})," ",(0,i.jsx)(n.em,{children:"float"})," - Length of the adjacent side."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"opposite"})," ",(0,i.jsx)(n.em,{children:"float"})," - Length of the opposite side."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"crop_size"})," ",(0,i.jsx)(n.em,{children:"tuple, optional"})," - Size of the cropped area. Defaults to (300, 300)."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"np.ndarray"})," - Rotation matrix."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"nan_helper",children:"nan_helper"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def nan_helper(y: np.ndarray) -> tuple\n"})}),"\n",(0,i.jsx)(n.p,{children:"Helper function to find indices of NaN values."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"y"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - Input array."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"tuple"})," - Indices of NaN values."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"interpol",children:"interpol"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def interpol(arr: np.ndarray) -> np.ndarray\n"})}),"\n",(0,i.jsx)(n.p,{children:"Interpolates NaN values in the given array."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"arr"})," ",(0,i.jsx)(n.em,{children:"np.ndarray"})," - Input array with NaN values."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"np.ndarray"})," - Array with interpolated NaN values."]}),"\n"]}),"\n",(0,i.jsx)(n.h4,{id:"get_animal_frames",children:"get_animal_frames"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"def get_animal_frames(\n    cfg: dict,\n    filename: str,\n    pose_ref_index: list,\n    start: int,\n    length: int,\n    subtract_background: bool,\n    file_format: str = '.mp4',\n    crop_size: tuple = (300, 300)) -> list\n"})}),"\n",(0,i.jsx)(n.p,{children:"Extracts frames of an animal from a video file and returns them as a list."}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"cfg"})," ",(0,i.jsx)(n.em,{children:"dict"})," - Configuration dictionary containing project information."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"filename"})," ",(0,i.jsx)(n.em,{children:"str"})," - Name of the video file."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"pose_ref_index"})," ",(0,i.jsx)(n.em,{children:"list"})," - List of reference coordinate indices for alignment."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"start"})," ",(0,i.jsx)(n.em,{children:"int"})," - Starting frame index."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"length"})," ",(0,i.jsx)(n.em,{children:"int"})," - Number of frames to extract."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"subtract_background"})," ",(0,i.jsx)(n.em,{children:"bool"})," - Whether to subtract background or not."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"file_format"})," ",(0,i.jsx)(n.em,{children:"str, optional"})," - Format of the video file. Defaults to '.mp4'."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"crop_size"})," ",(0,i.jsx)(n.em,{children:"tuple, optional"})," - Size of the cropped area. Defaults to (300, 300)."]}),"\n"]}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"list"})," - List of extracted frames."]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(a,{...e})}):a(e)}},8453:(e,n,r)=>{r.d(n,{R:()=>t,x:()=>c});var i=r(6540);const s={},l=i.createContext(s);function t(e){const n=i.useContext(l);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:t(e.components),i.createElement(l.Provider,{value:n},e.children)}}}]);