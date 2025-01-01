"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[504],{3719:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>r,default:()=>u,frontMatter:()=>s,metadata:()=>l,toc:()=>a});var o=t(4848),i=t(8453);const s={sidebar_label:"csv_to_npy",title:"util.csv_to_npy"},r=void 0,l={id:"reference/util/csv_to_npy",title:"util.csv_to_npy",description:"logger\\_config",source:"@site/docs/reference/util/csv_to_npy.md",sourceDirName:"reference/util",slug:"/reference/util/csv_to_npy",permalink:"/VAME/docs/reference/util/csv_to_npy",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"csv_to_npy",title:"util.csv_to_npy"},sidebar:"docsSidebar",previous:{title:"cli",permalink:"/VAME/docs/reference/util/cli"},next:{title:"data_manipulation",permalink:"/VAME/docs/reference/util/data_manipulation"}},c={},a=[{value:"logger_config",id:"logger_config",level:4},{value:"logger",id:"logger",level:4},{value:"pose_to_numpy",id:"pose_to_numpy",level:4}];function d(e){const n={code:"code",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.h4,{id:"logger_config",children:"logger_config"}),"\n",(0,o.jsx)(n.h4,{id:"logger",children:"logger"}),"\n",(0,o.jsx)(n.h4,{id:"pose_to_numpy",children:"pose_to_numpy"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:"@save_state(model=PoseToNumpyFunctionSchema)\ndef pose_to_numpy(config: dict, save_logs=False) -> None\n"})}),"\n",(0,o.jsx)(n.p,{children:"Converts a pose-estimation.csv file to a numpy array.\nNote that this code is only useful for data which is a priori egocentric, i.e. head-fixed\nor otherwise restrained animals."}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Parameters"})}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"config"})," (",(0,o.jsx)(n.code,{children:"dict"}),"): Configuration dictionary."]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"save_logs"})," (",(0,o.jsx)(n.code,{children:"bool, optional"}),"): If True, the logs will be saved to a file, by default False."]}),"\n"]}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Raises"})}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.code,{children:"ValueError"}),": If the config.yaml file indicates that the data is not egocentric."]}),"\n"]})]})}function u(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(d,{...e})}):d(e)}},8453:(e,n,t)=>{t.d(n,{R:()=>r,x:()=>l});var o=t(6540);const i={},s=o.createContext(i);function r(e){const n=o.useContext(s);return o.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:r(e.components),o.createElement(s.Provider,{value:n},e.children)}}}]);