"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[4693],{4333:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>o,contentTitle:()=>l,default:()=>h,frontMatter:()=>t,metadata:()=>c,toc:()=>a});var s=r(4848),i=r(8453);const t={sidebar_label:"tree_hierarchy",title:"vame.analysis.tree_hierarchy"},l=void 0,c={id:"reference/vame/analysis/tree_hierarchy",title:"vame.analysis.tree_hierarchy",description:"Variational Animal Motion Embedding 1.0-alpha Toolbox",source:"@site/docs/reference/vame/analysis/tree_hierarchy.md",sourceDirName:"reference/vame/analysis",slug:"/reference/vame/analysis/tree_hierarchy",permalink:"/undefined/docs/reference/vame/analysis/tree_hierarchy",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"tree_hierarchy",title:"vame.analysis.tree_hierarchy"},sidebar:"docsSidebar",previous:{title:"segment_behavior",permalink:"/undefined/docs/reference/vame/analysis/segment_behavior"},next:{title:"umap_visualization",permalink:"/undefined/docs/reference/vame/analysis/umap_visualization"}},o={},a=[{value:"hierarchy_pos",id:"hierarchy_pos",level:4},{value:"merge_func",id:"merge_func",level:4},{value:"graph_to_tree",id:"graph_to_tree",level:4},{value:"draw_tree",id:"draw_tree",level:4},{value:"traverse_tree",id:"traverse_tree",level:4},{value:"traverse_tree",id:"traverse_tree-1",level:4},{value:"traverse_tree_cutline",id:"traverse_tree_cutline",level:4}];function d(e){const n={a:"a",code:"code",em:"em",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.p,{children:"Variational Animal Motion Embedding 1.0-alpha Toolbox\n\xa9 K. Luxem & P. Bauer, Department of Cellular Neuroscience\nLeibniz Institute for Neurobiology, Magdeburg, Germany"}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.a,{href:"https://github.com/LINCellularNeuroscience/VAME",children:"https://github.com/LINCellularNeuroscience/VAME"}),"\nLicensed under GNU General Public License v3.0"]}),"\n",(0,s.jsx)(n.h4,{id:"hierarchy_pos",children:"hierarchy_pos"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def hierarchy_pos(G: nx.Graph,\n                  root: str = None,\n                  width: float = 0.5,\n                  vert_gap: float = 0.2,\n                  vert_loc: float = 0,\n                  xcenter: float = 0.5) -> Dict[str, Tuple[float, float]]\n"})}),"\n",(0,s.jsxs)(n.p,{children:["Positions nodes in a tree-like layout.\nRef: From Joel's answer at ",(0,s.jsx)(n.a,{href:"https://stackoverflow.com/a/29597209/2966723",children:"https://stackoverflow.com/a/29597209/2966723"}),"."]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"G"})," ",(0,s.jsx)(n.em,{children:"nx.Graph"})," - The input graph. Must be a tree."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"root"})," ",(0,s.jsx)(n.em,{children:"str, optional"})," - The root node of the tree. If None, the function selects a root node based on graph type."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"width"})," ",(0,s.jsx)(n.em,{children:"float, optional"})," - The horizontal space assigned to each level."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"vert_gap"})," ",(0,s.jsx)(n.em,{children:"float, optional"})," - The vertical gap between levels."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"vert_loc"})," ",(0,s.jsx)(n.em,{children:"float, optional"})," - The vertical location of the root node."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"xcenter"})," ",(0,s.jsx)(n.em,{children:"float, optional"})," - The horizontal location of the root node."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:"Dict[str, Tuple[float, float]]: A dictionary mapping node names to their positions (x, y)."}),"\n",(0,s.jsx)(n.h4,{id:"merge_func",children:"merge_func"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def merge_func(transition_matrix: np.ndarray, n_cluster: int,\n               motif_norm: np.ndarray,\n               merge_sel: int) -> Tuple[np.ndarray, np.ndarray]\n"})}),"\n",(0,s.jsx)(n.p,{children:"Merge nodes in a graph based on a selection criterion."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"transition_matrix"})," ",(0,s.jsx)(n.em,{children:"np.ndarray"})," - The transition matrix of the graph."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"n_cluster"})," ",(0,s.jsx)(n.em,{children:"int"})," - The number of clusters."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"motif_norm"})," ",(0,s.jsx)(n.em,{children:"np.ndarray"})," - The normalized motif matrix."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"merge_sel"})," ",(0,s.jsx)(n.em,{children:"int"})," - The merge selection criterion.","\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:"0: Merge nodes with highest transition probability."}),"\n",(0,s.jsx)(n.li,{children:"1: Merge nodes with lowest cost."}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Raises"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"ValueError"})," - If an invalid merge selection criterion is provided."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:"Tuple[np.ndarray, np.ndarray]: A tuple containing the merged nodes."}),"\n",(0,s.jsx)(n.h4,{id:"graph_to_tree",children:"graph_to_tree"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def graph_to_tree(motif_usage: np.ndarray,\n                  transition_matrix: np.ndarray,\n                  n_cluster: int,\n                  merge_sel: int = 1) -> nx.Graph\n"})}),"\n",(0,s.jsx)(n.p,{children:"Convert a graph to a tree."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"motif_usage"})," ",(0,s.jsx)(n.em,{children:"np.ndarray"})," - The motif usage matrix."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"transition_matrix"})," ",(0,s.jsx)(n.em,{children:"np.ndarray"})," - The transition matrix of the graph."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"n_cluster"})," ",(0,s.jsx)(n.em,{children:"int"})," - The number of clusters."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"merge_sel"})," ",(0,s.jsx)(n.em,{children:"int, optional"})," - The merge selection criterion. Defaults to 1.","\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:"0: Merge nodes with highest transition probability."}),"\n",(0,s.jsx)(n.li,{children:"1: Merge nodes with lowest cost."}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"nx.Graph"})," - The tree."]}),"\n"]}),"\n",(0,s.jsx)(n.h4,{id:"draw_tree",children:"draw_tree"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def draw_tree(T: nx.Graph) -> None\n"})}),"\n",(0,s.jsx)(n.p,{children:"Draw a tree."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"T"})," ",(0,s.jsx)(n.em,{children:"nx.Graph"})," - The tree to be drawn."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:"None"}),"\n",(0,s.jsx)(n.h4,{id:"traverse_tree",children:"traverse_tree"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def traverse_tree(T: nx.Graph, root_node: str = None) -> str\n"})}),"\n",(0,s.jsx)(n.p,{children:"Traverse a tree and return the traversal sequence."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"T"})," ",(0,s.jsx)(n.em,{children:"nx.Graph"})," - The tree to be traversed."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"root_node"})," ",(0,s.jsx)(n.em,{children:"str, optional"})," - The root node of the tree. If None, traversal starts from the root."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"str"})," - The traversal sequence."]}),"\n"]}),"\n",(0,s.jsx)(n.h4,{id:"traverse_tree-1",children:"traverse_tree"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def traverse_tree(T: nx.Graph, root_node: str = None) -> str\n"})}),"\n",(0,s.jsx)(n.p,{children:"Traverse a tree and return the traversal sequence."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"T"})," ",(0,s.jsx)(n.em,{children:"nx.Graph"})," - The tree to be traversed."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"root_node"})," ",(0,s.jsx)(n.em,{children:"str, optional"})," - The root node of the tree. If None, traversal starts from the root."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"str"})," - The traversal sequence."]}),"\n"]}),"\n",(0,s.jsx)(n.h4,{id:"traverse_tree_cutline",children:"traverse_tree_cutline"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def traverse_tree_cutline(T: nx.Graph,\n                          root_node: str = None,\n                          cutline: int = 2) -> List[List[str]]\n"})}),"\n",(0,s.jsx)(n.p,{children:"Traverse a tree with a cutline and return the community bags."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"T"})," ",(0,s.jsx)(n.em,{children:"nx.Graph"})," - The tree to be traversed."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"root_node"})," ",(0,s.jsx)(n.em,{children:"str, optional"})," - The root node of the tree. If None, traversal starts from the root."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"cutline"})," ",(0,s.jsx)(n.em,{children:"int, optional"})," - The cutline level."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"List[List[str]]"})," - List of community bags."]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(d,{...e})}):d(e)}},8453:(e,n,r)=>{r.d(n,{R:()=>l,x:()=>c});var s=r(6540);const i={},t=s.createContext(i);function l(e){const n=s.useContext(t);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:l(e.components),s.createElement(t.Provider,{value:n},e.children)}}}]);