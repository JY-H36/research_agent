"""
知识图谱可视化模块
基于 vis.js 生成交互式网络图，嵌入 Streamlit
"""
from typing import Optional

from knowledge_graph.graph_store import get_nx_graph, get_graph_stats
from utils.logger import get_logger

logger = get_logger(__name__)

# 节点颜色映射（按实体类型）
TYPE_COLORS = {
    "paper":    "#4FC3F7",  # 蓝
    "author":   "#FFB74D",  # 橙
    "method":   "#81C784",  # 绿
    "dataset":  "#EF5350",  # 红
    "task":     "#BA68C8",  # 紫
    "metric":   "#FFD54F",  # 黄
    "venue":    "#90A4AE",  # 灰
}

# 边颜色映射（按关系类型）
EDGE_COLORS = {
    "WRITTEN_BY":       "#FFB74D",
    "CITES":            "#90A4AE",
    "USES_METHOD":      "#81C784",
    "EVALUATES_ON":     "#EF5350",
    "TRAINS_ON":        "#EF5350",
    "BELONGS_TO":       "#BA68C8",
    "IMPROVES_UPON":    "#4FC3F7",
    "PUBLISHED_IN":     "#90A4AE",
    "REPORTS_METRIC":   "#FFD54F",
}

TYPE_LABELS = {
    "paper":   "📄 论文",
    "author":  "👤 作者",
    "method":  "🔧 方法",
    "dataset": "📊 数据集",
    "task":    "🎯 任务",
    "metric":  "📏 指标",
    "venue":   "🏛️ 发表地",
}


def build_pyvis_graph(height: str = "650px", filter_types: list = None,
                      max_nodes: int = 200) -> Optional[str]:
    """
    从 NetworkX 图生成交互式 HTML（vis.js），不依赖 pyvis 文件操作避免 GBK 编码问题。
    """
    import json as _json

    G = get_nx_graph()
    if G.number_of_nodes() == 0:
        return None

    if G.number_of_nodes() > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    # 构建节点 JSON
    nodes = []
    for node_id, attr in G.nodes(data=True):
        ntype = attr.get("type", "unknown")
        if filter_types and ntype not in filter_types:
            continue
        label = attr.get("label", node_id)[:60]
        color = TYPE_COLORS.get(ntype, "#CCCCCC")
        size_map = {"paper": 18, "author": 12, "method": 15,
                    "dataset": 15, "task": 14, "metric": 10, "venue": 10}
        size = size_map.get(ntype, 12)
        degree = G.degree(node_id) if hasattr(G, 'degree') else 1
        size = min(size + degree * 2, 40)

        title_parts = [f"<b>{_escape_html(label)}</b>",
                       f"类型: {TYPE_LABELS.get(ntype, ntype)}"]
        if ntype == "paper" and attr.get("year"):
            title_parts.append(f"年份: {attr['year']}")
        if ntype == "paper" and attr.get("citation_count"):
            title_parts.append(f"引用: {attr['citation_count']}")
        if ntype == "author" and attr.get("affiliation"):
            title_parts.append(f"机构: {_escape_html(str(attr['affiliation'])[:80])}")
        if ntype == "method" and attr.get("category"):
            title_parts.append(f"类别: {attr['category']}")

        nodes.append({
            "id": node_id,
            "label": label[:30],
            "title": "<br>".join(title_parts),
            "color": color,
            "size": size,
            "shape": "dot",
            "group": ntype,
        })

    # 构建边 JSON
    edges = []
    for source, target, data in G.edges(data=True):
        stype = G.nodes.get(source, {}).get("type", "")
        ttype = G.nodes.get(target, {}).get("type", "")
        if filter_types and (stype not in filter_types or ttype not in filter_types):
            continue
        rel = data.get("relation", "")
        edges.append({
            "from": source,
            "to": target,
            "title": rel,
            "color": EDGE_COLORS.get(rel, "#CCCCCC"),
            "label": rel if G.number_of_edges() < 50 else "",
            "arrows": "to",
        })

    nodes_json = _json.dumps(nodes, ensure_ascii=False)
    edges_json = _json.dumps(edges, ensure_ascii=False)

    # 读本地 vis-network JS（一次性，缓存到模块变量）
    global _vis_js_cache
    if _vis_js_cache is None:
        _vis_js_cache = _load_vis_js()

    html = _KG_HTML_TEMPLATE
    html = html.replace("__HEIGHT__", height)
    html = html.replace("__VIS_JS__", _vis_js_cache)
    html = html.replace("__NODES_JSON__", nodes_json)
    html = html.replace("__EDGES_JSON__", edges_json)
    html = html.replace("__NODE_COUNT__", str(len(nodes)))
    html = html.replace("__EDGE_COUNT__", str(len(edges)))
    return html


_vis_js_cache: str = None


def _load_vis_js() -> str:
    """加载本地 vis-network.min.js"""
    import os as _os
    _path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "vis-network.min.js")
    try:
        with open(_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("vis-network.min.js 未找到: %s", _path)
        return ""


def _escape_html(text: str) -> str:
    """转义 HTML 特殊字符"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


_KG_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script>__VIS_JS__</script>
<style>
  body { margin:0; background:#1a1a2e; font-family:"Microsoft YaHei",sans-serif; }
  #kg-viz { width:100%; height:__HEIGHT__; background:#ffffff; border-radius:6px; }
  .vis-tooltip { position:absolute; padding:8px 12px; background:rgba(30,30,30,0.92);
    color:#fff; border-radius:6px; font-size:12px; line-height:1.6; max-width:350px; white-space:normal; }
  .vis-network { outline:none; }
  .legend { position:absolute; bottom:10px; left:10px; background:rgba(0,0,0,0.7);
    color:#ccc; padding:6px 10px; border-radius:6px; font-size:11px; z-index:999; }
  .legend span { display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 4px 0 8px; }
  .legend span:first-child { margin-left:0; }
</style></head><body>
<div style="position:relative;">
  <div id="kg-viz"></div>
  <div class="legend">
    <span style="background:#4FC3F7"></span>论文 <span style="background:#FFB74D"></span>作者
    <span style="background:#81C784"></span>方法 <span style="background:#EF5350"></span>数据集
    <span style="background:#BA68C8"></span>任务 <span style="background:#FFD54F"></span>指标
    <span style="background:#90A4AE"></span>发表地
    &nbsp;|&nbsp; 节点: __NODE_COUNT__ &nbsp; 边: __EDGE_COUNT__
  </div>
</div>
<script>
(function(){
  var nodes=new vis.DataSet(__NODES_JSON__);
  var edges=new vis.DataSet(__EDGES_JSON__);
  var container=document.getElementById('kg-viz');
  var data={nodes:nodes,edges:edges};
  var options={
    physics:{
      barnesHut:{gravitationalConstant:-3000,centralGravity:0.3,springLength:140,springConstant:0.04,damping:0.3},
      maxVelocity:50,minVelocity:0.5,
      stabilization:{iterations:200,fit:true}
    },
    interaction:{hover:true,tooltipDelay:100,navigationButtons:true,keyboard:true},
    edges:{smooth:{type:"continuous",forceDirection:"none"},arrows:{to:{enabled:true,scaleFactor:0.5}}},
    nodes:{font:{size:12,face:"Microsoft YaHei,sans-serif"}}
  };
  new vis.Network(container,data,options);
})();
</script></body></html>"""


def build_stats_html() -> str:
    """生成图谱统计 HTML"""
    stats = get_graph_stats()
    rels = stats.get("relationships", {})
    total_rels = sum(rels.values())

    return f"""
    <style>
    .kg-stats {{
        font-family: 'Microsoft YaHei', sans-serif;
        padding: 10px;
    }}
    .kg-stats .stat-row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 15px;
    }}
    .kg-stats .stat-card {{
        background: #1E1E1E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 16px;
        min-width: 80px;
        text-align: center;
    }}
    .kg-stats .stat-card .num {{
        font-size: 24px;
        font-weight: bold;
        color: #4FC3F7;
    }}
    .kg-stats .stat-card .lbl {{
        font-size: 11px;
        color: #888;
    }}
    .kg-stats .legend {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        font-size: 12px;
        color: #AAA;
    }}
    .kg-stats .legend span {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 3px;
        vertical-align: middle;
    }}
    </style>
    <div class="kg-stats">
        <div class="stat-row">
            <div class="stat-card"><div class="num">{stats['papers']}</div><div class="lbl">📄 论文</div></div>
            <div class="stat-card"><div class="num">{stats['authors']}</div><div class="lbl">👤 作者</div></div>
            <div class="stat-card"><div class="num">{stats['methods']}</div><div class="lbl">🔧 方法</div></div>
            <div class="stat-card"><div class="num">{stats['datasets']}</div><div class="lbl">📊 数据集</div></div>
            <div class="stat-card"><div class="num">{stats['tasks']}</div><div class="lbl">🎯 任务</div></div>
            <div class="stat-card"><div class="num">{stats['metrics']}</div><div class="lbl">📏 指标</div></div>
            <div class="stat-card"><div class="num">{stats['venues']}</div><div class="lbl">🏛️ 发表地</div></div>
            <div class="stat-card"><div class="num">{total_rels}</div><div class="lbl">🔗 关系</div></div>
        </div>
        <div class="legend">
            <span style="background:#4FC3F7"></span> 论文
            <span style="background:#FFB74D"></span> 作者
            <span style="background:#81C784"></span> 方法
            <span style="background:#EF5350"></span> 数据集
            <span style="background:#BA68C8"></span> 任务
            <span style="background:#FFD54F"></span> 指标
            <span style="background:#90A4AE"></span> 发表地
        </div>
    </div>
    """
