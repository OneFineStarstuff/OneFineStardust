"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import * as d3 from "d3"

type NodeType = "orchestrator" | "inference" | "data" | "output" | "policy"

interface GraphNode extends d3.SimulationNodeDatum {
  id: string
  label: string
  type: NodeType
  vars: number
  risk: "low" | "medium" | "high"
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode
  target: string | GraphNode
  varName: string
  dataFlow: "bidirectional" | "read" | "write"
}

const NODES: GraphNode[] = [
  { id: "orch-1", label: "Primary Orchestrator", type: "orchestrator", vars: 24, risk: "high" },
  { id: "orch-2", label: "Safety Supervisor", type: "orchestrator", vars: 18, risk: "medium" },
  { id: "inf-1", label: "LLM Inference A", type: "inference", vars: 12, risk: "medium" },
  { id: "inf-2", label: "LLM Inference B", type: "inference", vars: 9, risk: "low" },
  { id: "inf-3", label: "Vision Model", type: "inference", vars: 7, risk: "medium" },
  { id: "dat-1", label: "Feature Store", type: "data", vars: 31, risk: "high" },
  { id: "dat-2", label: "RAG Index", type: "data", vars: 15, risk: "medium" },
  { id: "dat-3", label: "Training Cache", type: "data", vars: 22, risk: "high" },
  { id: "out-1", label: "Decision Output", type: "output", vars: 6, risk: "low" },
  { id: "out-2", label: "Audit Emitter", type: "output", vars: 8, risk: "low" },
  { id: "pol-1", label: "Policy Engine", type: "policy", vars: 19, risk: "high" },
  { id: "pol-2", label: "Consent Gate", type: "policy", vars: 11, risk: "medium" },
]

const LINKS: GraphLink[] = [
  { source: "orch-1", target: "inf-1", varName: "context_window", dataFlow: "write" },
  { source: "orch-1", target: "inf-2", varName: "prompt_tokens", dataFlow: "write" },
  { source: "orch-1", target: "dat-1", varName: "feature_batch", dataFlow: "bidirectional" },
  { source: "orch-1", target: "pol-1", varName: "policy_ctx", dataFlow: "write" },
  { source: "orch-2", target: "orch-1", varName: "safety_score", dataFlow: "write" },
  { source: "orch-2", target: "pol-2", varName: "consent_sig", dataFlow: "bidirectional" },
  { source: "inf-1", target: "out-1", varName: "logits", dataFlow: "write" },
  { source: "inf-1", target: "dat-2", varName: "retrieval_q", dataFlow: "read" },
  { source: "inf-2", target: "out-1", varName: "completion", dataFlow: "write" },
  { source: "inf-3", target: "inf-1", varName: "vision_embed", dataFlow: "write" },
  { source: "dat-1", target: "inf-1", varName: "embeddings", dataFlow: "read" },
  { source: "dat-1", target: "dat-3", varName: "training_rows", dataFlow: "bidirectional" },
  { source: "dat-2", target: "inf-2", varName: "rag_context", dataFlow: "read" },
  { source: "dat-3", target: "pol-1", varName: "data_lineage", dataFlow: "write" },
  { source: "pol-1", target: "out-2", varName: "decision_hash", dataFlow: "write" },
  { source: "pol-2", target: "out-2", varName: "consent_log", dataFlow: "write" },
  { source: "out-2", target: "orch-2", varName: "audit_event", dataFlow: "write" },
]

const NODE_COLORS: Record<NodeType, string> = {
  orchestrator: "oklch(0.72 0.16 195)",
  inference: "oklch(0.62 0.20 280)",
  data: "oklch(0.68 0.18 145)",
  output: "oklch(0.55 0.010 220)",
  policy: "oklch(0.78 0.17 75)",
}

const RISK_STROKE: Record<string, string> = {
  low: "oklch(0.68 0.18 145)",
  medium: "oklch(0.78 0.17 75)",
  high: "oklch(0.58 0.22 25)",
}

const NODE_TYPE_LABELS: Record<NodeType, string> = {
  orchestrator: "ORCH",
  inference: "INF",
  data: "DATA",
  output: "OUT",
  policy: "POL",
}

export function VariableMap() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [filterType, setFilterType] = useState<NodeType | "all">("all")
  const simRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null)

  const buildGraph = useCallback(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()

    const container = containerRef.current
    if (!container) return
    const W = container.clientWidth || 480
    const H = 340

    svg.attr("width", W).attr("height", H)

    // Defs: arrowheads
    const defs = svg.append("defs")
    const mkArrow = (id: string, color: string) => {
      defs.append("marker")
        .attr("id", id)
        .attr("viewBox", "0 -4 10 8")
        .attr("refX", 18)
        .attr("refY", 0)
        .attr("markerWidth", 5)
        .attr("markerHeight", 5)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-4L10,0L0,4")
        .attr("fill", color)
    }
    mkArrow("arrow-write", "oklch(0.72 0.16 195)")
    mkArrow("arrow-read", "oklch(0.68 0.18 145)")
    mkArrow("arrow-bi", "oklch(0.78 0.17 75)")

    const filteredNodes = filterType === "all"
      ? NODES
      : NODES.filter(n => n.type === filterType)
    const filteredIds = new Set(filteredNodes.map(n => n.id))
    const filteredLinks = LINKS.filter(
      l => filteredIds.has(l.source as string) && filteredIds.has(l.target as string)
    )

    const sim = d3.forceSimulation<GraphNode>(filteredNodes)
      .force("link", d3.forceLink<GraphNode, GraphLink>(filteredLinks).id(d => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide(28))
    simRef.current = sim

    const g = svg.append("g")

    // Zoom
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.4, 2.5])
        .on("zoom", (event) => g.attr("transform", event.transform))
    )

    // Links
    const link = g.selectAll<SVGLineElement, GraphLink>(".link")
      .data(filteredLinks)
      .enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke", d => d.dataFlow === "write" ? "oklch(0.72 0.16 195)" : d.dataFlow === "read" ? "oklch(0.68 0.18 145)" : "oklch(0.78 0.17 75)")
      .attr("stroke-width", 1.2)
      .attr("stroke-opacity", 0.6)
      .attr("marker-end", d => `url(#arrow-${d.dataFlow === "bidirectional" ? "bi" : d.dataFlow})`)

    // Link labels
    const linkLabel = g.selectAll<SVGTextElement, GraphLink>(".link-label")
      .data(filteredLinks)
      .enter()
      .append("text")
      .attr("class", "link-label")
      .attr("font-size", 7)
      .attr("font-family", "monospace")
      .attr("fill", "oklch(0.40 0.008 235)")
      .attr("text-anchor", "middle")
      .text(d => d.varName)

    // Nodes group
    const nodeGroup = g.selectAll<SVGGElement, GraphNode>(".node")
      .data(filteredNodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .style("cursor", "pointer")
      .on("click", (_e, d) => setSelected(prev => prev?.id === d.id ? null : d))
      .call(
        d3.drag<SVGGElement, GraphNode>()
          .on("start", (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y })
          .on("end", (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null })
      )

    // Node circle bg
    nodeGroup.append("circle")
      .attr("r", d => 14 + Math.sqrt(d.vars))
      .attr("fill", d => NODE_COLORS[d.type] + "22")
      .attr("stroke", d => RISK_STROKE[d.risk])
      .attr("stroke-width", 1.5)

    // Inner circle
    nodeGroup.append("circle")
      .attr("r", 12)
      .attr("fill", d => NODE_COLORS[d.type] + "44")
      .attr("stroke", d => NODE_COLORS[d.type])
      .attr("stroke-width", 1)

    // Type label
    nodeGroup.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", -1)
      .attr("font-size", 7)
      .attr("font-family", "monospace")
      .attr("font-weight", "bold")
      .attr("fill", d => NODE_COLORS[d.type])
      .text(d => NODE_TYPE_LABELS[d.type])

    // Var count
    nodeGroup.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", 8)
      .attr("font-size", 6)
      .attr("font-family", "monospace")
      .attr("fill", "oklch(0.55 0.010 220)")
      .text(d => `${d.vars}v`)

    // Node name below
    nodeGroup.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", 28)
      .attr("font-size", 8)
      .attr("font-family", "monospace")
      .attr("fill", "oklch(0.55 0.010 220)")
      .text(d => d.label.length > 16 ? d.label.slice(0, 14) + "…" : d.label)

    sim.on("tick", () => {
      link
        .attr("x1", d => (d.source as GraphNode).x ?? 0)
        .attr("y1", d => (d.source as GraphNode).y ?? 0)
        .attr("x2", d => (d.target as GraphNode).x ?? 0)
        .attr("y2", d => (d.target as GraphNode).y ?? 0)

      linkLabel
        .attr("x", d => (((d.source as GraphNode).x ?? 0) + ((d.target as GraphNode).x ?? 0)) / 2)
        .attr("y", d => (((d.source as GraphNode).y ?? 0) + ((d.target as GraphNode).y ?? 0)) / 2)

      nodeGroup.attr("transform", d => `translate(${d.x ?? 0},${d.y ?? 0})`)
    })
  }, [filterType])

  useEffect(() => {
    buildGraph()
    return () => { simRef.current?.stop() }
  }, [buildGraph])

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Controls */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {(["all", "orchestrator", "inference", "data", "output", "policy"] as const).map(t => (
          <button
            key={t}
            onClick={() => setFilterType(t)}
            className={`text-[9px] font-mono px-2 py-1 rounded-sm border transition-colors
              ${filterType === t
                ? "bg-primary text-primary-foreground border-primary"
                : "border-border text-muted-foreground hover:text-foreground hover:border-foreground"
              }`}
          >
            {t === "all" ? "ALL" : NODE_TYPE_LABELS[t as NodeType]}
          </button>
        ))}
        <span className="ml-auto text-[9px] font-mono text-muted-foreground">{NODES.length} nodes / {LINKS.length} edges</span>
      </div>

      {/* Graph */}
      <div ref={containerRef} className="relative bg-card border border-border rounded-sm overflow-hidden" style={{ height: 340 }}>
        <svg ref={svgRef} className="w-full h-full" aria-label="Agent variable dependency graph" role="img" />
        {/* Legend */}
        <div className="absolute bottom-2 left-2 flex flex-col gap-1 pointer-events-none">
          {(Object.entries(NODE_COLORS) as [NodeType, string][]).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-[8px] font-mono text-muted-foreground uppercase">{type}</span>
            </div>
          ))}
        </div>
        {/* Zoom hint */}
        <div className="absolute top-2 right-2 text-[8px] font-mono text-muted-foreground pointer-events-none">
          SCROLL TO ZOOM · DRAG TO PAN
        </div>
      </div>

      {/* Selected node detail */}
      {selected && (
        <div className="bg-card border border-primary/40 rounded-sm p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: NODE_COLORS[selected.type] }} />
            <span className="text-[10px] font-mono font-bold text-foreground">{selected.label}</span>
            <span className="ml-auto text-[9px] font-mono text-muted-foreground border border-border px-1.5 py-0.5 rounded-sm">{NODE_TYPE_LABELS[selected.type]}</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-muted rounded-sm px-2 py-1.5">
              <div className="text-sm font-mono font-bold text-primary">{selected.vars}</div>
              <div className="text-[8px] font-mono text-muted-foreground">VARIABLES</div>
            </div>
            <div className="bg-muted rounded-sm px-2 py-1.5">
              <div className={`text-sm font-mono font-bold ${RISK_STROKE[selected.risk] === "oklch(0.68 0.18 145)" ? "text-status-ok" : selected.risk === "medium" ? "text-status-warn" : "text-status-critical"}`}>
                {selected.risk.toUpperCase()}
              </div>
              <div className="text-[8px] font-mono text-muted-foreground">RISK LEVEL</div>
            </div>
            <div className="bg-muted rounded-sm px-2 py-1.5">
              <div className="text-sm font-mono font-bold text-foreground">
                {LINKS.filter(l =>
                  (typeof l.source === "string" ? l.source : (l.source as GraphNode).id) === selected.id ||
                  (typeof l.target === "string" ? l.target : (l.target as GraphNode).id) === selected.id
                ).length}
              </div>
              <div className="text-[8px] font-mono text-muted-foreground">CONNECTIONS</div>
            </div>
          </div>
          <div className="mt-2 flex flex-col gap-1">
            {LINKS.filter(l =>
              (typeof l.source === "string" ? l.source : (l.source as GraphNode).id) === selected.id ||
              (typeof l.target === "string" ? l.target : (l.target as GraphNode).id) === selected.id
            ).map((l, i) => {
              const srcId = typeof l.source === "string" ? l.source : (l.source as GraphNode).id
              const tgtId = typeof l.target === "string" ? l.target : (l.target as GraphNode).id
              const isOut = srcId === selected.id
              const peerId = isOut ? tgtId : srcId
              const peer = NODES.find(n => n.id === peerId)
              return (
                <div key={i} className="flex items-center gap-2 text-[9px] font-mono">
                  <span className={`${isOut ? "text-primary" : "text-status-ok"}`}>{isOut ? "OUT" : " IN"}</span>
                  <span className="text-muted-foreground font-mono">{l.varName}</span>
                  <span className="text-foreground ml-auto">{peer?.label}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
