"use client"

import { useState } from "react"

interface WorkflowRec {
  id: string
  title: string
  priority: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
  framework: string
  rationale: string
  impact: string
  effort: "LOW" | "MEDIUM" | "HIGH"
  status: "PENDING" | "ACCEPTED" | "DISMISSED"
  confidence: number
  linkedControls: string[]
}

const WORKFLOW_RECS: WorkflowRec[] = [
  {
    id: "WF-2025-001",
    title: "Implement Model Card Refresh Cycle",
    priority: "CRITICAL",
    framework: "EU AI Act Art.13",
    rationale: "Last model card update was 47 days ago. Mandatory refresh cycle for high-risk systems is 30 days.",
    impact: "Avoids potential Article 13(3)(b) non-compliance finding in next audit cycle.",
    effort: "LOW",
    status: "PENDING",
    confidence: 97,
    linkedControls: ["MC-001", "TR-003", "AUD-007"],
  },
  {
    id: "WF-2025-002",
    title: "Patch SGX Enclave Firmware on HW-004",
    priority: "CRITICAL",
    framework: "DORA Art.16",
    rationale: "Hardware node HW-004 running firmware v2.3.9 with known CVE-2025-1182. TPM attestation failing.",
    impact: "Unattested hardware increases risk score by 18 points and violates ICT security requirements.",
    effort: "MEDIUM",
    status: "PENDING",
    confidence: 99,
    linkedControls: ["HW-SEC-02", "DORA-ICT-04"],
  },
  {
    id: "WF-2025-003",
    title: "Conduct Bias Evaluation for VisionPipeline",
    priority: "HIGH",
    framework: "NIST AI RMF MEASURE-2.5",
    rationale: "VisionPipeline model has not been evaluated against protected attributes since deployment.",
    impact: "MEASURE function score increase by ~8 points. Reduces demographic harm risk.",
    effort: "HIGH",
    status: "PENDING",
    confidence: 84,
    linkedControls: ["BIAS-01", "NIST-M-25", "FAIR-003"],
  },
  {
    id: "WF-2025-004",
    title: "Add Consent Verification to RAGRetriever",
    priority: "HIGH",
    framework: "EU AI Act Art.10",
    rationale: "RAG data sources include 3 datasets with expired consent records.",
    impact: "Prevents data governance violation. Required for resuming suspended RAG agent.",
    effort: "MEDIUM",
    status: "ACCEPTED",
    confidence: 91,
    linkedControls: ["DG-004", "GDPR-ART6", "RAG-CTL-01"],
  },
  {
    id: "WF-2025-005",
    title: "Enable Differential Privacy on Training Cache",
    priority: "MEDIUM",
    framework: "NIST AI RMF GOVERN-1.3",
    rationale: "Training cache lacks DP guarantees. Epsilon value not bounded.",
    impact: "Improves GOVERN function score by ~6 points. Reduces PII leakage risk.",
    effort: "HIGH",
    status: "DISMISSED",
    confidence: 76,
    linkedControls: ["PRIV-01", "NIST-G-13"],
  },
]

const PRIORITY_COLOR: Record<WorkflowRec["priority"], string> = {
  CRITICAL: "text-status-critical border-status-critical",
  HIGH: "text-status-warn border-status-warn",
  MEDIUM: "text-primary border-primary",
  LOW: "text-muted-foreground border-border",
}

const PRIORITY_BG: Record<WorkflowRec["priority"], string> = {
  CRITICAL: "bg-status-critical/10",
  HIGH: "bg-status-warn/10",
  MEDIUM: "bg-primary/10",
  LOW: "bg-muted/10",
}

const STATUS_COLOR: Record<WorkflowRec["status"], string> = {
  PENDING: "text-status-warn",
  ACCEPTED: "text-status-ok",
  DISMISSED: "text-status-inactive",
}

export function WorkflowEngine() {
  const [recs, setRecs] = useState<WorkflowRec[]>(WORKFLOW_RECS)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [filter, setFilter] = useState<"ALL" | "PENDING" | "ACCEPTED" | "DISMISSED">("ALL")

  const handleAction = (id: string, action: "ACCEPTED" | "DISMISSED") => {
    setRecs(prev => prev.map(r => r.id === id ? { ...r, status: action } : r))
  }

  const filtered = filter === "ALL" ? recs : recs.filter(r => r.status === filter)
  const pendingCrit = recs.filter(r => r.status === "PENDING" && r.priority === "CRITICAL").length

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Header stats */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: "PENDING", val: recs.filter(r => r.status === "PENDING").length, color: "text-status-warn" },
          { label: "ACCEPTED", val: recs.filter(r => r.status === "ACCEPTED").length, color: "text-status-ok" },
          { label: "DISMISSED", val: recs.filter(r => r.status === "DISMISSED").length, color: "text-muted-foreground" },
          { label: "CRIT OPEN", val: pendingCrit, color: "text-status-critical" },
        ].map(stat => (
          <div key={stat.label} className="bg-card border border-border rounded-sm px-2 py-1.5 text-center">
            <div className={`text-base font-mono font-bold ${stat.color}`}>{stat.val}</div>
            <div className="text-[8px] font-mono text-muted-foreground tracking-widest">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Filter tabs */}
      <div className="flex gap-0 border border-border rounded-sm overflow-hidden">
        {(["ALL", "PENDING", "ACCEPTED", "DISMISSED"] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`flex-1 text-[9px] font-mono py-1.5 transition-colors border-r border-border last:border-0
              ${filter === f ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground hover:bg-muted"}`}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Recommendations */}
      <div className="flex flex-col gap-2 overflow-auto">
        {filtered.map(rec => (
          <div
            key={rec.id}
            className={`bg-card border rounded-sm overflow-hidden transition-all ${rec.status === "DISMISSED" ? "opacity-50" : ""} ${PRIORITY_BG[rec.priority]} border-border`}
          >
            <button
              className="w-full text-left px-3 py-2 flex items-start gap-2"
              onClick={() => setExpanded(expanded === rec.id ? null : rec.id)}
              aria-expanded={expanded === rec.id}
            >
              <span className={`text-[8px] font-mono font-bold border rounded-sm px-1 py-0.5 mt-0.5 shrink-0 ${PRIORITY_COLOR[rec.priority]}`}>
                {rec.priority}
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono font-semibold text-foreground">{rec.title}</span>
                  <span className={`ml-auto text-[9px] font-mono ${STATUS_COLOR[rec.status]}`}>{rec.status}</span>
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[8px] font-mono text-muted-foreground">{rec.id}</span>
                  <span className="text-[8px] font-mono text-muted-foreground border border-border px-1 rounded-sm">{rec.framework}</span>
                  <span className="ml-auto text-[8px] font-mono text-muted-foreground">CONF: {rec.confidence}%</span>
                </div>
              </div>
              <span className="text-muted-foreground text-[10px] ml-1 mt-0.5">{expanded === rec.id ? "▲" : "▼"}</span>
            </button>

            {expanded === rec.id && (
              <div className="border-t border-border px-3 py-2.5 flex flex-col gap-2">
                <div>
                  <div className="text-[9px] font-mono text-muted-foreground mb-0.5">RATIONALE</div>
                  <p className="text-[10px] font-mono text-foreground leading-relaxed">{rec.rationale}</p>
                </div>
                <div>
                  <div className="text-[9px] font-mono text-muted-foreground mb-0.5">EXPECTED IMPACT</div>
                  <p className="text-[10px] font-mono text-foreground leading-relaxed">{rec.impact}</p>
                </div>
                <div className="flex items-center gap-3">
                  <div>
                    <div className="text-[9px] font-mono text-muted-foreground">EFFORT</div>
                    <span className={`text-[9px] font-mono font-bold ${rec.effort === "LOW" ? "text-status-ok" : rec.effort === "MEDIUM" ? "text-status-warn" : "text-status-critical"}`}>{rec.effort}</span>
                  </div>
                  <div>
                    <div className="text-[9px] font-mono text-muted-foreground">CONTROLS</div>
                    <div className="flex flex-wrap gap-1">
                      {rec.linkedControls.map(c => (
                        <span key={c} className="text-[8px] font-mono border border-border text-muted-foreground px-1 rounded-sm">{c}</span>
                      ))}
                    </div>
                  </div>
                </div>
                {rec.status === "PENDING" && (
                  <div className="flex items-center gap-2 mt-1">
                    <button
                      onClick={() => handleAction(rec.id, "ACCEPTED")}
                      className="text-[9px] font-mono px-3 py-1 rounded-sm bg-status-ok/20 border border-status-ok text-status-ok hover:bg-status-ok hover:text-black transition-colors"
                    >
                      ACCEPT
                    </button>
                    <button
                      onClick={() => handleAction(rec.id, "DISMISSED")}
                      className="text-[9px] font-mono px-3 py-1 rounded-sm border border-border text-muted-foreground hover:text-foreground hover:border-foreground transition-colors"
                    >
                      DISMISS
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
