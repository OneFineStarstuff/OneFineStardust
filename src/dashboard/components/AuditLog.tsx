"use client"

import { useState, useEffect, useRef } from "react"

type EventType = "DECISION" | "POLICY" | "ACCESS" | "KILL_SWITCH" | "ATTESTATION" | "WORKFLOW" | "ANOMALY"
type ZKStatus = "VERIFIED" | "PENDING" | "FAILED" | "NOT_REQUIRED"

interface AuditEntry {
  id: string
  ts: string
  type: EventType
  actor: string
  action: string
  resource: string
  hash: string
  zkStatus: ZKStatus
  worm: boolean
  signature: string
}

const TYPE_COLOR: Record<EventType, string> = {
  DECISION: "text-primary",
  POLICY: "text-chart-5",
  ACCESS: "text-muted-foreground",
  KILL_SWITCH: "text-status-critical",
  ATTESTATION: "text-status-ok",
  WORKFLOW: "text-accent",
  ANOMALY: "text-status-critical",
}

const ZK_COLOR: Record<ZKStatus, string> = {
  VERIFIED: "text-status-ok",
  PENDING: "text-status-warn",
  FAILED: "text-status-critical",
  NOT_REQUIRED: "text-muted-foreground",
}

const ZK_ICON: Record<ZKStatus, string> = {
  VERIFIED: "ZK✓",
  PENDING: "ZK…",
  FAILED: "ZK✗",
  NOT_REQUIRED: "—",
}

const SEED_ENTRIES: AuditEntry[] = [
  { id: "AUD-00001", ts: "2025-06-15T09:14:32.001Z", type: "DECISION", actor: "AGT-001", action: "CLASSIFY_RISK", resource: "/models/gpt-4o/inference#3821", hash: "sha3:a9f1c3d2...", zkStatus: "VERIFIED", worm: true, signature: "0xab12..." },
  { id: "AUD-00002", ts: "2025-06-15T09:14:33.412Z", type: "POLICY", actor: "pol-1", action: "POLICY_ENFORCE", resource: "/policy/data-residency-eu", hash: "sha3:b2e4a7f1...", zkStatus: "VERIFIED", worm: true, signature: "0xcd34..." },
  { id: "AUD-00003", ts: "2025-06-15T09:14:34.892Z", type: "ACCESS", actor: "human:alice@corp.com", action: "VIEW_AUDIT_LOG", resource: "/audit/stream", hash: "sha3:c3f9b2e0...", zkStatus: "NOT_REQUIRED", worm: true, signature: "0xef56..." },
  { id: "AUD-00004", ts: "2025-06-15T09:14:36.100Z", type: "ATTESTATION", actor: "hw-attest-svc", action: "TPM_MEASURE", resource: "/hardware/gpu-node-01", hash: "sha3:d4e1c8a3...", zkStatus: "VERIFIED", worm: true, signature: "0x1234..." },
  { id: "AUD-00005", ts: "2025-06-15T09:14:37.500Z", type: "ANOMALY", actor: "monitor-v2", action: "DRIFT_DETECTED", resource: "/models/mistral-8x7b", hash: "sha3:e5f2d9b4...", zkStatus: "PENDING", worm: true, signature: "0x5678..." },
  { id: "AUD-00006", ts: "2025-06-15T09:14:39.201Z", type: "WORKFLOW", actor: "ai-engine", action: "REC_GENERATED", resource: "/workflow/WF-2025-001", hash: "sha3:f6a3e0c5...", zkStatus: "VERIFIED", worm: true, signature: "0x9abc..." },
  { id: "AUD-00007", ts: "2025-06-15T09:14:40.887Z", type: "DECISION", actor: "AGT-006", action: "CONTENT_BLOCK", resource: "/outputs/req#48821", hash: "sha3:a7b4f1d6...", zkStatus: "VERIFIED", worm: true, signature: "0xdef0..." },
]

function formatTs(ts: string) {
  return ts.replace("T", " ").replace("Z", "").slice(0, 23)
}

function generateEntry(seq: number): AuditEntry {
  const types: EventType[] = ["DECISION", "POLICY", "ACCESS", "ATTESTATION", "WORKFLOW"]
  const actors = ["AGT-001", "AGT-002", "AGT-005", "AGT-006", "pol-1", "pol-2", "hw-attest-svc"]
  const actions = ["CLASSIFY_RISK", "POLICY_ENFORCE", "INFERENCE_LOG", "TPM_MEASURE", "REC_GENERATED", "CONSENT_CHECK", "EMBED_STORE"]
  const zkStatuses: ZKStatus[] = ["VERIFIED", "VERIFIED", "VERIFIED", "PENDING", "NOT_REQUIRED"]
  const type = types[Math.floor(Math.random() * types.length)]
  const now = new Date().toISOString()
  const hexStr = Math.random().toString(16).slice(2, 10)
  return {
    id: `AUD-${String(seq).padStart(5, "0")}`,
    ts: now,
    type,
    actor: actors[Math.floor(Math.random() * actors.length)],
    action: actions[Math.floor(Math.random() * actions.length)],
    resource: `/objects/${Math.random().toString(36).slice(2, 10)}`,
    hash: `sha3:${hexStr}...`,
    zkStatus: zkStatuses[Math.floor(Math.random() * zkStatuses.length)],
    worm: true,
    signature: `0x${hexStr}`,
  }
}

export function AuditLog() {
  const [entries, setEntries] = useState<AuditEntry[]>(SEED_ENTRIES)
  const [search, setSearch] = useState("")
  const [typeFilter, setTypeFilter] = useState<EventType | "ALL">("ALL")
  const [zkFilter, setZkFilter] = useState<ZKStatus | "ALL">("ALL")
  const [streaming, setStreaming] = useState(true)
  const [selectedEntry, setSelectedEntry] = useState<AuditEntry | null>(null)
  const seqRef = useRef(SEED_ENTRIES.length + 1)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!streaming) return
    const interval = setInterval(() => {
      setEntries(prev => {
        const newEntry = generateEntry(seqRef.current++)
        const next = [newEntry, ...prev].slice(0, 200)
        return next
      })
    }, 1800)
    return () => clearInterval(interval)
  }, [streaming])

  const filtered = entries.filter(e => {
    const matchType = typeFilter === "ALL" || e.type === typeFilter
    const matchZK = zkFilter === "ALL" || e.zkStatus === zkFilter
    const matchSearch = !search || e.actor.includes(search) || e.action.includes(search) || e.id.includes(search) || e.resource.includes(search)
    return matchType && matchZK && matchSearch
  })

  const handleExport = () => {
    const content = filtered.map(e =>
      `${e.id}\t${e.ts}\t${e.type}\t${e.actor}\t${e.action}\t${e.resource}\t${e.hash}\t${e.zkStatus}\tWORM:${e.worm}\t${e.signature}`
    ).join("\n")
    const blob = new Blob(
      [`# WORM AUDIT LOG EXPORT\n# Generated: ${new Date().toISOString()}\n# Entries: ${filtered.length}\n# Integrity: SHA3-256 HASH CHAIN\n\n`, content],
      { type: "text/plain" }
    )
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `worm_audit_${Date.now()}.log`
    a.click()
    URL.revokeObjectURL(url)
  }

  const zkVerified = entries.filter(e => e.zkStatus === "VERIFIED").length
  const zkPending = entries.filter(e => e.zkStatus === "PENDING").length
  const zkFailed = entries.filter(e => e.zkStatus === "FAILED").length

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* ZK proof status */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: "ZK VERIFIED", val: zkVerified, color: "text-status-ok", bg: "border-status-ok/30" },
          { label: "ZK PENDING", val: zkPending, color: "text-status-warn", bg: "border-status-warn/30" },
          { label: "ZK FAILED", val: zkFailed, color: "text-status-critical", bg: "border-status-critical/30" },
        ].map(s => (
          <div key={s.label} className={`bg-card border rounded-sm px-2 py-1.5 text-center ${s.bg}`}>
            <div className={`text-base font-mono font-bold ${s.color}`}>{s.val}</div>
            <div className="text-[8px] font-mono text-muted-foreground tracking-widest">{s.label}</div>
          </div>
        ))}
      </div>

      {/* ZK chain integrity indicator */}
      <div className="flex items-center gap-2 bg-card border border-border rounded-sm px-3 py-2">
        <span className="w-2 h-2 rounded-full bg-status-ok animate-pulse" />
        <span className="text-[9px] font-mono text-muted-foreground">HASH CHAIN INTEGRITY</span>
        <span className="text-[9px] font-mono text-status-ok font-bold">VERIFIED — SHA3-256 MERKLE</span>
        <span className="ml-auto text-[9px] font-mono text-muted-foreground">WORM: IMMUTABLE</span>
        <span className="w-1.5 h-1.5 rounded-full bg-status-ok ml-1" />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 flex-wrap">
        <input
          type="search"
          placeholder="SEARCH ACTOR / ACTION / ID..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="flex-1 min-w-32 bg-muted border border-border rounded-sm px-2 py-1 text-[10px] font-mono text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary"
          aria-label="Search audit log"
        />
        <select
          value={typeFilter}
          onChange={e => setTypeFilter(e.target.value as EventType | "ALL")}
          className="bg-muted border border-border rounded-sm px-2 py-1 text-[10px] font-mono text-foreground focus:outline-none focus:border-primary"
          aria-label="Filter by event type"
        >
          <option value="ALL">ALL TYPES</option>
          {(["DECISION", "POLICY", "ACCESS", "KILL_SWITCH", "ATTESTATION", "WORKFLOW", "ANOMALY"] as EventType[]).map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <select
          value={zkFilter}
          onChange={e => setZkFilter(e.target.value as ZKStatus | "ALL")}
          className="bg-muted border border-border rounded-sm px-2 py-1 text-[10px] font-mono text-foreground focus:outline-none focus:border-primary"
          aria-label="Filter by ZK status"
        >
          <option value="ALL">ALL ZK</option>
          {(["VERIFIED", "PENDING", "FAILED", "NOT_REQUIRED"] as ZKStatus[]).map(z => (
            <option key={z} value={z}>{z}</option>
          ))}
        </select>
        <button
          onClick={() => setStreaming(s => !s)}
          className={`text-[9px] font-mono px-2 py-1 rounded-sm border transition-colors
            ${streaming ? "border-status-ok text-status-ok" : "border-border text-muted-foreground"}`}
          aria-pressed={streaming}
        >
          {streaming ? "● LIVE" : "○ PAUSED"}
        </button>
        <button
          onClick={handleExport}
          className="text-[9px] font-mono px-2 py-1 rounded-sm border border-primary text-primary hover:bg-primary hover:text-primary-foreground transition-colors"
          aria-label="Export WORM audit log"
        >
          EXPORT WORM
        </button>
      </div>

      {/* Log table */}
      <div className="flex-1 overflow-auto bg-card border border-border rounded-sm">
        <table className="w-full text-[9px] font-mono" role="table" aria-label="WORM audit log">
          <thead className="sticky top-0 bg-card border-b border-border z-10">
            <tr>
              {["ID", "TIMESTAMP", "TYPE", "ACTOR", "ACTION", "ZK", "WORM"].map(h => (
                <th key={h} className="text-left px-2 py-1.5 text-[8px] font-mono text-muted-foreground tracking-widest font-normal border-r border-border last:border-0">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((entry, i) => (
              <tr
                key={entry.id}
                className={`border-b border-border last:border-0 cursor-pointer transition-colors
                  ${selectedEntry?.id === entry.id ? "bg-primary/10" : i % 2 === 0 ? "bg-transparent" : "bg-muted/20"}
                  hover:bg-secondary`}
                onClick={() => setSelectedEntry(selectedEntry?.id === entry.id ? null : entry)}
                aria-selected={selectedEntry?.id === entry.id}
              >
                <td className="px-2 py-1 text-muted-foreground">{entry.id}</td>
                <td className="px-2 py-1 text-muted-foreground whitespace-nowrap">{formatTs(entry.ts)}</td>
                <td className={`px-2 py-1 font-semibold ${TYPE_COLOR[entry.type]}`}>{entry.type}</td>
                <td className="px-2 py-1 text-foreground">{entry.actor}</td>
                <td className="px-2 py-1 text-foreground">{entry.action}</td>
                <td className={`px-2 py-1 font-bold ${ZK_COLOR[entry.zkStatus]}`}>{ZK_ICON[entry.zkStatus]}</td>
                <td className="px-2 py-1">
                  <span className={`${entry.worm ? "text-status-ok" : "text-status-critical"}`}>{entry.worm ? "WORM" : "MUTABLE"}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div ref={bottomRef} />
      </div>

      {/* Entry detail */}
      {selectedEntry && (
        <div className="bg-card border border-primary/40 rounded-sm p-3 text-[9px] font-mono">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <div><span className="text-muted-foreground">ID: </span><span className="text-foreground">{selectedEntry.id}</span></div>
            <div><span className="text-muted-foreground">TIMESTAMP: </span><span className="text-foreground">{formatTs(selectedEntry.ts)}</span></div>
            <div><span className="text-muted-foreground">ACTOR: </span><span className="text-foreground">{selectedEntry.actor}</span></div>
            <div><span className="text-muted-foreground">ACTION: </span><span className="text-foreground">{selectedEntry.action}</span></div>
            <div className="col-span-2"><span className="text-muted-foreground">RESOURCE: </span><span className="text-foreground">{selectedEntry.resource}</span></div>
            <div className="col-span-2"><span className="text-muted-foreground">HASH: </span><span className="text-foreground">{selectedEntry.hash}</span></div>
            <div><span className="text-muted-foreground">SIGNATURE: </span><span className="text-foreground">{selectedEntry.signature}</span></div>
            <div><span className="text-muted-foreground">ZK-PROOF: </span><span className={ZK_COLOR[selectedEntry.zkStatus]}>{selectedEntry.zkStatus}</span></div>
          </div>
        </div>
      )}
    </div>
  )
}
