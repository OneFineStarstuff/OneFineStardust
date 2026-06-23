"use client"

import { useState, useEffect, useRef } from "react"

type AgentStatus = "ACTIVE" | "SUSPENDED" | "TERMINATED" | "INITIALIZING"
type HwStatus = "ATTESTED" | "DEGRADED" | "UNVERIFIED" | "OFFLINE"

interface AgentEntry {
  id: string
  name: string
  status: AgentStatus
  model: string
  cpu: number
  mem: number
  reqPerMin: number
}

interface HardwareNode {
  id: string
  hostname: string
  tpm: HwStatus
  sgx: HwStatus
  firmware: string
  lastAttest: string
  measurements: string
}

const INITIAL_AGENTS: AgentEntry[] = [
  { id: "AGT-001", name: "PrimaryOrchestrator", status: "ACTIVE", model: "gpt-4o", cpu: 34, mem: 62, reqPerMin: 128 },
  { id: "AGT-002", name: "SafetySupervisor", status: "ACTIVE", model: "claude-3-5", cpu: 12, mem: 38, reqPerMin: 64 },
  { id: "AGT-003", name: "VisionPipeline", status: "ACTIVE", model: "gpt-4o-vision", cpu: 78, mem: 85, reqPerMin: 32 },
  { id: "AGT-004", name: "RAGRetriever", status: "SUSPENDED", model: "ada-embed-3", cpu: 0, mem: 18, reqPerMin: 0 },
  { id: "AGT-005", name: "DecisionAuditor", status: "ACTIVE", model: "mistral-8x7b", cpu: 22, mem: 44, reqPerMin: 256 },
  { id: "AGT-006", name: "PolicyEnforcer", status: "ACTIVE", model: "llama-3.1-70b", cpu: 45, mem: 71, reqPerMin: 512 },
]

const HW_NODES: HardwareNode[] = [
  { id: "HW-001", hostname: "gpu-node-01", tpm: "ATTESTED", sgx: "ATTESTED", firmware: "v2.4.1", lastAttest: "00:04:22", measurements: "sha256:a3f9c1..." },
  { id: "HW-002", hostname: "gpu-node-02", tpm: "ATTESTED", sgx: "DEGRADED", firmware: "v2.4.0", lastAttest: "00:12:08", measurements: "sha256:b7d2e4..." },
  { id: "HW-003", hostname: "cpu-node-01", tpm: "ATTESTED", sgx: "ATTESTED", firmware: "v2.4.1", lastAttest: "00:02:51", measurements: "sha256:c9f1a7..." },
  { id: "HW-004", hostname: "cpu-node-02", tpm: "UNVERIFIED", sgx: "OFFLINE", firmware: "v2.3.9", lastAttest: "02:41:09", measurements: "sha256:—" },
]

const STATUS_COLOR: Record<AgentStatus, string> = {
  ACTIVE: "text-status-ok",
  SUSPENDED: "text-status-warn",
  TERMINATED: "text-status-critical",
  INITIALIZING: "text-primary",
}

const HW_COLOR: Record<HwStatus, string> = {
  ATTESTED: "text-status-ok",
  DEGRADED: "text-status-warn",
  UNVERIFIED: "text-status-critical",
  OFFLINE: "text-status-inactive",
}

const HW_DOT: Record<HwStatus, string> = {
  ATTESTED: "bg-status-ok",
  DEGRADED: "bg-status-warn",
  UNVERIFIED: "bg-status-critical",
  OFFLINE: "bg-status-inactive",
}

function PulsingDot({ active }: { active: boolean }) {
  return (
    <span className="relative flex h-2.5 w-2.5">
      {active && (
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-critical opacity-75" />
      )}
      <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${active ? "bg-status-critical" : "bg-status-ok"}`} />
    </span>
  )
}

function BarMeter({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
      <span className="text-[9px] font-mono text-muted-foreground w-6 text-right">{value}%</span>
    </div>
  )
}

interface KillSwitchButtonProps {
  agentId: string
  currentStatus: AgentStatus
  onAction: (id: string, action: AgentStatus) => void
}

function KillSwitchButton({ agentId, currentStatus, onAction }: KillSwitchButtonProps) {
  const [confirming, setConfirming] = useState(false)
  const [targetAction, setTargetAction] = useState<AgentStatus | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleClick = (action: AgentStatus) => {
    if (confirming && targetAction === action) {
      onAction(agentId, action)
      setConfirming(false)
      setTargetAction(null)
      if (timerRef.current) clearTimeout(timerRef.current)
    } else {
      setConfirming(true)
      setTargetAction(action)
      timerRef.current = setTimeout(() => { setConfirming(false); setTargetAction(null) }, 3000)
    }
  }

  if (currentStatus === "TERMINATED") {
    return <span className="text-[9px] font-mono text-status-inactive">TERMINATED</span>
  }

  return (
    <div className="flex items-center gap-1">
      {currentStatus === "ACTIVE" && (
        <button
          onClick={() => handleClick("SUSPENDED")}
          className={`text-[8px] font-mono px-1.5 py-0.5 rounded-sm border transition-colors
            ${confirming && targetAction === "SUSPENDED"
              ? "bg-status-warn text-black border-status-warn animate-pulse"
              : "border-status-warn text-status-warn hover:bg-status-warn hover:text-black"
            }`}
          aria-label={`Suspend agent ${agentId}`}
        >
          {confirming && targetAction === "SUSPENDED" ? "CONFIRM?" : "SUSPEND"}
        </button>
      )}
      {currentStatus === "SUSPENDED" && (
        <button
          onClick={() => handleClick("ACTIVE")}
          className="text-[8px] font-mono px-1.5 py-0.5 rounded-sm border border-status-ok text-status-ok hover:bg-status-ok hover:text-black transition-colors"
          aria-label={`Resume agent ${agentId}`}
        >
          RESUME
        </button>
      )}
      <button
        onClick={() => handleClick("TERMINATED")}
        className={`text-[8px] font-mono px-1.5 py-0.5 rounded-sm border transition-colors
          ${confirming && targetAction === "TERMINATED"
            ? "bg-status-critical text-white border-status-critical animate-pulse"
            : "border-status-critical text-status-critical hover:bg-status-critical hover:text-white"
          }`}
        aria-label={`Terminate agent ${agentId}`}
      >
        {confirming && targetAction === "TERMINATED" ? "CONFIRM KILL?" : "KILL"}
      </button>
    </div>
  )
}

export function OperationalCockpit() {
  const [agents, setAgents] = useState<AgentEntry[]>(INITIAL_AGENTS)
  const [globalKill, setGlobalKill] = useState(false)
  const [globalKillConfirm, setGlobalKillConfirm] = useState(false)
  const [activeTab, setActiveTab] = useState<"agents" | "hardware">("agents")
  const [systemTime, setSystemTime] = useState("")
  const globalKillTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    const tick = () => setSystemTime(new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC")
    tick()
    const interval = setInterval(tick, 1000)
    return () => clearInterval(interval)
  }, [])

  // Simulate live metrics
  useEffect(() => {
    if (globalKill) return
    const interval = setInterval(() => {
      setAgents(prev => prev.map(a => {
        if (a.status !== "ACTIVE") return a
        return {
          ...a,
          cpu: Math.max(5, Math.min(99, a.cpu + (Math.random() - 0.5) * 10)),
          mem: Math.max(10, Math.min(99, a.mem + (Math.random() - 0.5) * 5)),
          reqPerMin: Math.max(0, Math.round(a.reqPerMin + (Math.random() - 0.5) * 30)),
        }
      }))
    }, 2000)
    return () => clearInterval(interval)
  }, [globalKill])

  const handleAgentAction = (id: string, action: AgentStatus) => {
    setAgents(prev => prev.map(a => a.id === id ? { ...a, status: action, cpu: action === "TERMINATED" ? 0 : a.cpu, mem: action === "TERMINATED" ? 0 : a.mem, reqPerMin: action !== "ACTIVE" ? 0 : a.reqPerMin } : a))
  }

  const handleGlobalKill = () => {
    if (globalKillConfirm) {
      setGlobalKill(true)
      setAgents(prev => prev.map(a => ({ ...a, status: "TERMINATED" as AgentStatus, cpu: 0, mem: 0, reqPerMin: 0 })))
      setGlobalKillConfirm(false)
      if (globalKillTimerRef.current) clearTimeout(globalKillTimerRef.current)
    } else {
      setGlobalKillConfirm(true)
      globalKillTimerRef.current = setTimeout(() => setGlobalKillConfirm(false), 4000)
    }
  }

  const activeCount = agents.filter(a => a.status === "ACTIVE").length
  const criticalCount = agents.filter(a => a.cpu > 80 || a.mem > 80).length

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* System status bar */}
      <div className="flex items-center gap-2 bg-card border border-border rounded-sm px-3 py-2">
        <PulsingDot active={criticalCount > 0 && !globalKill} />
        <span className="text-[9px] font-mono text-muted-foreground">SYS</span>
        <span className={`text-[10px] font-mono font-bold ${globalKill ? "text-status-critical" : criticalCount > 0 ? "text-status-warn" : "text-status-ok"}`}>
          {globalKill ? "GLOBAL KILL ACTIVE" : criticalCount > 0 ? `${criticalCount} AGENT(S) CRITICAL` : "NOMINAL"}
        </span>
        <span className="ml-auto text-[9px] font-mono text-muted-foreground">{systemTime}</span>
        <span className="text-[9px] font-mono text-muted-foreground border-l border-border pl-2">
          {activeCount}/{agents.length} ACTIVE
        </span>
      </div>

      {/* Global kill switch */}
      <div className={`flex items-center gap-3 border rounded-sm px-3 py-2.5 ${globalKill ? "border-status-critical bg-status-critical/10" : "border-border bg-card"}`}>
        <div className="flex flex-col">
          <span className="text-[10px] font-mono font-bold text-foreground">GLOBAL EMERGENCY KILL SWITCH</span>
          <span className="text-[9px] font-mono text-muted-foreground">Immediately terminates all active agents. Action is irreversible for this session.</span>
        </div>
        <button
          onClick={handleGlobalKill}
          disabled={globalKill}
          className={`ml-auto shrink-0 px-4 py-2 text-[10px] font-mono font-bold rounded-sm border-2 transition-all
            ${globalKill
              ? "border-status-inactive text-status-inactive bg-muted cursor-not-allowed"
              : globalKillConfirm
              ? "border-status-critical bg-status-critical text-white animate-pulse scale-105"
              : "border-status-critical text-status-critical hover:bg-status-critical hover:text-white"
            }`}
          aria-label="Global kill switch"
        >
          {globalKill ? "TERMINATED" : globalKillConfirm ? "CONFIRM GLOBAL KILL" : "KILL ALL"}
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border">
        {(["agents", "hardware"] as const).map(t => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={`px-3 py-1.5 text-[10px] font-mono tracking-widest uppercase transition-colors
              ${activeTab === t ? "text-primary border-b-2 border-primary" : "text-muted-foreground hover:text-foreground"}`}
          >
            {t === "agents" ? "AGENT ROSTER" : "HARDWARE ATTESTATION"}
          </button>
        ))}
      </div>

      {/* Agent roster */}
      {activeTab === "agents" && (
        <div className="flex flex-col gap-1.5 overflow-auto">
          {agents.map(agent => (
            <div
              key={agent.id}
              className={`bg-card border rounded-sm px-3 py-2 transition-colors ${agent.status === "TERMINATED" ? "border-status-critical/30 opacity-60" : agent.status === "SUSPENDED" ? "border-status-warn/40" : "border-border"}`}
            >
              <div className="flex items-center gap-2 mb-1.5">
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${agent.status === "ACTIVE" ? "bg-status-ok animate-pulse" : agent.status === "SUSPENDED" ? "bg-status-warn" : agent.status === "TERMINATED" ? "bg-status-critical" : "bg-primary"}`} />
                <span className="text-[9px] font-mono text-muted-foreground">{agent.id}</span>
                <span className="text-[10px] font-mono font-semibold text-foreground">{agent.name}</span>
                <span className="text-[8px] font-mono border border-border text-muted-foreground px-1 rounded-sm">{agent.model}</span>
                <span className={`ml-auto text-[9px] font-mono font-bold ${STATUS_COLOR[agent.status]}`}>{agent.status}</span>
              </div>
              <div className="grid grid-cols-3 gap-3 mb-1.5">
                <div>
                  <div className="text-[8px] font-mono text-muted-foreground mb-0.5">CPU</div>
                  <BarMeter value={Math.round(agent.cpu)} color={agent.cpu > 80 ? "oklch(0.58 0.22 25)" : agent.cpu > 60 ? "oklch(0.78 0.17 75)" : "oklch(0.72 0.16 195)"} />
                </div>
                <div>
                  <div className="text-[8px] font-mono text-muted-foreground mb-0.5">MEM</div>
                  <BarMeter value={Math.round(agent.mem)} color={agent.mem > 80 ? "oklch(0.58 0.22 25)" : agent.mem > 60 ? "oklch(0.78 0.17 75)" : "oklch(0.68 0.18 145)"} />
                </div>
                <div>
                  <div className="text-[8px] font-mono text-muted-foreground mb-0.5">REQ/MIN</div>
                  <div className="text-[10px] font-mono text-foreground">{agent.reqPerMin}</div>
                </div>
              </div>
              <KillSwitchButton agentId={agent.id} currentStatus={agent.status} onAction={handleAgentAction} />
            </div>
          ))}
        </div>
      )}

      {/* Hardware attestation */}
      {activeTab === "hardware" && (
        <div className="flex flex-col gap-2 overflow-auto">
          {HW_NODES.map(hw => (
            <div key={hw.id} className={`bg-card border rounded-sm px-3 py-2.5 ${hw.tpm === "UNVERIFIED" || hw.sgx === "OFFLINE" ? "border-status-critical/50" : hw.sgx === "DEGRADED" ? "border-status-warn/50" : "border-border"}`}>
              <div className="flex items-center gap-2 mb-2">
                <span className={`w-1.5 h-1.5 rounded-full ${HW_DOT[hw.tpm]}`} />
                <span className="text-[9px] font-mono text-muted-foreground">{hw.id}</span>
                <span className="text-[10px] font-mono font-semibold text-foreground">{hw.hostname}</span>
                <span className="text-[8px] font-mono border border-border text-muted-foreground px-1 rounded-sm">{hw.firmware}</span>
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[9px] font-mono text-muted-foreground w-8">TPM</span>
                  <span className={`text-[9px] font-mono font-bold ${HW_COLOR[hw.tpm]}`}>{hw.tpm}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] font-mono text-muted-foreground w-8">SGX</span>
                  <span className={`text-[9px] font-mono font-bold ${HW_COLOR[hw.sgx]}`}>{hw.sgx}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] font-mono text-muted-foreground w-8">AGE</span>
                  <span className="text-[9px] font-mono text-foreground">{hw.lastAttest}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] font-mono text-muted-foreground w-8">PCR</span>
                  <span className="text-[9px] font-mono text-muted-foreground truncate">{hw.measurements}</span>
                </div>
              </div>
            </div>
          ))}
          <div className="bg-card border border-border rounded-sm px-3 py-2 flex items-center gap-2">
            <span className="text-[9px] font-mono text-muted-foreground">REMOTE ATTESTATION SERVICE:</span>
            <span className="text-[9px] font-mono text-status-ok font-bold">ONLINE</span>
            <span className="ml-auto text-[9px] font-mono text-muted-foreground">LAST CYCLE: 4m 22s AGO</span>
          </div>
        </div>
      )}
    </div>
  )
}
