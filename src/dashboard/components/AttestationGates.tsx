"use client"

import { useState } from "react"

type GateStatus = "LOCKED" | "PENDING_REVIEW" | "ATTESTED" | "FAILED" | "BYPASSED"
type ChallengeType = "CAPTCHA" | "BIOMETRIC" | "MFA_TOKEN" | "SUPERVISOR_APPROVAL" | "QUORUM"

interface AttestationGate {
  id: string
  name: string
  description: string
  status: GateStatus
  challengeType: ChallengeType
  requiredConfidence: number
  currentConfidence: number
  blockedOperations: string[]
  lastAttempt: string | null
  attempts: number
  maxAttempts: number
  ttl: string
  escalationPath: string
}

const GATES: AttestationGate[] = [
  {
    id: "CAG-001",
    name: "High-Risk Model Deployment",
    description: "Cognitive gate for deploying any EU AI Act Article 6(2) high-risk model to production.",
    status: "LOCKED",
    challengeType: "SUPERVISOR_APPROVAL",
    requiredConfidence: 95,
    currentConfidence: 0,
    blockedOperations: ["model.deploy", "model.promote", "inference.enable"],
    lastAttempt: null,
    attempts: 0,
    maxAttempts: 3,
    ttl: "—",
    escalationPath: "CISO → AI Ethics Board",
  },
  {
    id: "CAG-002",
    name: "Training Data Access",
    description: "Attestation required before accessing PII-containing training datasets.",
    status: "ATTESTED",
    challengeType: "MFA_TOKEN",
    requiredConfidence: 80,
    currentConfidence: 92,
    blockedOperations: ["data.read_pii", "data.export"],
    lastAttempt: "2025-06-15T09:01:12Z",
    attempts: 1,
    maxAttempts: 5,
    ttl: "47m remaining",
    escalationPath: "DPO",
  },
  {
    id: "CAG-003",
    name: "Kill-Switch Override",
    description: "Override gate for re-enabling a globally terminated agent cluster.",
    status: "PENDING_REVIEW",
    challengeType: "QUORUM",
    requiredConfidence: 99,
    currentConfidence: 66,
    blockedOperations: ["kill_switch.revert", "agent.reinstate"],
    lastAttempt: "2025-06-15T09:10:44Z",
    attempts: 2,
    maxAttempts: 3,
    ttl: "Awaiting quorum (2/3)",
    escalationPath: "3-of-3 Executive Quorum",
  },
  {
    id: "CAG-004",
    name: "WORM Log Export Authorization",
    description: "Exports of WORM audit logs to external systems require attestation.",
    status: "ATTESTED",
    challengeType: "MFA_TOKEN",
    requiredConfidence: 75,
    currentConfidence: 88,
    blockedOperations: ["audit.export_external"],
    lastAttempt: "2025-06-15T08:52:30Z",
    attempts: 1,
    maxAttempts: 5,
    ttl: "23m remaining",
    escalationPath: "Internal Audit → CISO",
  },
  {
    id: "CAG-005",
    name: "Policy Rule Modification",
    description: "Any modification to active policy rules requires a human cognitive review.",
    status: "FAILED",
    challengeType: "SUPERVISOR_APPROVAL",
    requiredConfidence: 90,
    currentConfidence: 0,
    blockedOperations: ["policy.modify", "policy.delete", "policy.disable"],
    lastAttempt: "2025-06-15T07:44:18Z",
    attempts: 3,
    maxAttempts: 3,
    ttl: "LOCKED OUT — 24h cooldown",
    escalationPath: "AI Governance Committee",
  },
]

const GATE_STATUS_COLOR: Record<GateStatus, string> = {
  LOCKED: "text-muted-foreground",
  PENDING_REVIEW: "text-status-warn",
  ATTESTED: "text-status-ok",
  FAILED: "text-status-critical",
  BYPASSED: "text-chart-5",
}

const GATE_STATUS_BG: Record<GateStatus, string> = {
  LOCKED: "border-border",
  PENDING_REVIEW: "border-status-warn/40",
  ATTESTED: "border-status-ok/30",
  FAILED: "border-status-critical/40",
  BYPASSED: "border-chart-5/40",
}

const CHALLENGE_LABELS: Record<ChallengeType, string> = {
  CAPTCHA: "CAPTCHA",
  BIOMETRIC: "BIOMETRIC",
  MFA_TOKEN: "MFA TOKEN",
  SUPERVISOR_APPROVAL: "SUPERVISOR",
  QUORUM: "QUORUM",
}

function ConfidenceMeter({ value, required }: { value: number; required: number }) {
  const pct = Math.min(100, value)
  const reqPct = required
  const color = value >= required
    ? "oklch(0.68 0.18 145)"
    : value > required * 0.5
    ? "oklch(0.78 0.17 75)"
    : "oklch(0.58 0.22 25)"

  return (
    <div className="relative h-2 bg-muted rounded-full overflow-visible">
      <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      {/* Required threshold marker */}
      <div
        className="absolute top-0 h-2 w-px bg-foreground/60"
        style={{ left: `${reqPct}%` }}
        title={`Required: ${reqPct}%`}
      />
    </div>
  )
}

interface GateChallengeModalProps {
  gate: AttestationGate
  onClose: () => void
  onAttest: (id: string) => void
}

function GateChallengeModal({ gate, onClose, onAttest }: GateChallengeModalProps) {
  const [step, setStep] = useState<"challenge" | "complete">("challenge")
  const [token, setToken] = useState("")
  const [approver, setApprover] = useState("")
  const [error, setError] = useState("")

  const handleSubmit = () => {
    if (gate.challengeType === "MFA_TOKEN" && token.length < 6) {
      setError("Token must be at least 6 characters.")
      return
    }
    if (gate.challengeType === "SUPERVISOR_APPROVAL" && !approver.trim()) {
      setError("Approver ID is required.")
      return
    }
    setStep("complete")
    setTimeout(() => { onAttest(gate.id); onClose() }, 1200)
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      role="dialog"
      aria-modal="true"
      aria-labelledby={`gate-modal-title-${gate.id}`}
    >
      <div className="bg-card border border-primary/50 rounded-sm w-full max-w-md mx-4 overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-secondary">
          <span className="w-2 h-2 rounded-full bg-status-warn animate-pulse" />
          <span id={`gate-modal-title-${gate.id}`} className="text-[10px] font-mono font-bold text-foreground">
            COGNITIVE ATTESTATION GATE — {gate.id}
          </span>
          <button onClick={onClose} className="ml-auto text-muted-foreground hover:text-foreground text-lg leading-none" aria-label="Close">×</button>
        </div>
        <div className="p-4 flex flex-col gap-4">
          <div>
            <div className="text-xs font-mono font-bold text-foreground mb-1">{gate.name}</div>
            <p className="text-[9px] font-mono text-muted-foreground leading-relaxed">{gate.description}</p>
          </div>

          <div className="bg-muted rounded-sm px-3 py-2">
            <div className="text-[8px] font-mono text-muted-foreground mb-1">CHALLENGE TYPE</div>
            <div className="text-[10px] font-mono font-bold text-accent">{CHALLENGE_LABELS[gate.challengeType]}</div>
          </div>

          {step === "challenge" && (
            <div className="flex flex-col gap-3">
              {gate.challengeType === "MFA_TOKEN" && (
                <div>
                  <label className="text-[9px] font-mono text-muted-foreground block mb-1" htmlFor={`mfa-${gate.id}`}>
                    ENTER OTP / HARDWARE TOKEN
                  </label>
                  <input
                    id={`mfa-${gate.id}`}
                    type="text"
                    value={token}
                    onChange={e => setToken(e.target.value)}
                    maxLength={12}
                    className="w-full bg-muted border border-border rounded-sm px-2 py-1.5 text-[11px] font-mono text-foreground tracking-widest focus:outline-none focus:border-primary"
                    placeholder="000000"
                    autoFocus
                  />
                </div>
              )}
              {gate.challengeType === "SUPERVISOR_APPROVAL" && (
                <div>
                  <label className="text-[9px] font-mono text-muted-foreground block mb-1" htmlFor={`approver-${gate.id}`}>
                    SUPERVISOR EMPLOYEE ID
                  </label>
                  <input
                    id={`approver-${gate.id}`}
                    type="text"
                    value={approver}
                    onChange={e => setApprover(e.target.value)}
                    className="w-full bg-muted border border-border rounded-sm px-2 py-1.5 text-[11px] font-mono text-foreground focus:outline-none focus:border-primary"
                    placeholder="EMP-XXXXXX"
                    autoFocus
                  />
                </div>
              )}
              {gate.challengeType === "QUORUM" && (
                <div className="text-[9px] font-mono text-muted-foreground">
                  Quorum attestation requires approval from <span className="text-foreground font-bold">3 designated principals</span>. Current approvals: <span className="text-status-warn font-bold">2/3</span>. Awaiting final approval from Principal-C.
                </div>
              )}
              {gate.challengeType === "BIOMETRIC" && (
                <div className="text-[9px] font-mono text-muted-foreground">
                  Biometric challenge requires hardware authenticator. Please use your registered FIDO2 device.
                </div>
              )}

              {error && <div className="text-[9px] font-mono text-status-critical">{error}</div>}

              <div className="flex gap-2">
                <button
                  onClick={handleSubmit}
                  className="flex-1 text-[9px] font-mono py-2 rounded-sm bg-primary text-primary-foreground font-bold hover:opacity-90 transition-opacity"
                >
                  SUBMIT ATTESTATION
                </button>
                <button onClick={onClose} className="text-[9px] font-mono py-2 px-3 rounded-sm border border-border text-muted-foreground hover:text-foreground transition-colors">
                  CANCEL
                </button>
              </div>
            </div>
          )}

          {step === "complete" && (
            <div className="text-center py-4">
              <div className="text-lg font-mono text-status-ok font-bold mb-1">ATTESTED</div>
              <div className="text-[9px] font-mono text-muted-foreground">Attestation verified. Gate unlocking...</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export function AttestationGates() {
  const [gates, setGates] = useState<AttestationGate[]>(GATES)
  const [activeGate, setActiveGate] = useState<AttestationGate | null>(null)

  const handleAttest = (id: string) => {
    setGates(prev => prev.map(g => g.id === id
      ? { ...g, status: "ATTESTED", currentConfidence: g.requiredConfidence + 5, lastAttempt: new Date().toISOString(), attempts: g.attempts + 1, ttl: "60m remaining" }
      : g
    ))
  }

  const locked = gates.filter(g => g.status === "LOCKED" || g.status === "FAILED").length
  const attested = gates.filter(g => g.status === "ATTESTED").length
  const pending = gates.filter(g => g.status === "PENDING_REVIEW").length

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: "ATTESTED", val: attested, color: "text-status-ok" },
          { label: "PENDING", val: pending, color: "text-status-warn" },
          { label: "LOCKED/FAILED", val: locked, color: "text-status-critical" },
        ].map(s => (
          <div key={s.label} className="bg-card border border-border rounded-sm px-2 py-1.5 text-center">
            <div className={`text-base font-mono font-bold ${s.color}`}>{s.val}</div>
            <div className="text-[8px] font-mono text-muted-foreground tracking-widest">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Gate list */}
      <div className="flex flex-col gap-2 overflow-auto">
        {gates.map(gate => (
          <div key={gate.id} className={`bg-card border rounded-sm px-3 py-2.5 ${GATE_STATUS_BG[gate.status]}`}>
            <div className="flex items-start gap-2 mb-2">
              <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[9px] font-mono text-muted-foreground">{gate.id}</span>
                  <span className={`text-[8px] font-mono font-bold border rounded-sm px-1 ${GATE_STATUS_BG[gate.status]} ${GATE_STATUS_COLOR[gate.status]} border-current`}>
                    {gate.status}
                  </span>
                  <span className="ml-auto text-[8px] font-mono border border-border text-muted-foreground px-1 rounded-sm">{CHALLENGE_LABELS[gate.challengeType]}</span>
                </div>
                <div className="text-[10px] font-mono font-semibold text-foreground">{gate.name}</div>
                <p className="text-[9px] font-mono text-muted-foreground leading-relaxed">{gate.description}</p>
              </div>
            </div>

            {/* Confidence meter */}
            <div className="mb-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[8px] font-mono text-muted-foreground">CONFIDENCE</span>
                <span className="text-[8px] font-mono text-muted-foreground">{gate.currentConfidence}% / {gate.requiredConfidence}% req.</span>
              </div>
              <ConfidenceMeter value={gate.currentConfidence} required={gate.requiredConfidence} />
            </div>

            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 mb-2">
              <div className="text-[8px] font-mono"><span className="text-muted-foreground">TTL: </span><span className="text-foreground">{gate.ttl}</span></div>
              <div className="text-[8px] font-mono"><span className="text-muted-foreground">ATTEMPTS: </span><span className="text-foreground">{gate.attempts}/{gate.maxAttempts}</span></div>
              <div className="text-[8px] font-mono col-span-2"><span className="text-muted-foreground">ESCALATION: </span><span className="text-foreground">{gate.escalationPath}</span></div>
            </div>

            <div className="mb-2">
              <div className="text-[8px] font-mono text-muted-foreground mb-0.5">BLOCKED OPS:</div>
              <div className="flex flex-wrap gap-1">
                {gate.blockedOperations.map(op => (
                  <span key={op} className="text-[8px] font-mono border border-status-critical/40 text-status-critical px-1 rounded-sm">{op}</span>
                ))}
              </div>
            </div>

            {(gate.status === "LOCKED" || gate.status === "PENDING_REVIEW") && gate.attempts < gate.maxAttempts && (
              <button
                onClick={() => setActiveGate(gate)}
                className="text-[9px] font-mono px-3 py-1 rounded-sm border border-primary text-primary hover:bg-primary hover:text-primary-foreground transition-colors"
                aria-label={`Initiate attestation for ${gate.name}`}
              >
                INITIATE ATTESTATION
              </button>
            )}
            {gate.status === "FAILED" && (
              <span className="text-[9px] font-mono text-status-critical">LOCKED OUT — MAX ATTEMPTS REACHED</span>
            )}
          </div>
        ))}
      </div>

      {/* Challenge modal */}
      {activeGate && (
        <GateChallengeModal
          gate={activeGate}
          onClose={() => setActiveGate(null)}
          onAttest={handleAttest}
        />
      )}
    </div>
  )
}
