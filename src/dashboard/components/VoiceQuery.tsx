"use client"

import { useState, useEffect, useRef, useCallback } from "react"

interface QueryResult {
  query: string
  answer: string
  ts: string
  source: "VOICE" | "TEXT"
  confidence: number
}

const CANNED_RESPONSES: Record<string, string> = {
  "show compliance": "Current compliance scores: EU AI Act 84%, DORA 89%, NIST AI RMF 83%. 2 critical gaps identified: SGX firmware CVE on HW-004 and model card refresh overdue by 17 days.",
  "kill switch status": "Global kill switch is INACTIVE. 5 of 6 agents are ACTIVE. Agent AGT-004 (RAGRetriever) is SUSPENDED pending data governance remediation.",
  "zk proof": "ZK-proof chain integrity: VERIFIED. 47 proofs verified in last hour. 3 pending generation. SHA3-256 Merkle root updated 4 minutes ago.",
  "hardware attestation": "HW-001: ATTESTED, HW-002: SGX DEGRADED, HW-003: ATTESTED, HW-004: UNVERIFIED (firmware v2.3.9, CVE-2025-1182 outstanding). Remote attestation service ONLINE.",
  "audit log": "WORM audit log contains 10,847 entries. Last entry 1.8 seconds ago. Hash chain integrity VERIFIED. 0 tamper events detected in last 30 days.",
  "workflow recommendations": "5 active recommendations: 2 CRITICAL (model card refresh, HW-004 firmware), 2 HIGH (bias evaluation, consent verification), 1 MEDIUM (differential privacy). 1 accepted, 1 dismissed.",
  "attestation gates": "5 cognitive attestation gates: 2 ATTESTED, 1 PENDING_REVIEW (kill-switch override, awaiting quorum 2/3), 1 LOCKED (high-risk deployment), 1 FAILED (policy modification, max attempts reached).",
  "agent status": "6 agents registered: AGT-001 PrimaryOrchestrator ACTIVE, AGT-002 SafetySupervisor ACTIVE, AGT-003 VisionPipeline ACTIVE (CPU 78%), AGT-004 RAGRetriever SUSPENDED, AGT-005 DecisionAuditor ACTIVE, AGT-006 PolicyEnforcer ACTIVE.",
  "eu ai act": "EU AI Act compliance: 84% aggregate. Weakest domain: Accuracy & Robustness (72%). 3 high-risk systems classified under Annex III. 0 prohibited use cases active.",
  "dora": "DORA ICT resilience: 89% aggregate. Current RTO: 2.4h, RPO: 1.2h. Last incident: 1 minor (June). Threat-led TLPT result: FAIL — remediation plan in progress.",
  "nist": "NIST AI RMF v1.0: 83% aggregate. Strongest: IDENTIFY (92%). Weakest: MEASURE (71%). Current profile TIER-3, target TIER-4. Gap delta: 19%.",
  "default": "Query processed. Relevant data retrieved from governance knowledge base. Confidence score reflects semantic match to indexed compliance artifacts and live telemetry.",
}

function matchQuery(q: string): { answer: string; confidence: number } {
  const lower = q.toLowerCase()
  for (const [key, val] of Object.entries(CANNED_RESPONSES)) {
    if (key !== "default" && lower.includes(key)) {
      return { answer: val, confidence: 92 + Math.floor(Math.random() * 7) }
    }
  }
  return { answer: CANNED_RESPONSES.default + ` No exact match found for: "${q}". Try: "show compliance", "kill switch status", "audit log", "hardware attestation".`, confidence: 42 + Math.floor(Math.random() * 20) }
}

export function VoiceQuery() {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [textInput, setTextInput] = useState("")
  const [results, setResults] = useState<QueryResult[]>([])
  const [error, setError] = useState("")
  const [supported, setSupported] = useState(true)
  const recognitionRef = useRef<any>(null)

  useEffect(() => {
    if (typeof window === "undefined") return
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRec) {
      setSupported(false)
      return
    }
    const rec = new SpeechRec()
    rec.continuous = false
    rec.interimResults = true
    rec.lang = "en-US"

    rec.onresult = (event: any) => {
      const result = Array.from(event.results).map((r: any) => r[0].transcript).join("")
      setTranscript(result)
    }

    rec.onend = () => {
      setIsListening(false)
    }

    rec.onerror = (event: any) => {
      setError(`Speech error: ${event.error}`)
      setIsListening(false)
    }

    recognitionRef.current = rec
  }, [])

  const processQuery = useCallback((query: string, source: "VOICE" | "TEXT") => {
    if (!query.trim()) return
    const { answer, confidence } = matchQuery(query)
    const newResult: QueryResult = {
      query,
      answer,
      ts: new Date().toISOString().replace("T", " ").slice(0, 19),
      source,
      confidence,
    }
    setResults(prev => [newResult, ...prev].slice(0, 20))
    setTranscript("")
    setTextInput("")
  }, [])

  const toggleListening = () => {
    if (!supported || !recognitionRef.current) return
    if (isListening) {
      recognitionRef.current.stop()
      if (transcript) processQuery(transcript, "VOICE")
    } else {
      setError("")
      setTranscript("")
      try {
        recognitionRef.current.start()
        setIsListening(true)
      } catch {
        setError("Could not start recognition. Check microphone permissions.")
      }
    }
  }

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    processQuery(textInput, "TEXT")
  }

  const SUGGESTIONS = ["show compliance", "kill switch status", "hardware attestation", "audit log", "workflow recommendations", "attestation gates", "agent status"]

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Header */}
      <div className="flex items-center gap-2 bg-card border border-border rounded-sm px-3 py-2">
        <span className={`w-2 h-2 rounded-full ${isListening ? "bg-status-critical animate-ping" : "bg-muted-foreground"}`} />
        <span className="text-[9px] font-mono text-muted-foreground">VOICE AUDIT QUERY INTERFACE</span>
        <span className="ml-auto text-[8px] font-mono border border-border text-muted-foreground px-1 py-0.5 rounded-sm">
          {supported ? "WEB SPEECH API" : "NOT SUPPORTED"}
        </span>
      </div>

      {/* Voice button */}
      <div className="flex flex-col items-center gap-3 bg-card border border-border rounded-sm px-4 py-5">
        <button
          onClick={toggleListening}
          disabled={!supported}
          className={`relative w-16 h-16 rounded-full border-2 transition-all font-mono text-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-primary
            ${isListening
              ? "border-status-critical bg-status-critical/20 text-status-critical"
              : supported
              ? "border-primary bg-primary/10 text-primary hover:bg-primary/20"
              : "border-border text-muted-foreground cursor-not-allowed"
            }`}
          aria-label={isListening ? "Stop listening" : "Start voice query"}
          aria-pressed={isListening}
        >
          {isListening && (
            <span className="absolute inset-0 rounded-full border-2 border-status-critical animate-ping opacity-40" />
          )}
          {isListening ? "■" : "⬤"}
        </button>

        <div className="text-center">
          <div className={`text-[10px] font-mono font-bold ${isListening ? "text-status-critical" : "text-muted-foreground"}`}>
            {isListening ? "LISTENING..." : "PRESS TO SPEAK"}
          </div>
          {isListening && transcript && (
            <div className="text-[9px] font-mono text-foreground mt-1 max-w-xs text-center">{transcript}</div>
          )}
          {!supported && (
            <div className="text-[9px] font-mono text-status-warn mt-1">Browser does not support Web Speech API</div>
          )}
        </div>

        {/* Text input fallback */}
        <form onSubmit={handleTextSubmit} className="flex items-center gap-2 w-full">
          <input
            type="text"
            value={textInput}
            onChange={e => setTextInput(e.target.value)}
            placeholder="TYPE A QUERY..."
            className="flex-1 bg-muted border border-border rounded-sm px-2 py-1.5 text-[10px] font-mono text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary"
            aria-label="Text audit query"
          />
          <button
            type="submit"
            className="text-[9px] font-mono px-3 py-1.5 rounded-sm border border-primary text-primary hover:bg-primary hover:text-primary-foreground transition-colors"
          >
            QUERY
          </button>
        </form>

        {error && <div className="text-[9px] font-mono text-status-critical">{error}</div>}
      </div>

      {/* Quick suggestions */}
      <div className="flex flex-wrap gap-1">
        {SUGGESTIONS.map(s => (
          <button
            key={s}
            onClick={() => processQuery(s, "TEXT")}
            className="text-[8px] font-mono border border-border text-muted-foreground px-1.5 py-0.5 rounded-sm hover:border-primary hover:text-primary transition-colors"
          >
            {s}
          </button>
        ))}
      </div>

      {/* Results */}
      <div className="flex flex-col gap-2 overflow-auto flex-1">
        {results.length === 0 && (
          <div className="text-center py-6 text-[9px] font-mono text-muted-foreground">
            No queries yet. Use voice or text to query the governance knowledge base.
          </div>
        )}
        {results.map((r, i) => (
          <div key={i} className="bg-card border border-border rounded-sm p-3">
            <div className="flex items-center gap-2 mb-1.5">
              <span className={`text-[8px] font-mono font-bold border px-1 rounded-sm ${r.source === "VOICE" ? "border-primary text-primary" : "border-muted-foreground text-muted-foreground"}`}>
                {r.source}
              </span>
              <span className="text-[9px] font-mono text-foreground font-semibold">{r.query}</span>
              <span className="ml-auto text-[8px] font-mono text-muted-foreground">{r.ts}</span>
            </div>
            <p className="text-[9px] font-mono text-muted-foreground leading-relaxed mb-1.5">{r.answer}</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
                <div className="h-full rounded-full" style={{ width: `${r.confidence}%`, backgroundColor: r.confidence > 80 ? "oklch(0.68 0.18 145)" : r.confidence > 60 ? "oklch(0.78 0.17 75)" : "oklch(0.58 0.22 25)" }} />
              </div>
              <span className="text-[8px] font-mono text-muted-foreground">{r.confidence}% conf</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
