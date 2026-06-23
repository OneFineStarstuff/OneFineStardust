"use client"

import { useState } from "react"
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, Legend,
} from "recharts"

const EU_AI_ACT_DATA = [
  { category: "Risk Classification", score: 91, max: 100 },
  { category: "Transparency", score: 78, max: 100 },
  { category: "Human Oversight", score: 85, max: 100 },
  { category: "Data Governance", score: 88, max: 100 },
  { category: "Accuracy & Robustness", score: 72, max: 100 },
  { category: "Cybersecurity", score: 94, max: 100 },
]

const RADAR_DATA = [
  { subject: "Risk Mgmt", EU: 91, DORA: 84, NIST: 89 },
  { subject: "Transparency", EU: 78, DORA: 72, NIST: 81 },
  { subject: "Oversight", EU: 85, DORA: 91, NIST: 87 },
  { subject: "Data Gov", EU: 88, DORA: 79, NIST: 92 },
  { subject: "Resilience", EU: 74, DORA: 96, NIST: 83 },
  { subject: "Security", EU: 94, DORA: 89, NIST: 91 },
]

const DORA_TREND = [
  { month: "Jan", rto: 4.2, rpo: 2.1, incidents: 3 },
  { month: "Feb", rto: 3.8, rpo: 1.9, incidents: 2 },
  { month: "Mar", rto: 5.1, rpo: 2.4, incidents: 5 },
  { month: "Apr", rto: 3.2, rpo: 1.6, incidents: 1 },
  { month: "May", rto: 2.8, rpo: 1.4, incidents: 2 },
  { month: "Jun", rto: 2.4, rpo: 1.2, incidents: 1 },
]

const NIST_RMF_FUNCTIONS = [
  { function: "GOVERN", score: 87, color: "#22d3ee" },
  { function: "IDENTIFY", score: 92, color: "#4ade80" },
  { function: "MANAGE", score: 76, color: "#facc15" },
  { function: "MAP", score: 83, color: "#a78bfa" },
  { function: "MEASURE", score: 71, color: "#fb923c" },
]

const CUSTOM_TOOLTIP_STYLE = {
  backgroundColor: "oklch(0.13 0.008 235)",
  border: "1px solid oklch(0.22 0.010 235)",
  borderRadius: "2px",
  fontSize: "11px",
  color: "oklch(0.93 0.008 220)",
  padding: "6px 10px",
}

type TabKey = "overview" | "eu" | "dora" | "nist"

const TABS: { key: TabKey; label: string; badge: string; badgeColor: string }[] = [
  { key: "overview", label: "OVERVIEW", badge: "LIVE", badgeColor: "text-status-ok" },
  { key: "eu", label: "EU AI ACT", badge: "ART.9", badgeColor: "text-primary" },
  { key: "dora", label: "DORA", badge: "RTS-ICT", badgeColor: "text-accent" },
  { key: "nist", label: "NIST AI RMF", badge: "1.0", badgeColor: "text-chart-5" },
]

function ScoreGauge({ score, label, color }: { score: number; label: string; color: string }) {
  const r = 28
  const circ = 2 * Math.PI * r
  const strokeDash = (score / 100) * circ
  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="72" height="72" viewBox="0 0 72 72" role="img" aria-label={`${label}: ${score}%`}>
        <circle cx="36" cy="36" r={r} fill="none" stroke="oklch(0.22 0.010 235)" strokeWidth="5" />
        <circle
          cx="36" cy="36" r={r} fill="none"
          stroke={color} strokeWidth="5"
          strokeDasharray={`${strokeDash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 36 36)"
        />
        <text x="36" y="40" textAnchor="middle" fontSize="14" fontWeight="700" fill="oklch(0.93 0.008 220)" fontFamily="monospace">
          {score}
        </text>
      </svg>
      <span className="text-[10px] font-mono text-muted-foreground tracking-widest uppercase">{label}</span>
    </div>
  )
}

function SectionHeader({ children, tag }: { children: React.ReactNode; tag?: string }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <span className="w-1 h-4 bg-primary rounded-sm" />
      <h3 className="text-xs font-mono font-semibold text-foreground tracking-widest uppercase">{children}</h3>
      {tag && (
        <span className="ml-auto text-[9px] font-mono text-muted-foreground border border-border px-1.5 py-0.5 rounded-sm">
          {tag}
        </span>
      )}
    </div>
  )
}

function PanelCard({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-card border border-border rounded-sm p-3 ${className}`}>
      {children}
    </div>
  )
}

export function ComplianceDashboard() {
  const [activeTab, setActiveTab] = useState<TabKey>("overview")

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Tab bar */}
      <div className="flex items-center gap-0 border border-border rounded-sm overflow-hidden">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            className={`flex items-center gap-1.5 px-3 py-2 text-[10px] font-mono tracking-widest transition-colors flex-1 justify-center
              ${activeTab === t.key
                ? "bg-secondary text-primary border-r border-border"
                : "text-muted-foreground hover:text-foreground hover:bg-muted border-r border-border"
              }`}
          >
            {t.label}
            <span className={`text-[8px] ${t.badgeColor}`}>[{t.badge}]</span>
          </button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {activeTab === "overview" && (
        <div className="grid grid-cols-1 gap-3">
          {/* Score gauges */}
          <PanelCard>
            <SectionHeader tag="3 FRAMEWORKS">AGGREGATE COMPLIANCE SCORES</SectionHeader>
            <div className="flex items-center justify-around py-2">
              <ScoreGauge score={84} label="EU AI ACT" color="oklch(0.72 0.16 195)" />
              <ScoreGauge score={89} label="DORA" color="oklch(0.68 0.18 145)" />
              <ScoreGauge score={83} label="NIST RMF" color="oklch(0.62 0.20 280)" />
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2 text-center border-t border-border pt-2">
              <div>
                <div className="text-[9px] font-mono text-muted-foreground">OPEN GAPS</div>
                <div className="text-sm font-mono font-bold text-status-warn">14</div>
              </div>
              <div>
                <div className="text-[9px] font-mono text-muted-foreground">CRITICAL</div>
                <div className="text-sm font-mono font-bold text-status-critical">2</div>
              </div>
              <div>
                <div className="text-[9px] font-mono text-muted-foreground">LAST AUDIT</div>
                <div className="text-sm font-mono font-bold text-foreground">06h 14m</div>
              </div>
            </div>
          </PanelCard>

          {/* Radar overlay */}
          <PanelCard>
            <SectionHeader tag="SPIDER">CROSS-FRAMEWORK COVERAGE</SectionHeader>
            <ResponsiveContainer width="100%" height={200}>
              <RadarChart data={RADAR_DATA} margin={{ top: 5, right: 15, bottom: 5, left: 15 }}>
                <PolarGrid stroke="oklch(0.22 0.010 235)" />
                <PolarAngleAxis
                  dataKey="subject"
                  tick={{ fontSize: 9, fill: "oklch(0.55 0.010 220)", fontFamily: "monospace" }}
                />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Radar name="EU AI Act" dataKey="EU" stroke="oklch(0.72 0.16 195)" fill="oklch(0.72 0.16 195)" fillOpacity={0.15} strokeWidth={1.5} />
                <Radar name="DORA" dataKey="DORA" stroke="oklch(0.68 0.18 145)" fill="oklch(0.68 0.18 145)" fillOpacity={0.15} strokeWidth={1.5} />
                <Radar name="NIST" dataKey="NIST" stroke="oklch(0.62 0.20 280)" fill="oklch(0.62 0.20 280)" fillOpacity={0.15} strokeWidth={1.5} />
                <Legend wrapperStyle={{ fontSize: 9, fontFamily: "monospace" }} />
                <Tooltip contentStyle={CUSTOM_TOOLTIP_STYLE} />
              </RadarChart>
            </ResponsiveContainer>
          </PanelCard>
        </div>
      )}

      {/* EU AI ACT TAB */}
      {activeTab === "eu" && (
        <div className="flex flex-col gap-3">
          <PanelCard>
            <SectionHeader tag="ART.9/17">ARTICLE-LEVEL COMPLIANCE SCORES</SectionHeader>
            <div className="flex flex-col gap-1.5 mt-1">
              {EU_AI_ACT_DATA.map((item) => (
                <div key={item.category} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-muted-foreground w-40 shrink-0">{item.category}</span>
                  <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${item.score}%`,
                        backgroundColor: item.score >= 85
                          ? "oklch(0.68 0.18 145)"
                          : item.score >= 70
                          ? "oklch(0.78 0.17 75)"
                          : "oklch(0.58 0.22 25)",
                      }}
                    />
                  </div>
                  <span className="text-[10px] font-mono font-semibold text-foreground w-7 text-right">{item.score}</span>
                </div>
              ))}
            </div>
          </PanelCard>
          <PanelCard>
            <SectionHeader tag="HIGH-RISK">RISK CATEGORY CLASSIFICATION</SectionHeader>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Prohibited Uses", count: 0, color: "text-status-ok" },
                { label: "High-Risk Systems", count: 3, color: "text-status-warn" },
                { label: "Limited Risk", count: 7, color: "text-primary" },
                { label: "Minimal Risk", count: 12, color: "text-muted-foreground" },
              ].map((item) => (
                <div key={item.label} className="bg-muted rounded-sm px-2.5 py-2">
                  <div className={`text-lg font-mono font-bold ${item.color}`}>{item.count}</div>
                  <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider">{item.label}</div>
                </div>
              ))}
            </div>
          </PanelCard>
        </div>
      )}

      {/* DORA TAB */}
      {activeTab === "dora" && (
        <div className="flex flex-col gap-3">
          <PanelCard>
            <SectionHeader tag="RTO/RPO">ICT RESILIENCE METRICS (H)</SectionHeader>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={DORA_TREND} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.22 0.010 235)" />
                <XAxis dataKey="month" tick={{ fontSize: 9, fill: "oklch(0.55 0.010 220)", fontFamily: "monospace" }} />
                <YAxis tick={{ fontSize: 9, fill: "oklch(0.55 0.010 220)", fontFamily: "monospace" }} />
                <Tooltip contentStyle={CUSTOM_TOOLTIP_STYLE} />
                <Area type="monotone" dataKey="rto" name="RTO (h)" stroke="oklch(0.72 0.16 195)" fill="oklch(0.72 0.16 195)" fillOpacity={0.2} strokeWidth={1.5} />
                <Area type="monotone" dataKey="rpo" name="RPO (h)" stroke="oklch(0.78 0.17 75)" fill="oklch(0.78 0.17 75)" fillOpacity={0.2} strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          </PanelCard>
          <PanelCard>
            <SectionHeader tag="ICT-TESTING">TLPT CYCLE STATUS</SectionHeader>
            <div className="flex flex-col gap-1.5">
              {[
                { test: "Penetration Testing", status: "PASS", last: "2025-04-12" },
                { test: "Scenario-Based Testing", status: "PASS", last: "2025-03-28" },
                { test: "Red Team Exercises", status: "PENDING", last: "2025-01-15" },
                { test: "Threat-Led TLPT", status: "FAIL", last: "2025-05-01" },
              ].map((r) => (
                <div key={r.test} className="flex items-center gap-2 py-1 border-b border-border last:border-0">
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${r.status === "PASS" ? "bg-status-ok" : r.status === "PENDING" ? "bg-status-warn" : "bg-status-critical"}`} />
                  <span className="text-[10px] font-mono text-foreground flex-1">{r.test}</span>
                  <span className="text-[9px] font-mono text-muted-foreground">{r.last}</span>
                  <span className={`text-[9px] font-mono font-bold ${r.status === "PASS" ? "text-status-ok" : r.status === "PENDING" ? "text-status-warn" : "text-status-critical"}`}>
                    {r.status}
                  </span>
                </div>
              ))}
            </div>
          </PanelCard>
        </div>
      )}

      {/* NIST RMF TAB */}
      {activeTab === "nist" && (
        <div className="flex flex-col gap-3">
          <PanelCard>
            <SectionHeader tag="6 FUNCTIONS">NIST AI RMF FUNCTION SCORES</SectionHeader>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={NIST_RMF_FUNCTIONS} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.22 0.010 235)" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 9, fill: "oklch(0.55 0.010 220)", fontFamily: "monospace" }} />
                <YAxis type="category" dataKey="function" tick={{ fontSize: 9, fill: "oklch(0.55 0.010 220)", fontFamily: "monospace" }} width={60} />
                <Tooltip contentStyle={CUSTOM_TOOLTIP_STYLE} />
                <Bar dataKey="score" radius={[0, 2, 2, 0]} barSize={14}>
                  {NIST_RMF_FUNCTIONS.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </PanelCard>
          <PanelCard>
            <SectionHeader tag="PROFILES">ORGANIZATIONAL PROFILE ALIGNMENT</SectionHeader>
            <div className="flex flex-col gap-1">
              {[
                { profile: "Current Profile", tier: "TIER-3", pct: 76, color: "oklch(0.72 0.16 195)" },
                { profile: "Target Profile", tier: "TIER-4", pct: 95, color: "oklch(0.68 0.18 145)" },
                { profile: "Gap Delta", tier: "DELTA", pct: 19, color: "oklch(0.78 0.17 75)" },
              ].map((item) => (
                <div key={item.profile} className="flex items-center gap-2">
                  <span className="text-[9px] font-mono text-muted-foreground w-24 shrink-0">{item.profile}</span>
                  <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${item.pct}%`, backgroundColor: item.color }} />
                  </div>
                  <span className="text-[9px] font-mono text-muted-foreground w-10 text-right">{item.tier}</span>
                </div>
              ))}
            </div>
          </PanelCard>
        </div>
      )}
    </div>
  )
}
