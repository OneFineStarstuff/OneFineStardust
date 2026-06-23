import React from 'react';
import { checkLicenseFile } from '../utils/licenseCheck';
import { checkMetadataFiles } from '../utils/metadataCheck';
import { checkReleaseProcess } from '../utils/releaseCheck';

const FAIRComplianceDashboard = () => {
  const licenseStatus = checkLicenseFile();
  const { citationStatus, codemetaStatus } = checkMetadataFiles();
  const releaseStatus = checkReleaseProcess();

  return (
    <div className="fair-compliance-dashboard">
      <div className="p-4 border-b border-border bg-secondary/30 mb-6">
        <h2 className="text-xl font-mono font-bold tracking-tight text-primary">FAIR COMPLIANCE DASHBOARD</h2>
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mt-1">Foundational Asset Integrity Monitor</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4">
        <div className="bg-card border border-border rounded-sm p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-1.5 h-4 bg-primary rounded-sm" />
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider">License Verification</h3>
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex justify-between items-center bg-muted/40 p-2 rounded-sm border border-border/50">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">Status</span>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-sm ${licenseStatus === 'Present' ? 'bg-status-ok/20 text-status-ok border border-status-ok' : 'bg-status-critical/20 text-status-critical border border-status-critical'}`}>
                {licenseStatus.toUpperCase()}
              </span>
            </div>
            <button className="text-[10px] font-mono py-1.5 px-3 rounded-sm border border-primary text-primary hover:bg-primary hover:text-primary-foreground transition-all">
              ADD LICENSE
            </button>
          </div>
        </div>

        <div className="bg-card border border-border rounded-sm p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-1.5 h-4 bg-accent rounded-sm" />
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider">Metadata Validation</h3>
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center bg-muted/40 p-2 rounded-sm border border-border/50">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">CITATION.CFF</span>
              <span className={`text-[10px] font-mono font-bold ${citationStatus === 'Present' ? 'text-status-ok' : 'text-status-critical'}`}>
                {citationStatus.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between items-center bg-muted/40 p-2 rounded-sm border border-border/50">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">CODEMETA.JSON</span>
              <span className={`text-[10px] font-mono font-bold ${codemetaStatus === 'Present' ? 'text-status-ok' : 'text-status-critical'}`}>
                {codemetaStatus.toUpperCase()}
              </span>
            </div>
            <button className="text-[10px] font-mono mt-1 py-1.5 px-3 rounded-sm border border-border text-muted-foreground hover:text-foreground hover:border-foreground transition-all">
              UPDATE METADATA
            </button>
          </div>
        </div>

        <div className="bg-card border border-border rounded-sm p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-1.5 h-4 bg-chart-5 rounded-sm" />
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider">Software Release</h3>
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex justify-between items-center bg-muted/40 p-2 rounded-sm border border-border/50">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">Release Track</span>
              <span className="text-[10px] font-mono font-bold text-foreground">
                {releaseStatus.toUpperCase()}
              </span>
            </div>
            <div className="bg-secondary/20 border border-border rounded-sm px-2 py-1.5">
              <p className="text-[9px] font-mono text-muted-foreground leading-relaxed">
                Automated release synchronization with Zenodo DOI assignment is active for production deployments.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-border mt-4">
        <p className="text-[8px] font-mono text-muted-foreground uppercase tracking-widest text-center">
          Omni-Sentinel FAIR Engine · Build 2026.06.23.01 · Integrity Verified
        </p>
      </div>
    </div>
  );
};

export default FAIRComplianceDashboard;
