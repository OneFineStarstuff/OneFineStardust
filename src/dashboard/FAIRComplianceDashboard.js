import React from 'react';
import { checkLicenseFile } from '../utils/licenseCheck';

const FAIRComplianceDashboard = () => {
  const licenseStatus = checkLicenseFile();
  return (
    <div className="fair-compliance-dashboard">
      <h2>FAIR Compliance Dashboard</h2>
      <div className="compliance-section">
        <h3>License Verification</h3>
        <button>Add License</button>
        <p>Status: <span className="status-badge">{licenseStatus}</span></p>
      </div>
      <div className="compliance-section">
        <h3>Metadata Validation</h3>
        <p>Status: <span className="status-badge">Unknown</span></p>
        <button>Update Metadata</button>
      </div>
      <div className="compliance-section">
        <h3>FAIR Software Release Process</h3>
        <p>Status: <span className="status-badge">Unknown</span></p>
      </div>
    </div>
  );
};

export default FAIRComplianceDashboard;
