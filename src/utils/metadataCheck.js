import fs from 'fs';
import path from 'path';

export const checkMetadataFiles = () => {
  const citationPath = path.resolve(process.cwd(), 'CITATION.cff');
  const codemetaPath = path.resolve(process.cwd(), 'codemeta.json');
  let citationStatus = 'Missing';
  let codemetaStatus = 'Missing';

  try {
    if (fs.existsSync(citationPath)) {
      citationStatus = 'Present';
    }
    if (fs.existsSync(codemetaPath)) {
      codemetaStatus = 'Present';
    }
  } catch (err) {
    console.error(err);
  }
  return { citationStatus, codemetaStatus };
};