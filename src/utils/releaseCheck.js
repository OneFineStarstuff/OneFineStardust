import fs from 'fs';
import path from 'path';

export const checkReleaseProcess = () => {
  const zenodoPath = path.resolve(process.cwd(), '.zenodo.json');
  try {
    if (fs.existsSync(zenodoPath)) {
      const data = JSON.parse(fs.readFileSync(zenodoPath, 'utf8'));
      if (data.metadata && data.metadata.version) {
        return `Active (v${data.metadata.version})`;
      }
    }
  } catch (err) {
    console.error('Error checking release process:', err);
  }
  return 'Not Configured';
};
