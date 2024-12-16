import fs from 'fs';
import path from 'path';

export const checkLicenseFile = () => {
  const licensePath = path.resolve(process.cwd(), 'LICENSE');
  try {
    if (fs.existsSync(licensePath)) {
      return 'Present';
    }
  } catch (err) {
    console.error(err);
  }
  return 'Missing';
};