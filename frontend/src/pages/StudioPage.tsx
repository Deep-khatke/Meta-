import React from 'react';
import { App } from '../App';

// Wraps the full engineering studio in a full-viewport overflow:hidden container
// so the existing studio layout is not affected by the landing page scroll CSS.
export function StudioPage() {
  return (
    <div className="studio-root" style={{ width: '100%', height: '100vh', overflow: 'hidden' }}>
      <App />
    </div>
  );
}
