import React, { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { LandingPage } from './pages/LandingPage';
import { LoginPage } from './pages/LoginPage';
import { SignupPage } from './pages/SignupPage';
import { BillingPage } from './pages/BillingPage';

// Lazy-load the heavy studio so the landing page is fast
const StudioPage = lazy(() => import('./pages/StudioPage').then(m => ({ default: m.StudioPage })));
const RobotPreview = lazy(() => import('./pages/RobotPreview').then(m => ({ default: m.RobotPreview })));

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/aurix" element={<Navigate to="/" replace />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/billing" element={<BillingPage />} />
      <Route path="/billing/return" element={<BillingPage />} />
      <Route path="/billing/cancel" element={<BillingPage />} />
      <Route
        path="/studio"
        element={
          <Suspense fallback={<StudioLoader />}>
            <StudioPage />
          </Suspense>
        }
      />
      <Route path="/robots" element={<Suspense fallback={<div />}><RobotPreview /></Suspense>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function StudioLoader() {
  return (
    <div style={{
      background: '#17191A',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '20px',
      fontFamily: 'IBM Plex Mono, monospace',
      color: '#F1EEE7'
    }}>
      <div style={{ fontSize: '13px', letterSpacing: '0.15em', opacity: 0.5 }}>ROBOFORGE</div>
      <div style={{ display: 'flex', gap: '6px' }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: '6px', height: '6px', borderRadius: '50%',
            background: '#C4472D',
            animation: `dot-pulse 1.2s ease-in-out ${i * 0.2}s infinite`
          }} />
        ))}
      </div>
      <div style={{ fontSize: '11px', letterSpacing: '0.1em', opacity: 0.35 }}>Initialising physics engine</div>
      <style>{`@keyframes dot-pulse { 0%,80%,100%{opacity:.2;transform:scale(1)} 40%{opacity:1;transform:scale(1.3)} }`}</style>
    </div>
  );
}

