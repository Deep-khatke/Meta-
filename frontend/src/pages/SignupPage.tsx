import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

const C = {
  ivory: '#F1EEE7',
  graphite: '#17191A',
  black: '#202222',
  red: '#C4472D',
  muted: '#9A9A96',
};

type FormState = 'idle' | 'submitting' | 'error' | 'success';

export function SignupPage() {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [state, setState] = useState<FormState>('idle');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !email || !password) { setState('error'); return; }
    setState('submitting');
    await new Promise(r => setTimeout(r, 1200));
    setState('success');
    setTimeout(() => navigate('/studio'), 700);
  };

  return (
    <div style={{
      minHeight: '100vh', background: C.ivory, fontFamily: 'Manrope, sans-serif',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '60px 20px',
    }}>
      {/* Back link */}
      <div style={{ position: 'fixed', top: '28px', left: '40px' }}>
        <Link to="/" style={{ textDecoration: 'none' }}>
          <span style={{
            fontFamily: 'IBM Plex Mono, monospace', fontSize: '12px',
            letterSpacing: '0.2em', color: 'rgba(32,34,34,0.4)', fontWeight: 500,
          }}>
            ROBOFORGE
          </span>
        </Link>
      </div>

      <div style={{ width: '100%', maxWidth: '400px' }}>
        {/* Header */}
        <div style={{ marginBottom: '48px', textAlign: 'center' }}>
          <div style={{
            fontFamily: 'IBM Plex Mono, monospace', fontSize: '10px',
            letterSpacing: '0.2em', color: C.red, marginBottom: '16px',
          }}>
            CREATE ACCOUNT
          </div>
          <h1 style={{
            fontSize: '36px', fontWeight: 800, letterSpacing: '-0.03em',
            margin: '0 0 10px', color: C.black,
          }}>
            Start building.
          </h1>
          <p style={{ fontSize: '14px', color: C.black, opacity: 0.5, margin: 0 }}>
            Create your RoboForge account to save projects and run experiments.
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {/* Name */}
          <div>
            <label style={{
              display: 'block', fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '10px', letterSpacing: '0.15em', color: C.muted, marginBottom: '8px',
            }}>
              FULL NAME
            </label>
            <input
              type="text" value={name} onChange={e => { setName(e.target.value); setState('idle'); }}
              placeholder="Your name"
              style={{
                width: '100%', padding: '12px 14px', boxSizing: 'border-box',
                border: `1px solid ${state === 'error' && !name ? C.red : 'rgba(32,34,34,0.15)'}`,
                borderRadius: '3px', background: 'white',
                fontFamily: 'Manrope, sans-serif', fontSize: '14px', color: C.black, outline: 'none',
              }}
            />
          </div>

          {/* Email */}
          <div>
            <label style={{
              display: 'block', fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '10px', letterSpacing: '0.15em', color: C.muted, marginBottom: '8px',
            }}>
              EMAIL
            </label>
            <input
              type="email" value={email} onChange={e => { setEmail(e.target.value); setState('idle'); }}
              placeholder="you@example.com"
              style={{
                width: '100%', padding: '12px 14px', boxSizing: 'border-box',
                border: `1px solid ${state === 'error' && !email ? C.red : 'rgba(32,34,34,0.15)'}`,
                borderRadius: '3px', background: 'white',
                fontFamily: 'Manrope, sans-serif', fontSize: '14px', color: C.black, outline: 'none',
              }}
            />
          </div>

          {/* Password */}
          <div>
            <label style={{
              display: 'block', fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '10px', letterSpacing: '0.15em', color: C.muted, marginBottom: '8px',
            }}>
              PASSWORD
            </label>
            <div style={{ position: 'relative' }}>
              <input
                type={showPw ? 'text' : 'password'} value={password}
                onChange={e => { setPassword(e.target.value); setState('idle'); }}
                placeholder="Minimum 8 characters"
                style={{
                  width: '100%', padding: '12px 44px 12px 14px', boxSizing: 'border-box',
                  border: `1px solid ${state === 'error' && !password ? C.red : 'rgba(32,34,34,0.15)'}`,
                  borderRadius: '3px', background: 'white',
                  fontFamily: 'Manrope, sans-serif', fontSize: '14px', color: C.black, outline: 'none',
                }}
              />
              <button
                type="button" onClick={() => setShowPw(s => !s)}
                style={{
                  position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)',
                  background: 'none', border: 'none', cursor: 'pointer',
                  fontFamily: 'IBM Plex Mono, monospace', fontSize: '9px', letterSpacing: '0.12em', color: C.muted,
                }}
              >
                {showPw ? 'HIDE' : 'SHOW'}
              </button>
            </div>
          </div>

          {state === 'error' && (
            <div style={{
              background: 'rgba(196,71,45,0.08)', border: '1px solid rgba(196,71,45,0.25)',
              borderRadius: '3px', padding: '10px 14px',
              fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px', color: C.red, letterSpacing: '0.06em',
            }}>
              Please complete all fields.
            </div>
          )}

          <button
            type="submit"
            disabled={state === 'submitting' || state === 'success'}
            style={{
              background: state === 'success' ? '#2a7a42' : C.black,
              color: C.ivory, padding: '14px', borderRadius: '3px',
              border: 'none', cursor: state === 'submitting' ? 'wait' : 'pointer',
              fontFamily: 'Manrope, sans-serif', fontWeight: 700, fontSize: '14px',
              letterSpacing: '0.04em', transition: 'background 0.3s',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
            }}
          >
            {state === 'submitting' && (
              <span style={{
                width: '14px', height: '14px', border: '2px solid rgba(241,238,231,0.3)',
                borderTopColor: C.ivory, borderRadius: '50%',
                animation: 'spin 0.7s linear infinite', display: 'inline-block',
              }} />
            )}
            {state === 'success' ? 'Account created — opening studio' : state === 'submitting' ? 'Creating account…' : 'Create account'}
          </button>
        </form>

        <div style={{ marginTop: '28px', textAlign: 'center', fontSize: '13px', color: C.black, opacity: 0.5 }}>
          Already have an account?{' '}
          <Link to="/login" style={{ color: C.red, textDecoration: 'none', fontWeight: 600 }}>
            Sign in
          </Link>
        </div>

        <div style={{ marginTop: '20px', textAlign: 'center', fontSize: '11px', color: C.black, opacity: 0.35 }}>
          By creating an account you agree to the MIT-licensed terms of this platform.
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        input:focus { border-color: rgba(32,34,34,0.45) !important; box-shadow: 0 0 0 3px rgba(196,71,45,0.08); }
      `}</style>
    </div>
  );
}

