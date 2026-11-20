import React, { useState } from 'react';
import {
  CheckCircle2,
  Clock,
  AlertCircle,
  ChevronRight,
  ChevronDown,
  Loader2,
  HelpCircle,
  Wrench,
  Layers,
  Cpu,
  Radio,
  Sliders,
  Play,
  Activity
} from 'lucide-react';

export type ExecutionState = 'pending' | 'running' | 'completed' | 'failed' | 'awaiting_approval';

export interface AIExecutionStepItem {
  id: string;
  title: string;
  category?: 'robot' | 'environment' | 'sensor' | 'training' | 'evaluation' | 'general';
  state: ExecutionState;
  details?: string;
  output?: string;
  durationMs?: number;
}

interface AIExecutionStepProps {
  steps: AIExecutionStepItem[];
  overallTitle?: string;
}

export const AIExecutionStep: React.FC<AIExecutionStepProps> = ({
  steps,
  overallTitle = 'Agent Execution Plan',
}) => {
  const [expanded, setExpanded] = useState<boolean>(true);
  const [selectedStepId, setSelectedStepId] = useState<string | null>(null);

  if (!steps || steps.length === 0) return null;

  const completedCount = steps.filter((s) => s.state === 'completed').length;
  const isRunning = steps.some((s) => s.state === 'running');
  const hasFailed = steps.some((s) => s.state === 'failed');

  const getCategoryIcon = (cat?: string) => {
    switch (cat) {
      case 'robot': return <Wrench size={13} color="#D35432" />;
      case 'environment': return <Layers size={13} color="#38BDF8" />;
      case 'sensor': return <Radio size={13} color="#F59E0B" />;
      case 'training': return <Play size={13} color="#A855F7" />;
      case 'evaluation': return <Activity size={13} color="#4ADE80" />;
      default: return <Cpu size={13} color="#94A3B8" />;
    }
  };

  const getStateBadge = (state: ExecutionState) => {
    switch (state) {
      case 'completed':
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#4ADE80', fontSize: 11 }}>
            <CheckCircle2 size={13} />
            <span>Completed</span>
          </div>
        );
      case 'running':
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#38BDF8', fontSize: 11 }}>
            <Loader2 size={13} className="animate-spin" />
            <span>Running…</span>
          </div>
        );
      case 'failed':
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#EF4444', fontSize: 11 }}>
            <AlertCircle size={13} />
            <span>Failed</span>
          </div>
        );
      case 'awaiting_approval':
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#F59E0B', fontSize: 11 }}>
            <HelpCircle size={13} />
            <span>Needs Approval</span>
          </div>
        );
      case 'pending':
      default:
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#71717A', fontSize: 11 }}>
            <Clock size={13} />
            <span>Pending</span>
          </div>
        );
    }
  };

  return (
    <div
      style={{
        margin: '12px 0',
        borderRadius: 10,
        backgroundColor: '#16171B',
        border: '1px solid #2B2C32',
        overflow: 'hidden',
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.35)',
      }}
    >
      {/* Header */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '9px 14px',
          background: '#1C1D22',
          cursor: 'pointer',
          borderBottom: expanded ? '1px solid #2B2C32' : 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ color: '#D35432' }}>
            {isRunning ? <Loader2 size={14} className="animate-spin" /> : <Cpu size={14} />}
          </div>
          <span style={{ fontSize: 12, fontWeight: 700, color: '#F5F5F5' }}>
            {overallTitle}
          </span>
          <span
            style={{
              fontSize: 10.5,
              padding: '1px 7px',
              borderRadius: 100,
              background: isRunning
                ? 'rgba(56, 189, 248, 0.15)'
                : hasFailed
                ? 'rgba(239, 68, 68, 0.15)'
                : 'rgba(74, 222, 128, 0.15)',
              color: isRunning ? '#38BDF8' : hasFailed ? '#EF4444' : '#4ADE80',
              fontWeight: 600,
              fontFamily: 'var(--font-mono, monospace)',
            }}
          >
            {completedCount}/{steps.length}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#71717A' }}>
          <span style={{ fontSize: 11 }}>{expanded ? 'Hide Details' : 'View Steps'}</span>
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </div>
      </div>

      {/* Steps List */}
      {expanded && (
        <div style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {steps.map((step, idx) => {
            const isSelected = selectedStepId === step.id;
            return (
              <div
                key={step.id || idx}
                style={{
                  borderRadius: 6,
                  backgroundColor: isSelected ? '#222329' : '#1A1B20',
                  border: isSelected ? '1px solid #3F4048' : '1px solid #26272D',
                  overflow: 'hidden',
                  transition: 'background-color 0.15s ease',
                }}
              >
                <div
                  onClick={() => setSelectedStepId(isSelected ? null : step.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '7px 10px',
                    cursor: (step.details || step.output) ? 'pointer' : 'default',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                    <span style={{ opacity: 0.8 }}>{getCategoryIcon(step.category)}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: '#E4E4E7' }}>
                      {step.title}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    {getStateBadge(step.state)}
                    {(step.details || step.output) && (
                      <span style={{ color: '#71717A' }}>
                        {isSelected ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                      </span>
                    )}
                  </div>
                </div>

                {/* Expanded Details / Output */}
                {isSelected && (step.details || step.output) && (
                  <div
                    style={{
                      padding: '8px 12px 10px 12px',
                      borderTop: '1px solid #2B2C32',
                      background: '#141518',
                      fontSize: 11.5,
                      color: '#A1A1AA',
                      lineHeight: 1.5,
                    }}
                  >
                    {step.details && <div style={{ marginBottom: step.output ? 6 : 0 }}>{step.details}</div>}
                    {step.output && (
                      <pre
                        style={{
                          margin: 0,
                          padding: '6px 10px',
                          borderRadius: 4,
                          background: '#0D0E11',
                          border: '1px solid #23242A',
                          color: '#38BDF8',
                          fontSize: 11,
                          fontFamily: 'var(--font-mono, monospace)',
                          overflowX: 'auto',
                        }}
                      >
                        {step.output}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

