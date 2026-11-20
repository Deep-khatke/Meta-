import React from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Compass,
  Maximize2,
  ExternalLink,
  Eye,
  Camera
} from 'lucide-react';
import { Project, SimulationState, SceneObject } from '../../../types';
import { Viewport3D } from '../../Viewport3D';

interface SimulationTabProps {
  project: Project;
  simState: SimulationState;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
  onChangeSpeed: (multiplier: number) => void;
  onOpenInEngineeringStudio: () => void;
  targetPosition?: [number, number, number] | null;
  onSelectTarget?: (pos: [number, number, number]) => void;
  onUpdateObject?: (obj: SceneObject) => void;
}

export const SimulationTab: React.FC<SimulationTabProps> = ({
  project,
  simState,
  onPlay,
  onPause,
  onStep,
  onReset,
  onChangeSpeed,
  onOpenInEngineeringStudio,
  targetPosition,
  onSelectTarget,
  onUpdateObject,
}) => {
  const isRunning = simState?.status === 'running';

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        backgroundColor: '#0D0E11',
        overflow: 'hidden',
      }}
    >
      {/* Simulation Top Bar Controls */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          background: '#16171B',
          borderBottom: '1px solid #26272D',
          zIndex: 10,
        }}
      >
        {/* Playback Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {isRunning ? (
            <button
              onClick={onPause}
              title="Pause Simulation"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 10px',
                borderRadius: 6,
                background: 'rgba(245, 158, 11, 0.15)',
                border: '1px solid rgba(245, 158, 11, 0.4)',
                color: '#F59E0B',
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              <Pause size={12} />
              <span>Pause</span>
            </button>
          ) : (
            <button
              onClick={onPlay}
              title="Run Simulation"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 10px',
                borderRadius: 6,
                background: 'linear-gradient(135deg, #0284C7 0%, #0369A1 100%)',
                border: '1px solid rgba(56, 189, 248, 0.5)',
                color: '#FFFFFF',
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
                boxShadow: '0 2px 8px rgba(2, 132, 199, 0.35)',
              }}
            >
              <Play size={12} fill="#FFFFFF" />
              <span>Run</span>
            </button>
          )}

          <button
            onClick={onStep}
            title="Step by 1 physics frame"
            style={{
              padding: '4px 8px',
              borderRadius: 6,
              background: '#222329',
              border: '1px solid #33343C',
              color: '#D4D4D8',
              fontSize: 11,
              cursor: 'pointer',
            }}
          >
            <StepForward size={12} />
          </button>

          <button
            onClick={onReset}
            title="Reset Simulation to Initial Conditions"
            style={{
              padding: '4px 8px',
              borderRadius: 6,
              background: '#222329',
              border: '1px solid #33343C',
              color: '#D4D4D8',
              fontSize: 11,
              cursor: 'pointer',
            }}
          >
            <RotateCcw size={12} />
          </button>
        </div>

        {/* Action: Open in Engineering Studio */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button
            onClick={onOpenInEngineeringStudio}
            title="Switch to full Engineering Studio to edit joints, sensors & kinematics"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '4px 10px',
              borderRadius: 6,
              background: 'rgba(211, 84, 50, 0.15)',
              border: '1px solid rgba(211, 84, 50, 0.4)',
              color: '#D35432',
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            <span>Open in Engineering Studio</span>
            <ExternalLink size={11} />
          </button>
        </div>
      </div>

      {/* 3D Simulation Viewport */}
      <div style={{ flex: 1, position: 'relative' }}>
        <Viewport3D
          project={project}
          simState={simState}
          selectedObjectId={null}
          onSelectObject={() => {}}
          activeKeys={{}}
          targetPosition={targetPosition}
          onUpdateTarget={onSelectTarget}
          onUpdateObject={onUpdateObject}
        />
      </div>
    </div>
  );
};
