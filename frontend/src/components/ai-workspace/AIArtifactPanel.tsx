import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  X,
  Maximize2,
  Minimize2,
  Box,
  Wrench,
  TrendingUp,
  Activity,
  FileCode,
  ExternalLink
} from 'lucide-react';
import { Project, SimulationState, SceneObject } from '../../types';
import { SimulationTab } from './tabs/SimulationTab';
import { RobotTab } from './tabs/RobotTab';
import { TrainingTab } from './tabs/TrainingTab';
import { ResultsTab } from './tabs/ResultsTab';
import { FilesTab } from './tabs/FilesTab';

export type ArtifactTabType = 'simulation' | 'robot' | 'training' | 'results' | 'files';

interface AIArtifactPanelProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: ArtifactTabType;
  onSelectTab: (tab: ArtifactTabType) => void;
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
  onShowToast?: (msg: string) => void;
}

export const AIArtifactPanel: React.FC<AIArtifactPanelProps> = ({
  isOpen,
  onClose,
  activeTab,
  onSelectTab,
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
  onShowToast,
}) => {
  const [panelWidth, setPanelWidth] = useState<number>(() => {
    const saved = localStorage.getItem('roboforge_ai_artifact_width');
    return saved ? parseInt(saved, 10) : 520;
  });
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  // Dragging left resize handle to expand/shrink the 3D viewport
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    if (isExpanded) {
      setIsExpanded(false);
    }
  }, [isExpanded]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      // Calculate new width relative to right edge of viewport
      const newWidth = window.innerWidth - e.clientX;
      const minWidth = 360;
      const maxWidth = Math.max(minWidth, window.innerWidth - 280);
      const clamped = Math.max(minWidth, Math.min(newWidth, maxWidth));
      setPanelWidth(clamped);
      localStorage.setItem('roboforge_ai_artifact_width', clamped.toString());
      window.dispatchEvent(new Event('resize'));
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      window.dispatchEvent(new Event('resize'));
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging]);

  const handleToggleExpand = useCallback(() => {
    setIsExpanded((prev) => {
      const next = !prev;
      setTimeout(() => window.dispatchEvent(new Event('resize')), 30);
      setTimeout(() => window.dispatchEvent(new Event('resize')), 150);
      setTimeout(() => window.dispatchEvent(new Event('resize')), 250);
      return next;
    });
  }, []);

  if (!isOpen) return null;

  const tabs: Array<{ id: ArtifactTabType; label: string; icon: React.ReactNode }> = [
    { id: 'simulation', label: 'Simulation', icon: <Box size={13} /> },
    { id: 'robot', label: 'Robot', icon: <Wrench size={13} /> },
    { id: 'training', label: 'Training', icon: <TrendingUp size={13} /> },
    { id: 'results', label: 'Results', icon: <Activity size={13} /> },
    { id: 'files', label: 'Files', icon: <FileCode size={13} /> },
  ];

  const currentWidth = isExpanded ? 'calc(100% - 260px)' : `${panelWidth}px`;

  return (
    <aside
      className="rf-ai-artifact-panel"
      style={{
        width: currentWidth,
        maxWidth: isExpanded ? '100%' : 'calc(100% - 240px)',
        minWidth: 360,
        height: '100%',
        backgroundColor: '#121316',
        borderLeft: isDragging ? '2px solid #38BDF8' : '1px solid #2B2C32',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        zIndex: 30,
        boxShadow: isDragging ? '-6px 0 28px rgba(56, 189, 248, 0.3)' : '-4px 0 24px rgba(0, 0, 0, 0.45)',
        transition: isDragging ? 'none' : isExpanded ? 'width 0.2s cubic-bezier(0.16, 1, 0.3, 1)' : 'none',
      }}
    >
      {/* ── Draggable Left Splitter Handle ── */}
      <div
        onMouseDown={handleMouseDown}
        style={{
          position: 'absolute',
          top: 0,
          left: -6,
          width: 12,
          height: '100%',
          cursor: 'col-resize',
          zIndex: 50,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          userSelect: 'none',
        }}
        title="Drag left/right to resize 3D simulation panel"
      >
        <div
          style={{
            width: isDragging ? 3 : 2,
            height: isDragging ? '100%' : 48,
            borderRadius: 2,
            backgroundColor: isDragging ? '#38BDF8' : 'rgba(255, 255, 255, 0.2)',
            boxShadow: isDragging ? '0 0 10px rgba(56, 189, 248, 0.9)' : 'none',
            transition: 'background-color 0.15s ease, height 0.15s ease',
          }}
        />
      </div>

      {/* ── Top Header with Tab Switcher & Window Controls ── */}
      <div
        style={{
          height: 44,
          minHeight: 44,
          padding: '0 12px',
          background: '#16171B',
          borderBottom: '1px solid #26272D',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          userSelect: 'none',
        }}
      >
        {/* Tab Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onSelectTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '5px 10px',
                  borderRadius: 6,
                  fontSize: 11.5,
                  fontWeight: isActive ? 700 : 500,
                  color: isActive ? '#FFFFFF' : '#A1A1AA',
                  background: isActive ? '#24252B' : 'transparent',
                  border: isActive ? '1px solid #383A42' : '1px solid transparent',
                  cursor: 'pointer',
                  transition: 'all 0.1s ease',
                }}
              >
                <span style={{ color: isActive ? '#D35432' : '#71717A' }}>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Window Actions: Expand/Contract & Close */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button
            onClick={handleToggleExpand}
            title={isExpanded ? 'Restore panel size' : 'Expand panel'}
            style={{
              width: 26,
              height: 26,
              borderRadius: 6,
              background: 'transparent',
              border: 'none',
              color: '#A1A1AA',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
            }}
          >
            {isExpanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
          </button>

          <button
            onClick={onClose}
            title="Close artifact panel"
            style={{
              width: 26,
              height: 26,
              borderRadius: 6,
              background: 'transparent',
              border: 'none',
              color: '#A1A1AA',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
            }}
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* ── Active Tab Viewport ── */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {activeTab === 'simulation' && (
          <SimulationTab
            project={project}
            simState={simState}
            onPlay={onPlay}
            onPause={onPause}
            onStep={onStep}
            onReset={onReset}
            onChangeSpeed={onChangeSpeed}
            onOpenInEngineeringStudio={onOpenInEngineeringStudio}
            targetPosition={targetPosition}
            onSelectTarget={onSelectTarget}
            onUpdateObject={onUpdateObject}
          />
        )}

        {activeTab === 'robot' && (
          <RobotTab
            project={project}
            simState={simState}
            onOpenInEngineeringStudio={onOpenInEngineeringStudio}
          />
        )}

        {activeTab === 'training' && (
          <TrainingTab
            simState={simState}
            onDeployPolicy={onOpenInEngineeringStudio}
            onShowToast={onShowToast}
          />
        )}

        {activeTab === 'results' && (
          <ResultsTab
            project={project}
            simState={simState}
            onShowToast={onShowToast}
          />
        )}

        {activeTab === 'files' && (
          <FilesTab
            project={project}
            onShowToast={onShowToast}
          />
        )}
      </div>
    </aside>
  );
};

