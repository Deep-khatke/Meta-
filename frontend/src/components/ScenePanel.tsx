import React from 'react';
import {
  Layers, Box, Circle, Cylinder, Eye, EyeOff, Trash2,
  Copy, Bot, Upload, GitFork, Navigation, ArrowUp, ArrowDown
} from 'lucide-react';
import { Project, SceneObject, SimulationState } from '../types';

interface ScenePanelProps {
  project: Project;
  simState: SimulationState;
  selectedObjectId: string | null;
  onSelectObject: (id: string | null) => void;
  onAddObstacle: (type: 'box' | 'sphere' | 'cylinder') => void;
  onToggleVisibility: (id: string) => void;
  onDeleteObject: (id: string) => void;
  onDuplicateObject?: (id: string) => void;
  onOpenURDFModal?: () => void;
  onClearGenericRobot?: () => void;
  onRemoveRobot?: () => void;
  onRestoreRobot?: () => void;
}

export const ScenePanel: React.FC<ScenePanelProps> = ({
  project,
  simState,
  selectedObjectId,
  onSelectObject,
  onAddObstacle,
  onToggleVisibility,
  onDeleteObject,
  onDuplicateObject,
  onOpenURDFModal,
  onClearGenericRobot,
  onRemoveRobot,
  onRestoreRobot,
}) => {
  const handleDroneFlightCommand = async (droneId: string, cmd: 'takeoff' | 'land') => {
    try {
      await fetch('/api/drone/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          drone_id: droneId,
          command: cmd,
          altitude: 1.5,
        }),
      });
    } catch {}
  };

  const getIconForType = (type: string, name?: string) => {
    if (name && (name.toLowerCase().includes('drone') || name.toLowerCase().includes('uav') || name.toLowerCase().includes('quad'))) {
      return <Navigation size={14} color="#06b6d4" />;
    }
    switch (type) {
      case 'box':
        return <Box size={14} color="#e06c75" />;
      case 'cylinder':
        return <Cylinder size={14} color="#98c379" />;
      case 'sphere':
        return <Circle size={14} color="#d19a66" />;
      default:
        return <Box size={14} />;
    }
  };

  return (
    <aside className="rf-left-sidebar">
      {/* Panel Header */}
      <div className="rf-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Layers size={13} color="#C4472D" />
          <span>Scene Hierarchy</span>
        </div>
        {onOpenURDFModal && (
          <button
            className="rf-btn"
            style={{ fontSize: 10, padding: '2px 8px', gap: 4, borderRadius: 100 }}
            onClick={onOpenURDFModal}
            title="Import Robot from URDF XML"
          >
            <Upload size={11} color="#C4472D" />
            <span>URDF</span>
          </button>
        )}
      </div>

      {/* Add Object Action Strip */}
      <div
        style={{
          padding: '10px 12px',
          borderBottom: '1px solid var(--border-subtle)',
          backgroundColor: 'rgba(255, 255, 255, 0.02)',
          display: 'flex',
          flexDirection: 'column',
          gap: 7,
        }}
      >
        <span style={{ fontSize: 9.5, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          Primitive Obstacles
        </span>
        <div className="rf-segmented" style={{ width: '100%', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 3, padding: 3 }}>
          <button
            className="rf-segmented-item"
            style={{ fontSize: 10.5, padding: '4px 6px', justifyContent: 'center' }}
            onClick={() => onAddObstacle('box')}
            title="Add Box Obstacle"
          >
            <Box size={12} color="#C4472D" />
            <span>Box</span>
          </button>
          <button
            className="rf-segmented-item"
            style={{ fontSize: 10.5, padding: '4px 6px', justifyContent: 'center' }}
            onClick={() => onAddObstacle('sphere')}
            title="Add Sphere Obstacle"
          >
            <Circle size={12} color="#FBBF24" />
            <span>Sphere</span>
          </button>
          <button
            className="rf-segmented-item"
            style={{ fontSize: 10.5, padding: '4px 6px', justifyContent: 'center' }}
            onClick={() => onAddObstacle('cylinder')}
            title="Add Cylinder Obstacle"
          >
            <Cylinder size={12} color="#34D399" />
            <span>Cylinder</span>
          </button>
        </div>
      </div>

      {/* Hierarchy Tree */}
      <div className="rf-panel-body" style={{ padding: '6px 8px' }}>
        {/* Mobile Robot Root Item (shown when a robot exists and is active in the workspace) */}
        {project.robot && project.robot_active !== false && (
          <>
            <div
              onClick={() => onSelectObject('robot')}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 8px',
                borderRadius: 4,
                cursor: 'pointer',
                backgroundColor: selectedObjectId === 'robot' ? 'var(--bg-panel-hover)' : 'transparent',
                border: selectedObjectId === 'robot' ? '1px solid var(--accent-blue)' : '1px solid transparent',
                marginBottom: 4,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Bot size={15} color="var(--accent-blue)" />
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{project.robot.name}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Diff-Drive</span>
                {onRemoveRobot && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm('Remove the active robot from the workspace? The scene will have nothing active.')) {
                        onRemoveRobot();
                      }
                    }}
                    title="Remove robot from workspace"
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, display: 'flex' }}
                  >
                    <Trash2 size={13} color="var(--accent-red, #e06c75)" />
                  </button>
                )}
              </div>
            </div>

            {/* Robot Parts Sub-items */}
            <div style={{ paddingLeft: 22, marginBottom: 12, borderLeft: '1px solid var(--border-subtle)', marginLeft: 14 }}>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', padding: '2px 0' }}>• Chassis Box</div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', padding: '2px 0' }}>• Left & Right Drive Wheels</div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', padding: '2px 0' }}>• 4 Caster Stabilizers</div>
              <div style={{ fontSize: 11, color: project.sensors?.lidar?.enabled !== false ? 'var(--accent-blue)' : 'var(--text-muted)', padding: '2px 0' }}>
                • LiDAR ({project.sensors?.lidar?.enabled !== false ? 'ON' : 'OFF'})
              </div>
              <div style={{ fontSize: 11, color: project.sensors?.imu?.enabled !== false ? 'var(--accent-green)' : 'var(--text-muted)', padding: '2px 0' }}>
                • 6-DOF IMU ({project.sensors?.imu?.enabled !== false ? 'ON' : 'OFF'})
              </div>
              <div style={{ fontSize: 11, color: project.sensors?.camera?.enabled !== false ? '#38bdf8' : 'var(--text-muted)', padding: '2px 0' }}>
                • Vision Camera ({project.sensors?.camera?.enabled !== false ? 'ON' : 'OFF'})
              </div>
            </div>
          </>
        )}

        {/* Inactive Robot Notice: robot exists but was removed from the workspace */}
        {project.robot && project.robot_active === false && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px',
              borderRadius: 4,
              marginBottom: 12,
              border: '1px dashed var(--border-subtle)',
              color: 'var(--text-muted)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Bot size={15} color="var(--text-muted)" />
              <span style={{ fontSize: 12 }}>No active robot</span>
            </div>
            {onRestoreRobot && (
              <button
                onClick={() => onRestoreRobot()}
                title="Add the built-in differential-drive robot"
                style={{
                  background: 'var(--bg-panel-hover)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 4,
                  cursor: 'pointer',
                  fontSize: 11,
                  color: 'var(--accent-blue)',
                  padding: '3px 8px',
                }}
              >
                + Add Robot
              </button>
            )}
          </div>
        )}

        {/* Empty Scene Robot Notice: no mobile robot and no generic/URDF robot at all */}
        {!project.robot && !project.generic_robot && (
          <div style={{
            padding: '20px 14px',
            margin: '8px 0 14px 0',
            borderRadius: 12,
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px dashed rgba(255, 255, 255, 0.1)',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 8,
          }}>
            <div style={{
              width: 34,
              height: 34,
              borderRadius: '50%',
              background: 'rgba(196, 71, 45, 0.12)',
              border: '1px solid rgba(196, 71, 45, 0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#C4472D'
            }}>
              <Bot size={16} />
            </div>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#F1EEE7' }}>
              No Active Robot
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Import a URDF or spawn a mobile rover
            </div>
            {onRestoreRobot && (
              <button
                onClick={() => onRestoreRobot()}
                className="rf-btn"
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  padding: '4px 12px',
                  marginTop: 4,
                  borderRadius: 100,
                  color: '#FFFFFF',
                  backgroundColor: '#C4472D',
                  borderColor: '#C4472D',
                  boxShadow: '0 2px 10px rgba(196, 71, 45, 0.4)'
                }}
              >
                + Spawn AMR Rover
              </button>
            )}
          </div>
        )}

        {/* Generic Kinematic Robot (if present) */}
        {project.generic_robot && (
          <div style={{ marginBottom: 12 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 8px',
                borderRadius: 4,
                cursor: 'pointer',
                backgroundColor: selectedObjectId === 'generic_robot' ? 'var(--bg-panel-hover)' : 'transparent',
                border: selectedObjectId === 'generic_robot' ? '1px solid var(--accent-green)' : '1px solid transparent',
                marginBottom: 4,
              }}
            >
              <div
                onClick={() => onSelectObject('generic_robot')}
                style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}
              >
                <GitFork size={14} color="var(--accent-green)" />
                <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 12 }}>
                  {project.generic_robot.name}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 10, color: 'var(--accent-green)' }}>
                  {project.generic_robot.joints.length} J
                </span>
                {onClearGenericRobot && (
                  <button
                    className="rf-btn-icon"
                    onClick={(e) => {
                      e.stopPropagation();
                      onClearGenericRobot();
                    }}
                    title="Remove URDF Robot"
                    style={{ padding: 2, background: 'transparent', border: 'none', cursor: 'pointer' }}
                  >
                    <Trash2 size={12} color="#ef4444" />
                  </button>
                )}
              </div>
            </div>
            <div style={{ paddingLeft: 22, borderLeft: '1px solid rgba(34,197,94,0.3)', marginLeft: 14 }}>
              {project.generic_robot.joints.map((j) => (
                <div key={j.name} style={{ fontSize: 11, color: 'var(--text-secondary)', padding: '2px 0' }}>
                  • {j.name} ({j.type})
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Obstacles Header */}
        <div
          style={{
            fontSize: 10,
            fontWeight: 600,
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            margin: '8px 4px 6px 4px',
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span>Obstacles & Props</span>
          <span>{project.objects.length}</span>
        </div>

        {/* Obstacles List */}
        {project.objects.length === 0 ? (
          <div style={{ padding: '14px 8px', color: 'var(--text-muted)', fontSize: 10.5, textAlign: 'center' }}>
            No obstacles in scene. Use the tools above to place geometry.
          </div>
        ) : (
          project.objects.map((obj: SceneObject) => {
            const isSelected = selectedObjectId === obj.id;
            const isVisible = obj.visible !== false;

            return (
              <div
                key={obj.id}
                onClick={() => onSelectObject(obj.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '5px 8px',
                  borderRadius: 4,
                  cursor: 'pointer',
                  backgroundColor: isSelected ? 'var(--bg-panel-hover)' : 'transparent',
                  border: isSelected ? '1px solid var(--accent-orange)' : '1px solid transparent',
                  marginBottom: 3,
                  opacity: isVisible ? 1.0 : 0.45,
                }}
              >
                {(() => {
                  const isDrone = obj.name.toLowerCase().includes('drone') || obj.id.startsWith('drone') || obj.name.toLowerCase().includes('uav') || obj.name.toLowerCase().includes('quad');
                  return (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
                        {getIconForType(obj.type, obj.name)}
                        <span
                          style={{
                            fontSize: 12,
                            color: isSelected ? 'var(--text-primary)' : 'var(--text-secondary)',
                            whiteSpace: 'nowrap',
                            textOverflow: 'ellipsis',
                            overflow: 'hidden',
                            maxWidth: 90,
                          }}
                        >
                          {obj.name}
                        </span>
                        {isDrone ? (
                          <span
                            style={{
                              fontSize: 9,
                              padding: '1px 4px',
                              borderRadius: 3,
                              backgroundColor: 'rgba(6, 182, 212, 0.2)',
                              color: '#22d3ee',
                              fontWeight: 700,
                              border: '1px solid rgba(6, 182, 212, 0.35)',
                            }}
                          >
                            DRONE
                          </span>
                        ) : (
                          <span
                            style={{
                              fontSize: 9,
                              padding: '1px 3px',
                              borderRadius: 3,
                              backgroundColor: obj.is_static ? 'rgba(100, 116, 139, 0.2)' : 'rgba(234, 88, 12, 0.2)',
                              color: obj.is_static ? '#94a3b8' : '#fb923c',
                              fontWeight: 600,
                            }}
                          >
                            {obj.is_static ? 'STAT' : 'DYN'}
                          </span>
                        )}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                        {isDrone && (
                          <>
                            <button
                              className="rf-btn rf-btn-icon"
                              style={{ width: 18, height: 18, padding: 2, background: 'rgba(6, 182, 212, 0.15)', border: '1px solid rgba(6, 182, 212, 0.3)' }}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDroneFlightCommand(obj.id, 'takeoff');
                              }}
                              title="Takeoff to 1.5m"
                            >
                              <ArrowUp size={11} color="#22d3ee" />
                            </button>
                            <button
                              className="rf-btn rf-btn-icon"
                              style={{ width: 18, height: 18, padding: 2, background: 'rgba(234, 179, 8, 0.15)', border: '1px solid rgba(234, 179, 8, 0.3)' }}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDroneFlightCommand(obj.id, 'land');
                              }}
                              title="Land Drone"
                            >
                              <ArrowDown size={11} color="#facc15" />
                            </button>
                          </>
                        )}
                        {onDuplicateObject && (
                          <button
                            className="rf-btn rf-btn-icon"
                            style={{ width: 18, height: 18, padding: 2, background: 'transparent', border: 'none' }}
                            onClick={(e) => {
                              e.stopPropagation();
                              onDuplicateObject(obj.id);
                            }}
                            title="Duplicate Object"
                          >
                            <Copy size={11} color="var(--text-secondary)" />
                          </button>
                        )}

                  <button
                    className="rf-btn rf-btn-icon"
                    style={{ width: 18, height: 18, padding: 2, background: 'transparent', border: 'none' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleVisibility(obj.id);
                    }}
                    title={isVisible ? 'Hide Object' : 'Show Object'}
                  >
                    {isVisible ? <Eye size={11} color="var(--text-secondary)" /> : <EyeOff size={11} color="var(--text-muted)" />}
                  </button>

                  <button
                    className="rf-btn rf-btn-icon rf-btn-danger"
                    style={{ width: 18, height: 18, padding: 2 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteObject(obj.id);
                    }}
                    title="Delete Object"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              </>
            );
          })()}
        </div>
      );
    })
        )}
      </div>

      {/* Scene Footer Stats */}
      <div
        style={{
          padding: '10px 14px',
          borderTop: '1px solid var(--border-subtle)',
          backgroundColor: 'rgba(255, 255, 255, 0.02)',
          fontSize: 10.5,
          fontFamily: 'var(--font-mono)',
          color: 'var(--text-muted)',
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>ACTIVE CONTACTS:</span>
          <span style={{ color: simState.contacts.length > 0 ? '#FB7185' : '#F1EEE7', fontWeight: 600 }}>
            {simState.contacts.length}
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>CONTAINMENT:</span>
          <span style={{ color: project.settings.enable_boundaries ? '#34D399' : 'var(--text-muted)', fontWeight: 600 }}>
            {project.settings.enable_boundaries ? 'BOUNDARIES ON' : 'FREE FLOOR'}
          </span>
        </div>
      </div>
    </aside>
  );
};
