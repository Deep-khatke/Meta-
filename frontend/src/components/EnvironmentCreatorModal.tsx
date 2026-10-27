import React, { useState } from 'react';
import {
  X,
  CheckCircle,
  AlertTriangle,
  Globe,
  Plus,
  Trash2,
  Box,
  Layers,
  Save,
  ArrowRight,
  Shield,
  Zap,
} from 'lucide-react';
import { SceneObject, ValidationResult } from '../types';
import { validateEnvironmentModel, createCustomAsset } from '../services/api';

interface EnvironmentCreatorModalProps {
  onClose: () => void;
  onApplyEnvironment: (envData: {
    gravity: [number, number, number];
    world_boundaries: [number, number, number, number, number, number];
    objects: SceneObject[];
  }) => void;
  onShowToast: (msg: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
}

export const EnvironmentCreatorModal: React.FC<EnvironmentCreatorModalProps> = ({
  onClose,
  onApplyEnvironment,
  onShowToast,
}) => {
  // World properties
  const [worldWidth, setWorldWidth] = useState<number>(10.0);
  const [worldDepth, setWorldDepth] = useState<number>(10.0);
  const [gravityPreset, setGravityPreset] = useState<'earth' | 'moon' | 'mars' | 'zero' | 'custom'>('earth');
  const [gravityZ, setGravityZ] = useState<number>(-9.81);
  const [groundFriction, setGroundFriction] = useState<number>(0.8);

  // Scene Objects state
  const [objects, setObjects] = useState<SceneObject[]>([
    {
      id: 'obs_wall_north',
      name: 'North Boundary Wall',
      type: 'box',
      size: [5.0, 0.1, 0.5],
      position: [0.0, 5.0, 0.5],
      rotation: [0, 0, 0],
      mass: 500.0,
      friction: [0.8, 0.005, 0.0001],
      is_static: true,
      color: '#3b82f6',
      visible: true,
    },
    {
      id: 'obs_wall_south',
      name: 'South Boundary Wall',
      type: 'box',
      size: [5.0, 0.1, 0.5],
      position: [0.0, -5.0, 0.5],
      rotation: [0, 0, 0],
      mass: 500.0,
      friction: [0.8, 0.005, 0.0001],
      is_static: true,
      color: '#3b82f6',
      visible: true,
    },
    {
      id: 'obs_crate_1',
      name: 'Central Crate',
      type: 'box',
      size: [0.3, 0.3, 0.3],
      position: [2.0, 1.5, 0.3],
      rotation: [0, 0, 0],
      mass: 15.0,
      friction: [0.6, 0.005, 0.0001],
      is_static: false,
      color: '#d97706',
      visible: true,
    },
    {
      id: 'obs_pylon_1',
      name: 'Slalom Pylon 1',
      type: 'cylinder',
      size: [0.15, 0.4],
      position: [1.0, 0.0, 0.4],
      rotation: [0, 0, 0],
      mass: 8.0,
      friction: [0.7, 0.005, 0.0001],
      is_static: false,
      color: '#ef4444',
      visible: true,
    },
  ]);

  // Validation state
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);

  // Save template state
  const [templateName, setTemplateName] = useState<string>('Custom Proving Ground');
  const [templateDesc, setTemplateDesc] = useState<string>('Parametric simulation arena with boundary walls and obstacles.');
  const [isSaving, setIsSaving] = useState<boolean>(false);

  const handleGravityPresetChange = (preset: 'earth' | 'moon' | 'mars' | 'zero' | 'custom') => {
    setGravityPreset(preset);
    if (preset === 'earth') setGravityZ(-9.81);
    else if (preset === 'moon') setGravityZ(-1.62);
    else if (preset === 'mars') setGravityZ(-3.72);
    else if (preset === 'zero') setGravityZ(0.0);
  };

  const handleAddObstacle = () => {
    const id = `obs_${Date.now().toString().slice(-4)}`;
    const newObs: SceneObject = {
      id,
      name: `Obstacle ${objects.length + 1}`,
      type: 'box',
      size: [0.25, 0.25, 0.25],
      position: [Number((Math.random() * 4 - 2).toFixed(2)), Number((Math.random() * 4 - 2).toFixed(2)), 0.25],
      rotation: [0, 0, 0],
      mass: 10.0,
      friction: [0.7, 0.005, 0.0001],
      is_static: false,
      color: '#10b981',
      visible: true,
    };
    setObjects([...objects, newObs]);
    setValidationResult(null);
  };

  const handleUpdateObstacle = (index: number, updates: Partial<SceneObject>) => {
    const updated = [...objects];
    updated[index] = { ...updated[index], ...updates };
    setObjects(updated);
    setValidationResult(null);
  };

  const handleRemoveObstacle = (index: number) => {
    setObjects(objects.filter((_, i) => i !== index));
    setValidationResult(null);
  };

  const handleGeneratePerimeterWalls = () => {
    const halfW = worldWidth / 2;
    const halfD = worldDepth / 2;
    const wallHeight = 0.6;
    const wallThick = 0.08;

    const newWalls: SceneObject[] = [
      {
        id: `wall_north_${Date.now()}`,
        name: 'North Boundary Wall',
        type: 'box',
        size: [halfW, wallThick, wallHeight / 2],
        position: [0, halfD, wallHeight / 2],
        rotation: [0, 0, 0],
        mass: 500,
        friction: [0.8, 0.005, 0.0001],
        is_static: true,
        color: '#2563eb',
        visible: true,
      },
      {
        id: `wall_south_${Date.now()}`,
        name: 'South Boundary Wall',
        type: 'box',
        size: [halfW, wallThick, wallHeight / 2],
        position: [0, -halfD, wallHeight / 2],
        rotation: [0, 0, 0],
        mass: 500,
        friction: [0.8, 0.005, 0.0001],
        is_static: true,
        color: '#2563eb',
        visible: true,
      },
      {
        id: `wall_east_${Date.now()}`,
        name: 'East Boundary Wall',
        type: 'box',
        size: [wallThick, halfD, wallHeight / 2],
        position: [halfW, 0, wallHeight / 2],
        rotation: [0, 0, 0],
        mass: 500,
        friction: [0.8, 0.005, 0.0001],
        is_static: true,
        color: '#2563eb',
        visible: true,
      },
      {
        id: `wall_west_${Date.now()}`,
        name: 'West Boundary Wall',
        type: 'box',
        size: [wallThick, halfD, wallHeight / 2],
        position: [-halfW, 0, wallHeight / 2],
        rotation: [0, 0, 0],
        mass: 500,
        friction: [0.8, 0.005, 0.0001],
        is_static: true,
        color: '#2563eb',
        visible: true,
      },
    ];

    const filtered = objects.filter((o) => !o.name.includes('Boundary Wall'));
    setObjects([...filtered, ...newWalls]);
    onShowToast(`Generated 4 perimeter boundary walls for ${worldWidth}x${worldDepth}m arena`, 'info');
  };

  const getEnvPayload = () => {
    const halfW = worldWidth / 2;
    const halfD = worldDepth / 2;
    return {
      settings: {
        gravity: [0.0, 0.0, gravityZ] as [number, number, number],
        world_boundaries: [-halfW, -halfD, 0.0, halfW, halfD, 4.0] as [number, number, number, number, number, number],
        enable_boundaries: true,
        timestep: 0.005,
        speed_multiplier: 1.0,
      },
      objects,
    };
  };

  const handleValidate = async () => {
    setIsValidating(true);
    try {
      const res = await validateEnvironmentModel(getEnvPayload());
      setValidationResult(res);
      if (res.valid) {
        onShowToast('MuJoCo environment physics validation passed', 'success');
      } else {
        onShowToast(`Validation failed: ${res.errors.join('; ')}`, 'error');
      }
    } catch (err: any) {
      setValidationResult({
        valid: false,
        errors: [err.message || 'Validation request failed'],
        warnings: [],
      });
      onShowToast(`Validation error: ${err.message}`, 'error');
    } finally {
      setIsValidating(false);
    }
  };

  const handleApply = async () => {
    if (!validationResult || !validationResult.valid) {
      try {
        const res = await validateEnvironmentModel(getEnvPayload());
        if (!res.valid) {
          setValidationResult(res);
          onShowToast(`Cannot apply invalid environment: ${res.errors[0]}`, 'error');
          return;
        }
      } catch (err: any) {
        onShowToast(`Validation error: ${err.message}`, 'error');
        return;
      }
    }

    const halfW = worldWidth / 2;
    const halfD = worldDepth / 2;
    onApplyEnvironment({
      gravity: [0.0, 0.0, gravityZ],
      world_boundaries: [-halfW, -halfD, 0.0, halfW, halfD, 4.0],
      objects,
    });
    onShowToast(`Environment applied to scene (${objects.length} obstacles)`, 'success');
    onClose();
  };

  const handleSaveAsAsset = async () => {
    if (!templateName.trim()) {
      onShowToast('Please provide a template name', 'warning');
      return;
    }
    setIsSaving(true);
    try {
      const halfW = worldWidth / 2;
      const halfD = worldDepth / 2;
      await createCustomAsset({
        category: 'template',
        name: templateName.trim(),
        description: templateDesc.trim(),
        tags: ['template', 'environment', 'arena'],
        data: {
          schema_version: '1.0.0',
          name: templateName.trim(),
          description: templateDesc.trim(),
          robot: {
            id: 'default_robot',
            name: 'ForgeBot-DD1',
            initial_position: [0.0, 0.0, 0.07],
            initial_yaw: 0.0,
            chassis_size: [0.20, 0.15, 0.05],
            wheel_radius: 0.06,
            wheel_width: 0.025,
            wheel_base: 0.28,
            caster_radius: 0.02,
            chassis_mass: 3.5,
            wheel_mass: 0.3,
            max_linear_velocity: 1.5,
            max_angular_velocity: 3.0,
          },
          objects,
          settings: {
            timestep: 0.005,
            speed_multiplier: 1.0,
            gravity: [0.0, 0.0, gravityZ],
            world_boundaries: [-halfW, -halfD, 0.0, halfW, halfD, 4.0],
            enable_boundaries: true,
          },
        },
      });
      onShowToast(`Saved "${templateName}" to Asset Library`, 'success');
    } catch (err: any) {
      onShowToast(`Failed to save template: ${err.message}`, 'error');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/25 backdrop-blur-[2px] flex items-center justify-center p-4">
      <div className="bg-[#12141a] border border-[#2a2e39] rounded-xl shadow-2xl w-full max-w-5xl h-[88vh] flex flex-col overflow-hidden text-gray-200">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#2a2e39] bg-[#161922]">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Globe size={20} />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-wide">Environment & World Builder</h2>
              <p className="text-xs text-gray-400">
                Configure arena bounds, physics constants, ground friction, and obstacle layout with real MuJoCo validation
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-[#202533] transition"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Validation Banner */}
          {validationResult && (
            <div
              className={`p-4 rounded-xl border flex items-start gap-3 transition-all ${
                validationResult.valid
                  ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200'
                  : 'bg-rose-950/30 border-rose-500/40 text-rose-200'
              }`}
            >
              {validationResult.valid ? (
                <CheckCircle size={20} className="text-emerald-400 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle size={20} className="text-rose-400 shrink-0 mt-0.5" />
              )}
              <div className="flex-1 text-xs space-y-1">
                <div className="font-semibold text-sm">
                  {validationResult.valid ? 'Environment Physics Validated' : 'Validation Errors Detected'}
                </div>
                {validationResult.errors.map((e, idx) => (
                  <div key={idx} className="text-rose-300 font-mono">
                    • {e}
                  </div>
                ))}
                {validationResult.warnings.map((w, idx) => (
                  <div key={idx} className="text-amber-300 font-mono">
                    ▲ Warning: {w}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Section 1: World Dimensions & Bounds */}
          <div className="bg-[#181c26] border border-[#262c3a] rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Box size={16} className="text-indigo-400" />
                World Dimensions & Boundary
              </h3>
              <button
                type="button"
                onClick={handleGeneratePerimeterWalls}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-600/30 transition flex items-center gap-1.5"
              >
                <Shield size={14} />
                Regenerate Perimeter Walls
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div>
                <label className="text-gray-400 block mb-1">Arena Width X (m)</label>
                <input
                  type="number"
                  step="0.5"
                  min="2"
                  max="100"
                  value={worldWidth}
                  onChange={(e) => {
                    setWorldWidth(Math.max(1, parseFloat(e.target.value) || 10));
                    setValidationResult(null);
                  }}
                  className="w-full bg-[#12141c] border border-[#2d3345] rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="text-gray-400 block mb-1">Arena Depth Y (m)</label>
                <input
                  type="number"
                  step="0.5"
                  min="2"
                  max="100"
                  value={worldDepth}
                  onChange={(e) => {
                    setWorldDepth(Math.max(1, parseFloat(e.target.value) || 10));
                    setValidationResult(null);
                  }}
                  className="w-full bg-[#12141c] border border-[#2d3345] rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="text-gray-400 block mb-1">Ground Friction</label>
                <input
                  type="number"
                  step="0.05"
                  min="0.01"
                  max="2.0"
                  value={groundFriction}
                  onChange={(e) => {
                    setGroundFriction(Math.max(0.01, parseFloat(e.target.value) || 0.8));
                    setValidationResult(null);
                  }}
                  className="w-full bg-[#12141c] border border-[#2d3345] rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="text-gray-400 block mb-1">Floor Area (Total)</label>
                <div className="w-full bg-[#12141c] border border-[#2d3345] rounded-lg px-3 py-2 text-indigo-400 font-mono font-medium">
                  {(worldWidth * worldDepth).toFixed(1)} m²
                </div>
              </div>
            </div>
          </div>

          {/* Section 2: Gravity & Physics Constants */}
          <div className="bg-[#181c26] border border-[#262c3a] rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Zap size={16} className="text-amber-400" />
              Gravity & Environmental Physics
            </h3>

            <div className="flex flex-wrap gap-2">
              {(['earth', 'moon', 'mars', 'zero', 'custom'] as const).map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => handleGravityPresetChange(preset)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition ${
                    gravityPreset === preset
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                      : 'bg-[#12141c] text-gray-400 border border-[#2b3040] hover:text-white'
                  }`}
                >
                  {preset === 'earth' && '🌍 Earth (-9.81 m/s²)'}
                  {preset === 'moon' && '🌑 Moon (-1.62 m/s²)'}
                  {preset === 'mars' && '🔴 Mars (-3.72 m/s²)'}
                  {preset === 'zero' && '✨ Zero-G (0.0 m/s²)'}
                  {preset === 'custom' && '⚙ Custom Gravity'}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs pt-2">
              <div>
                <label className="text-gray-400 block mb-1">Gravity X (m/s²)</label>
                <input
                  type="number"
                  disabled
                  value={0}
                  className="w-full bg-[#12141c] opacity-60 border border-[#2d3345] rounded-lg px-3 py-2 text-gray-400 font-mono"
                />
              </div>
              <div>
                <label className="text-gray-400 block mb-1">Gravity Y (m/s²)</label>
                <input
                  type="number"
                  disabled
                  value={0}
                  className="w-full bg-[#12141c] opacity-60 border border-[#2d3345] rounded-lg px-3 py-2 text-gray-400 font-mono"
                />
              </div>
              <div>
                <label className="text-gray-400 block mb-1">Gravity Z (m/s²)</label>
                <input
                  type="number"
                  step="0.1"
                  value={gravityZ}
                  onChange={(e) => {
                    setGravityZ(parseFloat(e.target.value) || 0);
                    setGravityPreset('custom');
                    setValidationResult(null);
                  }}
                  className="w-full bg-[#12141c] border border-[#2d3345] rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-amber-500"
                />
              </div>
            </div>
          </div>

          {/* Section 3: Obstacle Layout Designer */}
          <div className="bg-[#181c26] border border-[#262c3a] rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Layers size={16} className="text-cyan-400" />
                Obstacle & Collision Geometry Layout ({objects.length} objects)
              </h3>
              <button
                type="button"
                onClick={handleAddObstacle}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-600/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-600/30 transition flex items-center gap-1.5"
              >
                <Plus size={14} />
                Add Obstacle
              </button>
            </div>

            <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
              {objects.map((obs, idx) => (
                <div
                  key={obs.id || idx}
                  className="bg-[#12141c] border border-[#2a3040] rounded-xl p-3 text-xs space-y-2 hover:border-[#384055] transition"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 flex-1">
                      <span className="w-5 h-5 rounded-full bg-cyan-950 text-cyan-400 flex items-center justify-center font-bold text-[10px]">
                        {idx + 1}
                      </span>
                      <input
                        type="text"
                        value={obs.name}
                        onChange={(e) => handleUpdateObstacle(idx, { name: e.target.value })}
                        className="bg-transparent border-b border-transparent hover:border-gray-600 focus:border-cyan-500 text-white font-medium px-1 py-0.5 outline-none"
                      />
                    </div>

                    <div className="flex items-center gap-2">
                      <select
                        value={obs.type}
                        onChange={(e) =>
                          handleUpdateObstacle(idx, { type: e.target.value as 'box' | 'cylinder' | 'sphere' })
                        }
                        className="bg-[#181c26] border border-[#2d3345] rounded-md px-2 py-1 text-gray-300 text-xs focus:outline-none"
                      >
                        <option value="box">Box</option>
                        <option value="cylinder">Cylinder</option>
                        <option value="sphere">Sphere</option>
                      </select>

                      <label className="flex items-center gap-1 text-gray-400 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={obs.is_static}
                          onChange={(e) => handleUpdateObstacle(idx, { is_static: e.target.checked })}
                          className="rounded border-[#2d3345] bg-[#181c26] text-cyan-500"
                        />
                        <span>Static</span>
                      </label>

                      <button
                        type="button"
                        onClick={() => handleRemoveObstacle(idx)}
                        className="text-gray-500 hover:text-rose-400 p-1 rounded transition"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>

                  {/* Dimensions & Physics Inputs */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
                    <div>
                      <span className="text-gray-500 text-[10px] block">Position [X, Y, Z] (m)</span>
                      <div className="flex gap-1 font-mono">
                        <input
                          type="number"
                          step="0.1"
                          value={obs.position[0]}
                          onChange={(e) =>
                            handleUpdateObstacle(idx, {
                              position: [parseFloat(e.target.value) || 0, obs.position[1], obs.position[2]],
                            })
                          }
                          className="w-1/3 bg-[#181c26] border border-[#2d3345] rounded px-1.5 py-1 text-white text-[11px]"
                        />
                        <input
                          type="number"
                          step="0.1"
                          value={obs.position[1]}
                          onChange={(e) =>
                            handleUpdateObstacle(idx, {
                              position: [obs.position[0], parseFloat(e.target.value) || 0, obs.position[2]],
                            })
                          }
                          className="w-1/3 bg-[#181c26] border border-[#2d3345] rounded px-1.5 py-1 text-white text-[11px]"
                        />
                        <input
                          type="number"
                          step="0.05"
                          value={obs.position[2]}
                          onChange={(e) =>
                            handleUpdateObstacle(idx, {
                              position: [obs.position[0], obs.position[1], parseFloat(e.target.value) || 0],
                            })
                          }
                          className="w-1/3 bg-[#181c26] border border-[#2d3345] rounded px-1.5 py-1 text-white text-[11px]"
                        />
                      </div>
                    </div>

                    <div>
                      <span className="text-gray-500 text-[10px] block">
                        {obs.type === 'box'
                          ? 'Half-Sizes [X, Y, Z] (m)'
                          : obs.type === 'cylinder'
                          ? 'Radius, Half-Height (m)'
                          : 'Radius (m)'}
                      </span>
                      <div className="flex gap-1 font-mono">
                        <input
                          type="number"
                          step="0.05"
                          min="0.01"
                          value={obs.size[0]}
                          onChange={(e) =>
                            handleUpdateObstacle(idx, {
                              size: [
                                Math.max(0.01, parseFloat(e.target.value) || 0.1),
                                obs.size[1] || 0.1,
                                obs.size[2] || 0.1,
                              ],
                            })
                          }
                          className="w-1/3 bg-[#181c26] border border-[#2d3345] rounded px-1.5 py-1 text-white text-[11px]"
                        />
                        {obs.type !== 'sphere' && (
                          <input
                            type="number"
                            step="0.05"
                            min="0.01"
                            value={obs.size[1] || obs.size[0]}
                            onChange={(e) =>
                              handleUpdateObstacle(idx, {
                                size: [
                                  obs.size[0],
                                  Math.max(0.01, parseFloat(e.target.value) || 0.1),
                                  obs.size[2] || 0.1,
                                ],
                              })
                            }
                            className="w-1/3 bg-[#181c26] border border-[#2d3345] rounded px-1.5 py-1 text-white text-[11px]"
                          />
                        )}
                        {obs.type === 'box' && (
                          <input
                            type="number"
                            step="0.05"
                            min="0.01"
                            value={obs.size[2] || obs.size[0]}
                            onChange={(e) =>
                              handleUpdateObstacle(idx, {
                                size: [
                                  obs.size[0],
                                  obs.size[1],
                                  Math.max(0.01, parseFloat(e.target.value) || 0.1),
                                ],
                              })
                            }
                            className="w-1/3 bg-[#181c26] border border-[#2d3345] rounded px-1.5 py-1 text-white text-[11px]"
                          />
                        )}
                      </div>
                    </div>

                    <div>
                      <span className="text-gray-500 text-[10px] block">Mass (kg)</span>
                      <input
                        type="number"
                        step="0.5"
                        min="0.01"
                        value={obs.mass}
                        onChange={(e) =>
                          handleUpdateObstacle(idx, { mass: Math.max(0.01, parseFloat(e.target.value) || 1.0) })
                        }
                        className="w-full bg-[#181c26] border border-[#2d3345] rounded px-2 py-1 text-white font-mono text-[11px]"
                      />
                    </div>

                    <div>
                      <span className="text-gray-500 text-[10px] block">Friction (sliding)</span>
                      <input
                        type="number"
                        step="0.05"
                        min="0.01"
                        max="2.0"
                        value={obs.friction ? obs.friction[0] : 0.7}
                        onChange={(e) =>
                          handleUpdateObstacle(idx, {
                            friction: [Math.max(0.01, parseFloat(e.target.value) || 0.7), 0.005, 0.0001],
                          })
                        }
                        className="w-full bg-[#181c26] border border-[#2d3345] rounded px-2 py-1 text-white font-mono text-[11px]"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Section 4: Save as Reusable Template */}
          <div className="bg-[#181c26] border border-[#262c3a] rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Save size={16} className="text-amber-400" />
              Save Environment as Asset Library Template
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div>
                <label className="text-gray-400 block mb-1">Template Name</label>
                <input
                  type="text"
                  value={templateName}
                  onChange={(e) => setTemplateName(e.target.value)}
                  className="w-full bg-[#12141c] border border-[#2d3345] rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-amber-500"
                />
              </div>
              <div>
                <label className="text-gray-400 block mb-1">Description</label>
                <input
                  type="text"
                  value={templateDesc}
                  onChange={(e) => setTemplateDesc(e.target.value)}
                  className="w-full bg-[#12141c] border border-[#2d3345] rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-amber-500"
                />
              </div>
            </div>
            <div className="flex justify-end pt-1">
              <button
                type="button"
                onClick={handleSaveAsAsset}
                disabled={isSaving}
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-600/20 text-amber-300 border border-amber-500/30 hover:bg-amber-600/30 transition flex items-center gap-1.5 disabled:opacity-50"
              >
                <Save size={14} />
                {isSaving ? 'Saving Template...' : 'Save into Asset Library'}
              </button>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-[#2a2e39] bg-[#161922]">
          <button
            type="button"
            onClick={handleValidate}
            disabled={isValidating}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-[#222838] border border-[#30384f] text-gray-200 hover:bg-[#2b3347] hover:text-white transition flex items-center gap-2 disabled:opacity-50"
          >
            <Shield size={15} className="text-indigo-400" />
            {isValidating ? 'Validating MuJoCo Physics...' : 'Validate MuJoCo Physics'}
          </button>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-xs font-medium text-gray-400 hover:text-white hover:bg-[#202533] transition"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleApply}
              className="px-5 py-2 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-950/40 transition flex items-center gap-2"
            >
              Apply to Simulation Scene
              <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
