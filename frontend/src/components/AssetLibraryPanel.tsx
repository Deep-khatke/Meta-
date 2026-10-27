import React, { useState, useEffect } from 'react';
import {
  AssetModel, AssetCategory, Project
} from '../types';
import {
  fetchAssets, applyAsset, duplicateAsset, deleteCustomAsset,
  saveCurrentRobotAsset, saveCurrentSceneTemplate
} from '../services/api';
import {
  Package, Search, Filter, Plus, Copy, Trash2, CheckCircle2,
  AlertTriangle, Bot, Layers, Radio, Video, Cpu, Sparkles,
  LayoutTemplate, ArrowRight, Download, Upload, Eye, X
} from 'lucide-react';

interface AssetLibraryPanelProps {
  project: Project;
  onAssetApplied: (updatedProject: Project) => void;
  onOpenRobotCreator: () => void;
  onOpenEnvironmentCreator?: () => void;
  onOpenRobotStudio?: (robotModel?: any) => void;
  onShowToast: (msg: string) => void;
  onClose?: () => void;
  isDocked?: boolean;
}

export const AssetLibraryPanel: React.FC<AssetLibraryPanelProps> = ({
  project,
  onAssetApplied,
  onOpenRobotCreator,
  onOpenEnvironmentCreator,
  onOpenRobotStudio,
  onShowToast,
  onClose,
  isDocked = false,
}) => {
  const [assets, setAssets] = useState<AssetModel[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedTag, setSelectedTag] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedAsset, setSelectedAsset] = useState<AssetModel | null>(null);
  const [isApplying, setIsApplying] = useState<boolean>(false);
  const [dockView, setDockView] = useState<'catalog' | 'inspector'>('catalog');

  // Quick Save Modal State
  const [showSaveModal, setShowSaveModal] = useState<'robot' | 'template' | null>(null);
  const [saveName, setSaveName] = useState<string>('');
  const [saveDesc, setSaveDesc] = useState<string>('');

  const loadAssets = async () => {
    setIsLoading(true);
    try {
      const res = await fetchAssets(
        selectedCategory === 'all' ? undefined : selectedCategory,
        searchQuery || undefined,
        selectedTag || undefined
      );
      setAssets(res.assets);
      if (res.assets.length > 0 && !selectedAsset) {
        setSelectedAsset(res.assets[0]);
      }
    } catch (err: any) {
      onShowToast(err.message || 'Failed to load assets');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadAssets();
  }, [selectedCategory, searchQuery, selectedTag]);

  const handleApply = async (asset: AssetModel) => {
    setIsApplying(true);
    try {
      const res = await applyAsset(asset.id);
      onAssetApplied(res.project);
      onShowToast(`Applied '${asset.name}' to active simulation scene.`);
    } catch (err: any) {
      onShowToast(err.message || 'Failed to apply asset');
    } finally {
      setIsApplying(false);
    }
  };

  const handleDuplicate = async (asset: AssetModel) => {
    try {
      const clone = await duplicateAsset(asset.id);
      onShowToast(`Duplicated asset as '${clone.name}'`);
      await loadAssets();
      setSelectedAsset(clone);
    } catch (err: any) {
      onShowToast(err.message || 'Failed to duplicate asset');
    }
  };

  const handleDelete = async (asset: AssetModel) => {
    if (asset.is_built_in) {
      onShowToast('System built-in assets cannot be deleted.');
      return;
    }
    if (!window.confirm(`Delete custom asset '${asset.name}'?`)) return;

    try {
      await deleteCustomAsset(asset.id);
      onShowToast(`Deleted asset '${asset.name}'`);
      if (selectedAsset?.id === asset.id) {
        setSelectedAsset(null);
      }
      await loadAssets();
    } catch (err: any) {
      onShowToast(err.message || 'Failed to delete asset');
    }
  };

  const handleQuickSave = async () => {
    if (!saveName.trim()) {
      onShowToast('Please enter an asset name.');
      return;
    }
    try {
      if (showSaveModal === 'robot') {
        const saved = await saveCurrentRobotAsset(saveName, saveDesc);
        onShowToast(`Saved current robot as reusable asset '${saved.name}'`);
      } else {
        const saved = await saveCurrentSceneTemplate(saveName, saveDesc);
        onShowToast(`Saved current scene as template '${saved.name}'`);
      }
      setShowSaveModal(null);
      setSaveName('');
      setSaveDesc('');
      await loadAssets();
    } catch (err: any) {
      onShowToast(err.message || 'Failed to save asset');
    }
  };

  const getCategoryIcon = (cat: AssetCategory, hint?: string) => {
    if (cat === 'robot') {
      return hint === 'arm' ? <Cpu size={14} color="#10b981" /> : <Bot size={14} color="#3b82f6" />;
    }
    if (cat === 'obstacle') {
      return <Layers size={14} color="#e06c75" />;
    }
    if (cat === 'sensor') {
      return hint === 'video' ? <Video size={14} color="#38bdf8" /> : <Radio size={14} color="#c084fc" />;
    }
    return <LayoutTemplate size={14} color="#f59e0b" />;
  };

  // Collect all unique tags from currently loaded assets
  const availableTags = Array.from(new Set(assets.flatMap((a) => a.tags))).slice(0, 8);

  return (
    <div className="rf-asset-library-container" style={{
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      height: '100%',
      backgroundColor: 'var(--bg-primary)',
      color: 'var(--text-primary)',
      overflow: 'hidden'
    }}>
      {/* Top Header Bar */}
      <div style={{
        padding: isDocked ? '8px 12px' : '10px 16px',
        backgroundColor: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 8,
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Package size={isDocked ? 15 : 17} color="var(--accent-blue)" />
          <span style={{ fontWeight: 700, fontSize: isDocked ? 12 : 13, letterSpacing: '0.5px' }}>
            {isDocked ? 'ASSETS' : 'ASSET & SCENE LIBRARY'}
          </span>
          <span style={{ fontSize: 10, padding: '1px 5px', backgroundColor: 'rgba(59, 130, 246, 0.15)', color: 'var(--accent-blue)', borderRadius: 4 }}>
            {assets.length}
          </span>
        </div>

        {isDocked ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {/* View Switcher: Catalog vs Inspector */}
            <div style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', borderRadius: 5, padding: 2, border: '1px solid var(--border-subtle)' }}>
              <button
                className="rf-btn"
                style={{
                  fontSize: 10,
                  padding: '2px 8px',
                  background: dockView === 'catalog' ? 'var(--accent-blue)' : 'transparent',
                  color: dockView === 'catalog' ? '#fff' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: 4
                }}
                onClick={() => setDockView('catalog')}
              >
                Catalog
              </button>
              <button
                className="rf-btn"
                style={{
                  fontSize: 10,
                  padding: '2px 8px',
                  background: dockView === 'inspector' ? 'var(--accent-blue)' : 'transparent',
                  color: dockView === 'inspector' ? '#fff' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: 4
                }}
                onClick={() => setDockView('inspector')}
                disabled={!selectedAsset}
                title={selectedAsset ? selectedAsset.name : 'Select an asset first'}
              >
                Inspect
              </button>
            </div>
            {onClose && (
              <button
                className="rf-btn rf-btn-icon"
                onClick={onClose}
                title="Close Asset Library"
                style={{ width: 22, height: 22, padding: 3 }}
              >
                <X size={12} />
              </button>
            )}
          </div>
        ) : (
          /* Quick Actions */
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button
              className="rf-btn"
              style={{ fontSize: 11, padding: '4px 10px', gap: 5 }}
              onClick={() => {
                setSaveName(project.generic_robot ? project.generic_robot.name : project.robot.name);
                setSaveDesc('Exported from active simulation');
                setShowSaveModal('robot');
              }}
              title="Save the currently active robot configuration into your reusable Asset Library"
            >
              <Download size={12} color="var(--accent-green)" />
              <span>Save Robot</span>
            </button>

            <button
              className="rf-btn"
              style={{ fontSize: 11, padding: '4px 10px', gap: 5 }}
              onClick={() => {
                setSaveName(project.name + ' Template');
                setSaveDesc(project.description || 'Custom simulation scene template');
                setShowSaveModal('template');
              }}
              title="Save the entire current scene (robot, obstacles, boundaries, sensors) as a reusable template"
            >
              <LayoutTemplate size={12} color="#f59e0b" />
              <span>Save Template</span>
            </button>

            <button
              className="rf-btn rf-btn-primary"
              style={{ fontSize: 11, padding: '4px 12px', gap: 5 }}
              onClick={onOpenRobotCreator}
              title="Open Robot Creator: Build a new mobile or articulated robot with physics validation"
            >
              <Plus size={13} />
              <span>Create Robot</span>
            </button>

            {onClose && (
              <button
                className="rf-btn rf-btn-icon"
                onClick={onClose}
                title="Close Asset Library"
                style={{ marginLeft: 4 }}
              >
                <X size={14} />
              </button>
            )}
          </div>
        )}
      </div>

      {/* Filter and Search Bar */}
      <div style={{
        padding: isDocked ? '6px 10px' : '8px 16px',
        backgroundColor: 'var(--bg-panel)',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: isDocked ? 'column' : 'row',
        alignItems: isDocked ? 'stretch' : 'center',
        justifyContent: 'space-between',
        gap: isDocked ? 6 : 12,
        flexShrink: 0
      }}>
        {/* Category Tabs */}
        <div style={{ display: 'flex', gap: 4, overflowX: 'auto', paddingBottom: 2 }}>
          {[
            { id: 'all', label: 'All' },
            { id: 'robot', label: 'Robots' },
            { id: 'obstacle', label: 'Obstacles' },
            { id: 'sensor', label: 'Sensors' },
            { id: 'template', label: 'Templates' },
          ].map((cat) => (
            <button
              key={cat.id}
              className={`rf-btn ${selectedCategory === cat.id ? 'rf-btn-primary' : ''}`}
              style={{ fontSize: 10, padding: '2px 8px', whiteSpace: 'nowrap' }}
              onClick={() => setSelectedCategory(cat.id)}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Search Input */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: 1, maxWidth: isDocked ? '100%' : 320 }}>
          <div style={{ position: 'relative', width: '100%' }}>
            <Search size={12} style={{ position: 'absolute', left: 8, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              className="rf-input"
              style={{ width: '100%', paddingLeft: 26, fontSize: 11 }}
              placeholder="Search assets or tags..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          {selectedTag && (
            <button
              className="rf-btn"
              style={{ fontSize: 10, padding: '2px 6px', color: 'var(--accent-orange)' }}
              onClick={() => setSelectedTag('')}
              title="Clear tag filter"
            >
              #{selectedTag} ✕
            </button>
          )}
        </div>
      </div>

      {/* Main Split Body: Cards Grid on Left, Inspector on Right */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Asset Cards Grid */}
        <div style={{
          flex: 1,
          padding: isDocked ? 8 : 14,
          overflowY: 'auto',
          display: isDocked && dockView === 'inspector' ? 'none' : 'grid',
          gridTemplateColumns: isDocked ? '1fr' : 'repeat(auto-fill, minmax(260px, 1fr))',
          gap: isDocked ? 8 : 12,
          alignContent: 'start'
        }}>
          {isLoading ? (
            <div style={{ gridColumn: '1 / -1', padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
              Loading asset catalog...
            </div>
          ) : assets.length === 0 ? (
            <div style={{ gridColumn: '1 / -1', padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
              No assets match current filters.
            </div>
          ) : (
            assets.map((asset) => {
              const isSelected = selectedAsset?.id === asset.id;
              return (
                <div
                  key={asset.id}
                  onClick={() => setSelectedAsset(asset)}
                  style={{
                    backgroundColor: isSelected ? 'var(--bg-panel-hover)' : 'var(--bg-panel)',
                    border: isSelected ? '1px solid var(--accent-blue)' : '1px solid var(--border-subtle)',
                    borderRadius: 7,
                    padding: 12,
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8,
                    transition: 'border-color 0.15s, background-color 0.15s',
                    boxShadow: isSelected ? '0 0 12px rgba(59, 130, 246, 0.2)' : 'none'
                  }}
                >
                  {/* Card Header */}
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 6 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{
                        padding: 6,
                        borderRadius: 6,
                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        {getCategoryIcon(asset.category, asset.thumbnail_hint)}
                      </div>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: 12, color: 'var(--text-primary)' }}>
                          {asset.name}
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                          {asset.category}
                        </div>
                      </div>
                    </div>

                    <span style={{
                      fontSize: 9,
                      padding: '2px 5px',
                      borderRadius: 3,
                      fontWeight: 600,
                      backgroundColor: asset.is_built_in ? 'rgba(59, 130, 246, 0.15)' : 'rgba(34, 197, 94, 0.15)',
                      color: asset.is_built_in ? 'var(--accent-blue)' : 'var(--accent-green)'
                    }}>
                      {asset.is_built_in ? 'SYSTEM' : 'CUSTOM'}
                    </span>
                  </div>

                  {/* Description */}
                  <div style={{
                    fontSize: 11,
                    color: 'var(--text-secondary)',
                    lineHeight: 1.35,
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden'
                  }}>
                    {asset.description || 'Reusable simulation component.'}
                  </div>

                  {/* Tags */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {asset.tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedTag(tag);
                        }}
                        style={{
                          fontSize: 9,
                          padding: '1px 5px',
                          backgroundColor: 'rgba(255, 255, 255, 0.04)',
                          border: '1px solid var(--border-subtle)',
                          borderRadius: 3,
                          color: 'var(--text-muted)'
                        }}
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>

                  {/* Card Action Strip */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginTop: 4,
                    paddingTop: 8,
                    borderTop: '1px solid var(--border-subtle)'
                  }}>
                    <div style={{ display: 'flex', gap: 5 }}>
                      <button
                        className="rf-btn rf-btn-primary"
                        style={{ fontSize: 10, padding: '3px 8px', gap: 4 }}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleApply(asset);
                        }}
                        disabled={isApplying}
                      >
                        <ArrowRight size={11} />
                        <span>{asset.category === 'template' ? 'Apply' : 'Add'}</span>
                      </button>

                      {isDocked && (
                        <button
                          className="rf-btn"
                          style={{ fontSize: 10, padding: '3px 8px', gap: 4 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedAsset(asset);
                            setDockView('inspector');
                          }}
                        >
                          <Eye size={11} />
                          <span>Inspect</span>
                        </button>
                      )}
                    </div>

                    <div style={{ display: 'flex', gap: 4 }}>
                      <button
                        className="rf-btn rf-btn-icon"
                        style={{ width: 22, height: 22, padding: 3 }}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDuplicate(asset);
                        }}
                        title="Duplicate as new custom asset"
                      >
                        <Copy size={11} />
                      </button>

                      {!asset.is_built_in && (
                        <button
                          className="rf-btn rf-btn-icon rf-btn-danger"
                          style={{ width: 22, height: 22, padding: 3 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(asset);
                          }}
                          title="Delete custom asset"
                        >
                          <Trash2 size={11} />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Selected Asset Inspector Sidebar */}
        {selectedAsset && (
          <aside style={{
            width: isDocked ? '100%' : 320,
            borderLeft: isDocked ? 'none' : '1px solid var(--border-subtle)',
            backgroundColor: 'var(--bg-secondary)',
            display: isDocked ? (dockView === 'inspector' ? 'flex' : 'none') : 'flex',
            flexDirection: 'column',
            overflowY: 'auto',
            flex: isDocked ? 1 : undefined
          }}>
            <div style={{
              padding: '10px 12px',
              borderBottom: '1px solid var(--border-subtle)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: 'var(--bg-panel)',
              flexShrink: 0
            }}>
              {isDocked && (
                <button
                  className="rf-btn"
                  onClick={() => setDockView('catalog')}
                  style={{ fontSize: 10, padding: '3px 8px', marginRight: 6 }}
                >
                  ← Catalog
                </button>
              )}
              <span style={{ fontWeight: 600, fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Asset Inspector
              </span>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                {selectedAsset.id}
              </span>
            </div>

            <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
                  {selectedAsset.name}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                  Category: <strong style={{ color: 'var(--text-secondary)' }}>{selectedAsset.category}</strong> • v{selectedAsset.version}
                </div>
              </div>

              <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                {selectedAsset.description || 'No description provided.'}
              </div>

              {/* Physical Properties Overview */}
              <div style={{ background: 'var(--bg-panel)', padding: 10, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>
                  Physical Specifications
                </div>

                {selectedAsset.category === 'robot' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11 }}>
                    {selectedAsset.data.type === 'mobile' ? (
                      <>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Chassis Size:</span>
                          <span>[{selectedAsset.data.config.chassis_size.join(', ')}]m</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Wheel Base:</span>
                          <span>{selectedAsset.data.config.wheel_base}m</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Total Mass:</span>
                          <span>{selectedAsset.data.config.chassis_mass} kg</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Max Linear Vel:</span>
                          <span>{selectedAsset.data.config.max_linear_velocity} m/s</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Kinematic Links:</span>
                          <span>{selectedAsset.data.config.links?.length || 0} links</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Actuated Joints:</span>
                          <span>{selectedAsset.data.config.joints?.length || 0} joints</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Base Type:</span>
                          <span>{selectedAsset.data.config.is_fixed_base ? 'Fixed Base' : 'Mobile Base'}</span>
                        </div>
                      </>
                    )}
                  </div>
                )}

                {selectedAsset.category === 'obstacle' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Geometry Type:</span>
                      <span style={{ textTransform: 'capitalize' }}>{selectedAsset.data.type}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Dimensions:</span>
                      <span>[{selectedAsset.data.size?.join(', ')}]m</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Mass:</span>
                      <span>{selectedAsset.data.mass} kg</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Physics Mode:</span>
                      <span>{selectedAsset.data.is_static ? 'Static Barrier' : 'Dynamic Rigid Body'}</span>
                    </div>
                  </div>
                )}

                {selectedAsset.category === 'sensor' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Sensor Modality:</span>
                      <span style={{ textTransform: 'uppercase' }}>{selectedAsset.data.type}</span>
                    </div>
                    {selectedAsset.data.type === 'lidar' && (
                      <>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Beams:</span>
                          <span>{selectedAsset.data.config?.num_beams} rays</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Field of View:</span>
                          <span>{selectedAsset.data.config?.fov_deg}°</span>
                        </div>
                      </>
                    )}
                    {selectedAsset.data.type === 'camera' && (
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Optics FOV:</span>
                        <span>{selectedAsset.data.config?.fov_deg}°</span>
                      </div>
                    )}
                  </div>
                )}

                {selectedAsset.category === 'template' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Robot Model:</span>
                      <span>{selectedAsset.data.robot?.name || 'Default'}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Obstacle Count:</span>
                      <span>{selectedAsset.data.objects?.length || 0} objects</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Boundaries:</span>
                      <span>{selectedAsset.data.settings?.enable_boundaries ? 'Enclosed Walls' : 'Open Floor'}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons in Inspector */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8 }}>
                <button
                  className="rf-btn rf-btn-primary"
                  style={{ padding: '8px 12px', fontSize: 12, fontWeight: 600, gap: 6 }}
                  onClick={() => handleApply(selectedAsset)}
                  disabled={isApplying}
                >
                  <ArrowRight size={13} />
                  <span>{selectedAsset.category === 'template' ? 'Instantiate Scene Template' : 'Add to Current Scene'}</span>
                </button>

                {selectedAsset.category === 'robot' && onOpenRobotStudio && (
                  <button
                    className="rf-btn"
                    style={{
                      padding: '7px 12px',
                      fontSize: 11,
                      gap: 6,
                      background: 'rgba(59, 130, 246, 0.15)',
                      borderColor: 'rgba(59, 130, 246, 0.4)',
                      color: '#60a5fa',
                      fontWeight: 600,
                    }}
                    onClick={() => {
                      const robotData = selectedAsset.data?.config || selectedAsset.data;
                      onOpenRobotStudio(robotData);
                    }}
                  >
                    <Bot size={12} />
                    <span>Open in Robot Studio</span>
                  </button>
                )}

                <button
                  className="rf-btn"
                  style={{ padding: '6px 12px', fontSize: 11, gap: 6 }}
                  onClick={() => handleDuplicate(selectedAsset)}
                >
                  <Copy size={12} />
                  <span>Duplicate as Custom Asset</span>
                </button>

                {!selectedAsset.is_built_in && (
                  <button
                    className="rf-btn rf-btn-danger"
                    style={{ padding: '6px 12px', fontSize: 11, gap: 6 }}
                    onClick={() => handleDelete(selectedAsset)}
                  >
                    <Trash2 size={12} />
                    <span>Delete Custom Asset</span>
                  </button>
                )}
              </div>
            </div>
          </aside>
        )}
      </div>

      {/* Quick Save Modal */}
      {showSaveModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.7)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 8,
            width: 420,
            padding: 18,
            display: 'flex',
            flexDirection: 'column',
            gap: 14,
            boxShadow: '0 12px 32px rgba(0,0,0,0.6)'
          }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
              {showSaveModal === 'robot' ? 'Save Current Robot as Reusable Asset' : 'Save Scene as Reusable Template'}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Asset Name</span>
              <input
                type="text"
                className="rf-input"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                placeholder="e.g., Heavy Patrol Scout v2"
              />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Description</span>
              <textarea
                className="rf-input"
                style={{ height: 60, resize: 'none' }}
                value={saveDesc}
                onChange={(e) => setSaveDesc(e.target.value)}
                placeholder="Physical setup and application notes..."
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 4 }}>
              <button className="rf-btn" onClick={() => setShowSaveModal(null)}>Cancel</button>
              <button className="rf-btn rf-btn-primary" onClick={handleQuickSave}>Save to Library</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
