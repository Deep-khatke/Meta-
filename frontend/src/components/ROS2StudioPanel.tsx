import React, { useState, useEffect } from 'react';
import {
  Radio, Play, Square, Download, Trash2, RefreshCw,
  Send, Database, Share2, Layers, Cpu, ArrowUp, ArrowDown,
  ArrowLeft, ArrowRight, StopCircle, CheckCircle2, AlertCircle,
  FileText, Users
} from 'lucide-react';
import {
  fetchROS2Topics, fetchROS2TopicMessage, fetchROS2TF, sendROS2CmdVel,
  startROS2Bag, stopROS2Bag, fetchROS2BagStatus, fetchROS2Bags,
  deleteROS2Bag, deploySwarm, fetchSwarmStatus, clearSwarm,
  ROS2TopicInfo, ROS2BagMetadata, ROS2BagStatus, SwarmStatusResponse
} from '../services/api';
import { SimulationState } from '../types';

interface ROS2StudioPanelProps {
  onShowToast: (msg: string) => void;
  simState?: SimulationState;
}

export const ROS2StudioPanel: React.FC<ROS2StudioPanelProps> = ({
  onShowToast,
  simState
}) => {
  const [subTab, setSubTab] = useState<'topics' | 'teleop' | 'tf' | 'bags' | 'swarm'>('topics');
  
  // Topics state
  const [topics, setTopics] = useState<ROS2TopicInfo[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<string>('/odom');
  const [topicData, setTopicData] = useState<any>(null);
  const [loadingTopic, setLoadingTopic] = useState(false);
  const [autoEcho, setAutoEcho] = useState(true);

  // Teleop publisher state
  const [linearX, setLinearX] = useState<number>(0.0);
  const [angularZ, setAngularZ] = useState<number>(0.0);
  const [lastPublishedTwist, setLastPublishedTwist] = useState<any>(null);

  // TF Tree state
  const [tfData, setTfData] = useState<any>(null);
  const [loadingTf, setLoadingTf] = useState(false);

  // Bag Recorder state
  const [bagStatus, setBagStatus] = useState<ROS2BagStatus>({
    is_recording: false,
    current_bag_id: null,
    current_bag_name: null,
    message_count: 0
  });
  const [newBagName, setNewBagName] = useState('simulation_run_dataset');
  const [savedBags, setSavedBags] = useState<ROS2BagMetadata[]>([]);
  const [inspectingBag, setInspectingBag] = useState<any>(null);

  // Swarm state
  const [swarmFormation, setSwarmFormation] = useState<'v_shape' | 'line' | 'column' | 'circle' | 'diamond'>('v_shape');
  const [swarmCount, setSwarmCount] = useState<number>(4);
  const [swarmSpacing, setSwarmSpacing] = useState<number>(1.2);
  const [swarmStatus, setSwarmStatus] = useState<SwarmStatusResponse | null>(null);

  // Initial load
  useEffect(() => {
    loadTopics();
    loadBags();
    loadSwarmStatus();
  }, []);

  // Poll topic data if autoEcho is on
  useEffect(() => {
    if (!autoEcho || subTab !== 'topics' || !selectedTopic) return;
    const interval = setInterval(() => {
      echoTopic(selectedTopic, false);
    }, 1000);
    return () => clearInterval(interval);
  }, [autoEcho, subTab, selectedTopic]);

  // Poll bag recording status if recording
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const stat = await fetchROS2BagStatus();
        setBagStatus(stat);
      } catch (e) {
        // quiet
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const loadTopics = async () => {
    try {
      const res = await fetchROS2Topics();
      setTopics(res.topics || []);
      if (!selectedTopic && res.topics?.length > 0) {
        setSelectedTopic(res.topics[0].topic);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const echoTopic = async (topicName: string, showToast = true) => {
    setLoadingTopic(true);
    try {
      const data = await fetchROS2TopicMessage(topicName);
      setTopicData(data);
      if (showToast) onShowToast(`Topic ${topicName} refreshed`);
    } catch (e: any) {
      setTopicData({ error: e.message || 'Topic unavailable or no publisher' });
    } finally {
      setLoadingTopic(false);
    }
  };

  const loadTf = async () => {
    setLoadingTf(true);
    try {
      const data = await fetchROS2TF();
      setTfData(data);
    } catch (e: any) {
      onShowToast(`Failed to load TF2: ${e.message}`);
    } finally {
      setLoadingTf(false);
    }
  };

  const loadBags = async () => {
    try {
      const res = await fetchROS2Bags();
      setSavedBags(res.bags || []);
    } catch (e) {
      console.error(e);
    }
  };

  const handleStartRecording = async () => {
    try {
      await startROS2Bag(newBagName || 'rosbag2_recording');
      onShowToast(`Started recording ROS 2 Bag: ${newBagName}`);
      const stat = await fetchROS2BagStatus();
      setBagStatus(stat);
    } catch (e: any) {
      onShowToast(`Failed to start bag: ${e.message}`);
    }
  };

  const handleStopRecording = async () => {
    try {
      const res = await stopROS2Bag();
      onShowToast(`Saved ROS 2 Bag with ${res.metadata?.total_messages || 0} messages`);
      setBagStatus({ is_recording: false, current_bag_id: null, current_bag_name: null, message_count: 0 });
      loadBags();
    } catch (e: any) {
      onShowToast(`Failed to stop bag: ${e.message}`);
    }
  };

  const handleDeleteBag = async (bagId: string) => {
    try {
      await deleteROS2Bag(bagId);
      onShowToast(`Deleted bag ${bagId}`);
      loadBags();
      if (inspectingBag?.metadata?.bag_id === bagId) {
        setInspectingBag(null);
      }
    } catch (e: any) {
      onShowToast(`Delete failed: ${e.message}`);
    }
  };

  const handlePublishCmdVel = async (lin: number, ang: number) => {
    const payload = {
      linear: { x: Number(lin.toFixed(2)), y: 0.0, z: 0.0 },
      angular: { x: 0.0, y: 0.0, z: Number(ang.toFixed(2)) }
    };
    try {
      await sendROS2CmdVel(payload);
      setLastPublishedTwist(payload);
      setLinearX(lin);
      setAngularZ(ang);
    } catch (e: any) {
      onShowToast(`Twist publish failed: ${e.message}`);
    }
  };

  const loadSwarmStatus = async () => {
    try {
      const res = await fetchSwarmStatus();
      setSwarmStatus(res);
    } catch (e) {
      console.error(e);
    }
  };

  const handleDeploySwarm = async () => {
    try {
      const res = await deploySwarm(swarmFormation, swarmCount, swarmSpacing);
      onShowToast(`Deployed ${swarmCount} robots in ${swarmFormation.toUpperCase()} formation`);
      loadSwarmStatus();
    } catch (e: any) {
      onShowToast(`Swarm deployment failed: ${e.message}`);
    }
  };

  const handleClearSwarm = async () => {
    try {
      await clearSwarm();
      onShowToast('Cleared swarm fleet from scene');
      loadSwarmStatus();
    } catch (e: any) {
      onShowToast(`Clear failed: ${e.message}`);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', color: 'var(--text-primary)', fontSize: 12 }}>
      {/* Top Banner / Sub-tab Selector */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '6px 10px',
        backgroundColor: 'var(--bg-panel)',
        borderBottom: '1px solid var(--border-subtle)',
        gap: 8
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Radio size={14} color="#38bdf8" />
          <span style={{ fontWeight: 600, color: '#38bdf8' }}>ROS 2 Robotics Bridge</span>
          <span style={{
            fontSize: 10,
            padding: '1px 6px',
            borderRadius: 10,
            backgroundColor: 'rgba(56, 189, 248, 0.15)',
            color: '#38bdf8',
            fontFamily: 'var(--font-mono)'
          }}>
            /roboforge_bridge_node
          </span>
          {bagStatus.is_recording && (
            <span style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 10,
              padding: '1px 6px',
              borderRadius: 10,
              backgroundColor: 'rgba(239, 68, 68, 0.2)',
              color: 'var(--accent-red)',
              fontWeight: 600
            }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: 'var(--accent-red)' }} />
              REC ({bagStatus.message_count} msgs)
            </span>
          )}
        </div>

        {/* Subtab Buttons */}
        <div style={{ display: 'flex', gap: 4 }}>
          {[
            { id: 'topics', label: 'Topics & Live Echo', icon: <Radio size={12} /> },
            { id: 'teleop', label: '/cmd_vel Publisher', icon: <Send size={12} /> },
            { id: 'tf', label: 'TF2 Tree', icon: <Share2 size={12} /> },
            { id: 'bags', label: 'Bag Recorder', icon: <Database size={12} /> },
            { id: 'swarm', label: 'Swarm Fleet', icon: <Users size={12} /> },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setSubTab(tab.id as any);
                if (tab.id === 'tf') loadTf();
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '3px 9px',
                borderRadius: 4,
                border: 'none',
                backgroundColor: subTab === tab.id ? 'var(--accent-blue)' : 'var(--bg-primary)',
                color: subTab === tab.id ? '#fff' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: 11,
                fontWeight: subTab === tab.id ? 600 : 400
              }}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 10 }}>
        {/* --- 1. TOPICS & ECHO TAB --- */}
        {subTab === 'topics' && (
          <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 10, height: '100%' }}>
            {/* Topic List */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 8,
              display: 'flex',
              flexDirection: 'column',
              gap: 6
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: 4, borderBottom: '1px solid var(--border-subtle)' }}>
                <span style={{ fontWeight: 600, fontSize: 11 }}>Active Topics ({topics.length})</span>
                <button
                  onClick={loadTopics}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
                  title="Refresh Topics"
                >
                  <RefreshCw size={12} />
                </button>
              </div>

              <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 }}>
                {topics.map((t) => {
                  const isSelected = selectedTopic === t.topic;
                  return (
                    <div
                      key={t.topic}
                      onClick={() => {
                        setSelectedTopic(t.topic);
                        echoTopic(t.topic);
                      }}
                      style={{
                        padding: '6px 8px',
                        borderRadius: 4,
                        cursor: 'pointer',
                        backgroundColor: isSelected ? 'rgba(56, 189, 248, 0.15)' : 'var(--bg-primary)',
                        border: isSelected ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid transparent',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontWeight: 600, color: isSelected ? '#38bdf8' : 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {t.topic}
                        </span>
                        <span style={{
                          fontSize: 9,
                          padding: '0 4px',
                          borderRadius: 3,
                          backgroundColor: t.mode === 'publication' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(168, 85, 247, 0.15)',
                          color: t.mode === 'publication' ? 'var(--accent-green)' : '#c084fc'
                        }}>
                          {t.mode === 'publication' ? 'PUB' : 'SUB'}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
                        <span style={{ fontFamily: 'var(--font-mono)' }}>{t.type}</span>
                        {t.rate_hz && <span>{t.rate_hz} Hz</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Topic Live Echo Inspector */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 8,
              display: 'flex',
              flexDirection: 'column',
              gap: 6
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: 4, borderBottom: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontWeight: 600, fontSize: 11 }}>Live Echo:</span>
                  <span style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{selectedTopic}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={autoEcho}
                      onChange={(e) => setAutoEcho(e.target.checked)}
                    />
                    <span>Stream (1Hz)</span>
                  </label>
                  <button
                    onClick={() => echoTopic(selectedTopic)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      padding: '2px 8px',
                      borderRadius: 4,
                      backgroundColor: 'var(--bg-primary)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-primary)',
                      cursor: 'pointer',
                      fontSize: 10
                    }}
                  >
                    <RefreshCw size={10} className={loadingTopic ? 'animate-spin' : ''} />
                    <span>Echo Now</span>
                  </button>
                </div>
              </div>

              <div style={{
                flex: 1,
                overflowY: 'auto',
                backgroundColor: 'var(--bg-primary)',
                borderRadius: 4,
                padding: 8,
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                color: '#38bdf8'
              }}>
                <pre style={{ margin: 0 }}>
                  {topicData ? JSON.stringify(topicData, null, 2) : '// Click a topic to inspect serialized ROS 2 payload...'}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* --- 2. /cmd_vel TELEOP PUBLISHER TAB --- */}
        {subTab === 'teleop' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, height: '100%' }}>
            {/* Controls Card */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Send size={14} color="#38bdf8" />
                <span style={{ fontWeight: 600 }}>geometry_msgs/msg/Twist Publisher (/cmd_vel)</span>
              </div>

              {/* Directional Pad */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, padding: '10px 0' }}>
                <button
                  className="rf-btn"
                  onClick={() => handlePublishCmdVel(0.5, 0.0)}
                  style={{ width: 60, height: 40, justifyContent: 'center' }}
                  title="Forward (0.5 m/s)"
                >
                  <ArrowUp size={16} />
                </button>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    className="rf-btn"
                    onClick={() => handlePublishCmdVel(0.0, 1.0)}
                    style={{ width: 60, height: 40, justifyContent: 'center' }}
                    title="Rotate Left (1.0 rad/s)"
                  >
                    <ArrowLeft size={16} />
                  </button>
                  <button
                    className="rf-btn rf-btn-danger"
                    onClick={() => handlePublishCmdVel(0.0, 0.0)}
                    style={{ width: 60, height: 40, justifyContent: 'center' }}
                    title="Stop (0, 0)"
                  >
                    <StopCircle size={16} />
                  </button>
                  <button
                    className="rf-btn"
                    onClick={() => handlePublishCmdVel(0.0, -1.0)}
                    style={{ width: 60, height: 40, justifyContent: 'center' }}
                    title="Rotate Right (-1.0 rad/s)"
                  >
                    <ArrowRight size={16} />
                  </button>
                </div>
                <button
                  className="rf-btn"
                  onClick={() => handlePublishCmdVel(-0.5, 0.0)}
                  style={{ width: 60, height: 40, justifyContent: 'center' }}
                  title="Reverse (-0.5 m/s)"
                >
                  <ArrowDown size={16} />
                </button>
              </div>

              {/* Sliders */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                    <span>Linear Velocity X (m/s)</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>{linearX.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="-1.5"
                    max="1.5"
                    step="0.05"
                    value={linearX}
                    onChange={(e) => handlePublishCmdVel(parseFloat(e.target.value), angularZ)}
                    style={{ width: '100%' }}
                  />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                    <span>Angular Velocity Z (rad/s)</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>{angularZ.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="-3.0"
                    max="3.0"
                    step="0.1"
                    value={angularZ}
                    onChange={(e) => handlePublishCmdVel(linearX, parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
            </div>

            {/* Telemetry & Last Sent Twist */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 8
            }}>
              <span style={{ fontWeight: 600, fontSize: 11 }}>Published ROS 2 Message Frame</span>
              <div style={{
                flex: 1,
                backgroundColor: 'var(--bg-primary)',
                borderRadius: 4,
                padding: 10,
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                color: '#38bdf8',
                overflowY: 'auto'
              }}>
                <pre style={{ margin: 0 }}>
                  {JSON.stringify(
                    lastPublishedTwist || {
                      topic: '/cmd_vel',
                      type: 'geometry_msgs/msg/Twist',
                      linear: { x: linearX, y: 0.0, z: 0.0 },
                      angular: { x: 0.0, y: 0.0, z: angularZ }
                    },
                    null,
                    2
                  )}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* --- 3. TF2 TREE TAB --- */}
        {subTab === 'tf' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, height: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Share2 size={14} color="#38bdf8" />
                <span style={{ fontWeight: 600 }}>TF2 Coordinate Tree (tf2_msgs/msg/TFMessage)</span>
              </div>
              <button
                className="rf-btn"
                onClick={loadTf}
                style={{ fontSize: 11 }}
              >
                <RefreshCw size={11} className={loadingTf ? 'animate-spin' : ''} />
                <span>Refresh Tree</span>
              </button>
            </div>

            {/* Visual Transformation Graph */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: 10,
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              overflowX: 'auto'
            }}>
              {['map', 'odom', 'base_footprint', 'base_link', 'lidar_link / imu_link'].map((node, i, arr) => (
                <React.Fragment key={node}>
                  <div style={{
                    padding: '6px 12px',
                    borderRadius: 4,
                    backgroundColor: 'var(--bg-primary)',
                    border: '1px solid #38bdf8',
                    color: '#38bdf8',
                    fontWeight: 600,
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    whiteSpace: 'nowrap'
                  }}>
                    {node}
                  </div>
                  {i < arr.length - 1 && <span style={{ color: 'var(--text-muted)' }}>&rarr;</span>}
                </React.Fragment>
              ))}
            </div>

            {/* Raw Transforms JSON */}
            <div style={{
              flex: 1,
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 10,
              overflowY: 'auto',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: '#38bdf8'
            }}>
              <pre style={{ margin: 0 }}>
                {tfData ? JSON.stringify(tfData, null, 2) : '// Loading TF2 transform graph...'}
              </pre>
            </div>
          </div>
        )}

        {/* --- 4. ROS 2 BAG RECORDER TAB --- */}
        {subTab === 'bags' && (
          <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 12, height: '100%' }}>
            {/* Recorder Controls */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Database size={14} color="#38bdf8" />
                <span style={{ fontWeight: 600 }}>ROS 2 Bag Dataset Recorder</span>
              </div>

              <div>
                <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                  Dataset Bag Name
                </label>
                <input
                  type="text"
                  value={newBagName}
                  onChange={(e) => setNewBagName(e.target.value)}
                  disabled={bagStatus.is_recording}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    borderRadius: 4,
                    backgroundColor: 'var(--bg-primary)',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-primary)',
                    fontSize: 11
                  }}
                />
              </div>

              {/* Status Box */}
              <div style={{
                padding: 10,
                borderRadius: 4,
                backgroundColor: bagStatus.is_recording ? 'rgba(239, 68, 68, 0.1)' : 'var(--bg-primary)',
                border: bagStatus.is_recording ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid var(--border-subtle)',
                display: 'flex',
                flexDirection: 'column',
                gap: 4
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                  <span>Recording Status:</span>
                  <span style={{
                    fontWeight: 600,
                    color: bagStatus.is_recording ? 'var(--accent-red)' : 'var(--text-muted)'
                  }}>
                    {bagStatus.is_recording ? 'RECORDING ACTIVE' : 'IDLE'}
                  </span>
                </div>
                {bagStatus.is_recording && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                    <span>Messages Buffered:</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#38bdf8' }}>
                      {bagStatus.message_count}
                    </span>
                  </div>
                )}
              </div>

              {/* Start / Stop Action Buttons */}
              <div style={{ display: 'flex', gap: 8 }}>
                {!bagStatus.is_recording ? (
                  <button
                    className="rf-btn rf-btn-primary"
                    onClick={handleStartRecording}
                    style={{ flex: 1, padding: '8px 12px', justifyContent: 'center' }}
                  >
                    <Play size={13} />
                    <span>Start Recording</span>
                  </button>
                ) : (
                  <button
                    className="rf-btn rf-btn-danger"
                    onClick={handleStopRecording}
                    style={{ flex: 1, padding: '8px 12px', justifyContent: 'center' }}
                  >
                    <Square size={13} />
                    <span>Stop & Save Bag</span>
                  </button>
                )}
              </div>
            </div>

            {/* Saved Bags Table */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 10,
              display: 'flex',
              flexDirection: 'column',
              gap: 8
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: 11 }}>Recorded ROS 2 Datasets ({savedBags.length})</span>
                <button
                  onClick={loadBags}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
                >
                  <RefreshCw size={12} />
                </button>
              </div>

              <div style={{ flex: 1, overflowY: 'auto' }}>
                {savedBags.length === 0 ? (
                  <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: 40, fontSize: 11 }}>
                    No recorded bags yet. Click "Start Recording" to capture simulation telemetry.
                  </div>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', textAlign: 'left' }}>
                        <th style={{ padding: '6px 8px' }}>Bag Name</th>
                        <th style={{ padding: '6px 8px' }}>Created</th>
                        <th style={{ padding: '6px 8px' }}>Duration</th>
                        <th style={{ padding: '6px 8px' }}>Packets</th>
                        <th style={{ padding: '6px 8px' }}>Size</th>
                        <th style={{ padding: '6px 8px', textAlign: 'right' }}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {savedBags.map((b) => (
                        <tr key={b.bag_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                          <td style={{ padding: '6px 8px', fontWeight: 600, color: '#38bdf8' }}>{b.bag_name}</td>
                          <td style={{ padding: '6px 8px', color: 'var(--text-secondary)' }}>
                            {new Date(b.created_at * 1000).toLocaleTimeString()}
                          </td>
                          <td style={{ padding: '6px 8px', color: 'var(--accent-orange)' }}>
                            {b.wall_duration_sec}s
                          </td>
                          <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>
                            {b.total_messages}
                          </td>
                          <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>
                            {((b.file_size_bytes || 0) / 1024).toFixed(1)} KB
                          </td>
                          <td style={{ padding: '6px 8px', textAlign: 'right' }}>
                            <div style={{ display: 'inline-flex', gap: 6 }}>
                              <a
                                href={`/api/ros2/bag/${b.bag_id}`}
                                target="_blank"
                                rel="noreferrer"
                                className="rf-btn"
                                style={{ padding: '2px 6px', fontSize: 10, textDecoration: 'none' }}
                                title="Download JSON"
                              >
                                <Download size={11} />
                              </a>
                              <button
                                className="rf-btn rf-btn-danger"
                                onClick={() => handleDeleteBag(b.bag_id)}
                                style={{ padding: '2px 6px', fontSize: 10 }}
                                title="Delete Bag"
                              >
                                <Trash2 size={11} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </div>
        )}

        {/* --- 5. SWARM FLEET ORCHESTRATOR TAB --- */}
        {subTab === 'swarm' && (
          <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 12, height: '100%' }}>
            {/* Formation Deployment Controls */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Users size={14} color="#06b6d4" />
                <span style={{ fontWeight: 600 }}>Multi-Robot Swarm Orchestration</span>
              </div>

              {/* Formation Selection */}
              <div>
                <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                  Formation Pattern
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
                  {[
                    { id: 'v_shape', label: 'V-Shape' },
                    { id: 'line', label: 'Line' },
                    { id: 'column', label: 'Column' },
                    { id: 'circle', label: 'Circle' },
                    { id: 'diamond', label: 'Diamond' },
                  ].map((f) => (
                    <button
                      key={f.id}
                      onClick={() => setSwarmFormation(f.id as any)}
                      style={{
                        padding: '6px 4px',
                        borderRadius: 4,
                        border: swarmFormation === f.id ? '1px solid #06b6d4' : '1px solid var(--border-subtle)',
                        backgroundColor: swarmFormation === f.id ? 'rgba(6, 182, 212, 0.15)' : 'var(--bg-primary)',
                        color: swarmFormation === f.id ? '#06b6d4' : 'var(--text-secondary)',
                        fontSize: 10,
                        fontWeight: swarmFormation === f.id ? 600 : 400,
                        cursor: 'pointer'
                      }}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Agent Count & Spacing Sliders */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                    <span>Fleet Size (Leader + Followers)</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: '#06b6d4' }}>{swarmCount} agents</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="8"
                    step="1"
                    value={swarmCount}
                    onChange={(e) => setSwarmCount(parseInt(e.target.value, 10))}
                    style={{ width: '100%' }}
                  />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                    <span>Inter-Agent Spacing (meters)</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>{swarmSpacing.toFixed(2)}m</span>
                  </div>
                  <input
                    type="range"
                    min="0.6"
                    max="2.5"
                    step="0.1"
                    value={swarmSpacing}
                    onChange={(e) => setSwarmSpacing(parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>

              {/* Actions */}
              <div style={{ display: 'flex', gap: 8, marginTop: 'auto' }}>
                <button
                  className="rf-btn rf-btn-primary"
                  onClick={handleDeploySwarm}
                  style={{ flex: 1, padding: '8px 12px', justifyContent: 'center' }}
                >
                  <Users size={13} />
                  <span>Deploy Swarm Fleet</span>
                </button>
                <button
                  className="rf-btn rf-btn-danger"
                  onClick={handleClearSwarm}
                  style={{ padding: '8px 12px', justifyContent: 'center' }}
                  title="Clear Swarm Followers"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            </div>

            {/* Active Swarm Status Card */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 10
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: 11 }}>Active Fleet Configuration</span>
                <span style={{
                  fontSize: 10,
                  padding: '2px 8px',
                  borderRadius: 10,
                  backgroundColor: swarmStatus?.active ? 'rgba(34, 197, 94, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                  color: swarmStatus?.active ? 'var(--accent-green)' : 'var(--text-muted)',
                  fontWeight: 600
                }}>
                  {swarmStatus?.active ? 'FLEET ACTIVE' : 'NO SWARM'}
                </span>
              </div>

              {swarmStatus?.active && swarmStatus.config ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, fontSize: 11 }}>
                    <div style={{ padding: 8, backgroundColor: 'var(--bg-primary)', borderRadius: 4 }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block' }}>Formation</span>
                      <span style={{ fontWeight: 600, color: '#06b6d4', textTransform: 'uppercase' }}>
                        {swarmStatus.config.formation}
                      </span>
                    </div>
                    <div style={{ padding: 8, backgroundColor: 'var(--bg-primary)', borderRadius: 4 }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block' }}>Total Agents</span>
                      <span style={{ fontWeight: 600 }}>{swarmStatus.config.agent_count}</span>
                    </div>
                    <div style={{ padding: 8, backgroundColor: 'var(--bg-primary)', borderRadius: 4 }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block' }}>Spacing</span>
                      <span style={{ fontWeight: 600, color: 'var(--accent-orange)' }}>{swarmStatus.config.spacing}m</span>
                    </div>
                  </div>

                  <span style={{ fontSize: 11, fontWeight: 600, marginTop: 4 }}>Follower Agents in Scene</span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 180, overflowY: 'auto' }}>
                    {swarmStatus.config.followers.map((f, i) => (
                      <div
                        key={f.id}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: '6px 8px',
                          borderRadius: 4,
                          backgroundColor: 'var(--bg-primary)',
                          fontFamily: 'var(--font-mono)',
                          fontSize: 10
                        }}
                      >
                        <span style={{ color: '#06b6d4', fontWeight: 600 }}>{f.name}</span>
                        <span style={{ color: 'var(--text-muted)' }}>
                          [{f.position.map(n => n.toFixed(2)).join(', ')}]
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: 40, fontSize: 11 }}>
                  No active swarm deployed. Select formation and click "Deploy Swarm Fleet" to spawn coordinated agents.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
