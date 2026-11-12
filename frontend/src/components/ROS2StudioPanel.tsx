import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Radio, Play, Square, Download, Trash2, RefreshCw,
  Send, Database, Share2, Layers, Cpu, ArrowUp, ArrowDown,
  ArrowLeft, ArrowRight, StopCircle, CheckCircle2, AlertCircle,
  FileText, Users, Terminal, Navigation, Sliders, ChevronRight,
  FastForward, RotateCcw, Crosshair, HelpCircle, Check, PlayCircle
} from 'lucide-react';
import {
  fetchROS2Topics, fetchROS2TopicMessage, fetchROS2TF, sendROS2CmdVel,
  startROS2Bag, stopROS2Bag, fetchROS2BagStatus, fetchROS2Bags,
  deleteROS2Bag, deploySwarm, fetchSwarmStatus, clearSwarm,
  executeROS2CLI, sendROS2JointCommands, sendROS2GoalPose,
  replayROS2Bag, stopROS2BagReplay, fetchROS2BagReplayStatus, getROS2PackageExportUrl,
  ROS2TopicInfo, ROS2BagMetadata, ROS2BagStatus, SwarmStatusResponse, ROS2CLIResult
} from '../services/api';
import { SimulationState, Project } from '../types';

interface ROS2StudioPanelProps {
  onShowToast: (msg: string) => void;
  simState?: SimulationState;
  project?: Project | null;
}

export const ROS2StudioPanel: React.FC<ROS2StudioPanelProps> = ({
  onShowToast,
  simState,
  project = null
}) => {
  // --- Real rosbridge v2 server status (external ROS clients) ---
  const [bridge, setBridge] = useState<any>(null);
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const r = await fetch('/api/ros2/bridge/status');
        if (r.ok && alive) setBridge(await r.json());
      } catch { if (alive) setBridge(null); }
    };
    poll();
    const t = setInterval(poll, 2000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const [subTab, setSubTab] = useState<'cli' | 'arm' | 'nav2' | 'topics' | 'teleop' | 'tf' | 'bags' | 'swarm'>('cli');

  // --- CLI Terminal State ---
  const [cliInput, setCliInput] = useState('');
  const [cliHistory, setCliHistory] = useState<Array<{ cmd: string; exitCode: number; output: string; time: string }>>([
    {
      cmd: 'ros2 doctor',
      exitCode: 0,
      output: 'Built-in ROS 2 CLI emulator (answers from the live simulation; no ROS install needed).\nFor real ROS tools, connect to the rosbridge server: ws://localhost:9090 (status bar above).',
      time: new Date().toLocaleTimeString(),
    }
  ]);
  const [commandHistory, setCommandHistory] = useState<string[]>(['ros2 topic list -t', 'ros2 node info /roboforge_physics_bridge']);
  const [historyNavIdx, setHistoryNavIdx] = useState<number>(-1);
  const cliTerminalEndRef = useRef<HTMLDivElement>(null);

  // --- Arm Joint Controller State ---
  const [jointTargets, setJointTargets] = useState<Record<string, number>>({});
  const [autoStreamJoints, setAutoStreamJoints] = useState(false);

  // --- Nav2 Goal Pose State ---
  const [navX, setNavX] = useState<number>(2.0);
  const [navY, setNavY] = useState<number>(2.0);
  const [navYawDeg, setNavYawDeg] = useState<number>(0.0);
  const [navStatus, setNavStatus] = useState<string>('Ready to dispatch goal');

  // --- Topics State ---
  const [topics, setTopics] = useState<ROS2TopicInfo[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<string>('/joint_states');
  const [topicData, setTopicData] = useState<any>(null);
  const [loadingTopic, setLoadingTopic] = useState(false);
  const [autoEcho, setAutoEcho] = useState(true);
  const [customPubPayload, setCustomPubPayload] = useState<string>('{"data": [0.0, 0.5, -0.3, 0.2, 0.0, 0.0]}');

  // --- Teleop /cmd_vel State ---
  const [linearX, setLinearX] = useState<number>(0.0);
  const [angularZ, setAngularZ] = useState<number>(0.0);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(1.0);
  const [lastPublishedTwist, setLastPublishedTwist] = useState<any>(null);

  // --- TF2 Tree State ---
  const [tfData, setTfData] = useState<any>(null);
  const [loadingTf, setLoadingTf] = useState(false);

  // --- Bag Recording & 3D Replay State ---
  const [bagStatus, setBagStatus] = useState<ROS2BagStatus>({
    is_recording: false,
    current_bag_id: null,
    current_bag_name: null,
    message_count: 0
  });
  const [newBagName, setNewBagName] = useState('simulation_run_dataset');
  const [savedBags, setSavedBags] = useState<ROS2BagMetadata[]>([]);
  const [inspectingBag, setInspectingBag] = useState<any>(null);
  const [replaySpeed, setReplaySpeed] = useState<number>(1.0);
  const [replayStatus, setReplayStatus] = useState<{ is_playing: boolean; current_bag_id: string | null; playback_progress: number }>({
    is_playing: false,
    current_bag_id: null,
    playback_progress: 0.0
  });

  // --- Swarm Fleet State ---
  const [swarmFormation, setSwarmFormation] = useState<'v_shape' | 'line' | 'column' | 'circle' | 'diamond'>('v_shape');
  const [swarmCount, setSwarmCount] = useState<number>(4);
  const [swarmSpacing, setSwarmSpacing] = useState<number>(1.2);
  const [swarmStatus, setSwarmStatus] = useState<SwarmStatusResponse | null>(null);

  // --- Discover Arm Joints Dynamically (from REAL scene data only) ---
  const activeArmJoints = useMemo(() => {
    // 1. Real articulated robot joints from the project model (the authoritative source).
    if (project?.generic_robot?.joints && project.generic_robot.joints.length > 0) {
      return project.generic_robot.joints
        .filter((j) => j.type !== 'fixed')
        .map((j) => ({
          name: j.name,
          type: j.type,
          lower: j.limit?.lower ?? -3.14,
          upper: j.limit?.upper ?? 3.14,
          current: simState?.joint_positions?.[j.name] ?? 0.0,
        }));
    }
    // 2. No articulated robot in the scene -> NO joints. Never fabricate phantom joints
    //    (the old hardcoded 6-axis fallback showed joints even on an empty scene).
    return [] as Array<{ name: string; type: string; lower: number; upper: number; current: number }>;
  }, [project?.generic_robot?.joints, simState?.joint_positions]);

  const hasArm = activeArmJoints.length > 0;

  // Sync initial targets
  useEffect(() => {
    setJointTargets((prev) => {
      const next = { ...prev };
      activeArmJoints.forEach((j) => {
        if (next[j.name] === undefined) {
          next[j.name] = j.current;
        }
      });
      return next;
    });
  }, [activeArmJoints]);

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

  // Poll bag recording & replay status
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const stat = await fetchROS2BagStatus();
        setBagStatus(stat);
        const rep = await fetchROS2BagReplayStatus();
        setReplayStatus(rep);
      } catch (e) {
        // quiet
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Scroll terminal to bottom on new output
  useEffect(() => {
    if (subTab === 'cli') {
      cliTerminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [cliHistory, subTab]);

  // --- API Handlers ---
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

  const handleRunCLI = async (cmdToRun?: string) => {
    const cmd = (cmdToRun || cliInput).trim();
    if (!cmd) return;
    setCliInput('');
    setCommandHistory((prev) => [...prev, cmd]);
    setHistoryNavIdx(-1);
    try {
      const res = await executeROS2CLI(cmd);
      setCliHistory((prev) => [
        ...prev,
        {
          cmd,
          exitCode: res.exit_code,
          output: res.output,
          time: new Date().toLocaleTimeString(),
        }
      ]);
    } catch (err: any) {
      setCliHistory((prev) => [
        ...prev,
        {
          cmd,
          exitCode: 1,
          output: err.message || 'Error executing ROS 2 CLI command',
          time: new Date().toLocaleTimeString(),
        }
      ]);
    }
  };

  const handlePublishJointTargets = async (targetsToApply?: Record<string, number>) => {
    const targets = targetsToApply || jointTargets;
    try {
      await sendROS2JointCommands({ targets });
      onShowToast(`Published /joint_commands: ${Object.keys(targets).length} actuators targeted`);
    } catch (e: any) {
      onShowToast(`Failed to publish joint commands: ${e.message}`);
    }
  };

  const handleJointSliderChange = (jname: string, val: number) => {
    const updated = { ...jointTargets, [jname]: val };
    setJointTargets(updated);
    if (autoStreamJoints) {
      sendROS2JointCommands({ joint_name: jname, target: val }).catch(() => {});
    }
  };

  const handleApplyArmPreset = (presetType: 'home' | 'ready' | 'reach' | 'pick' | 'place_left' | 'place_right') => {
    const updated: Record<string, number> = {};
    activeArmJoints.forEach((j, idx) => {
      if (presetType === 'home') {
        updated[j.name] = 0.0;
      } else if (presetType === 'ready') {
        const presets = [0.0, -0.5, 1.2, -0.7, 0.0, 0.0];
        updated[j.name] = presets[idx % presets.length];
      } else if (presetType === 'reach') {
        const presets = [0.0, 0.4, 0.8, 0.2, 0.0, 0.04];
        updated[j.name] = presets[idx % presets.length];
      } else if (presetType === 'pick') {
        const presets = [0.0, 0.85, 1.15, -0.4, 0.0, 0.06];
        updated[j.name] = presets[idx % presets.length];
      } else if (presetType === 'place_left') {
        const presets = [1.57, 0.7, 0.9, -0.2, 0.0, 0.0];
        updated[j.name] = presets[idx % presets.length];
      } else if (presetType === 'place_right') {
        const presets = [-1.57, 0.7, 0.9, -0.2, 0.0, 0.0];
        updated[j.name] = presets[idx % presets.length];
      }
    });
    setJointTargets(updated);
    handlePublishJointTargets(updated);
  };

  const handleSyncJointsFromSim = () => {
    if (!simState?.joint_positions) return;
    const updated = { ...jointTargets };
    activeArmJoints.forEach((j) => {
      if (simState.joint_positions?.[j.name] !== undefined) {
        updated[j.name] = Number(simState.joint_positions[j.name].toFixed(3));
      }
    });
    setJointTargets(updated);
    onShowToast('Synced joint slider positions from live simulation physics');
  };

  const handleDispatchNavGoal = async (gx?: number, gy?: number, gyaw?: number) => {
    const x = gx ?? navX;
    const y = gy ?? navY;
    const yaw = gyaw ?? ((navYawDeg * Math.PI) / 180);
    try {
      setNavStatus(`Dispatching Nav2 /goal_pose -> (${x.toFixed(2)}, ${y.toFixed(2)})...`);
      const res = await sendROS2GoalPose({ x, y, yaw });
      setNavStatus(`Navigating: ${res.waypoint_count} collision-free waypoints planned`);
      onShowToast(`Dispatched Nav2 Goal (${x.toFixed(2)}, ${y.toFixed(2)}) via /goal_pose`);
    } catch (e: any) {
      setNavStatus(`Nav2 Dispatch Failed: ${e.message}`);
      onShowToast(`Nav2 dispatch failed: ${e.message}`);
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

  const handleReplayBag = async (bagId: string) => {
    try {
      await replayROS2Bag(bagId, replaySpeed);
      onShowToast(`Started 3D Viewport replay for bag ${bagId} at ${replaySpeed}x speed`);
      const rep = await fetchROS2BagReplayStatus();
      setReplayStatus(rep);
    } catch (e: any) {
      onShowToast(`Bag replay failed: ${e.message}`);
    }
  };

  const handleStopReplay = async () => {
    try {
      await stopROS2BagReplay();
      onShowToast('Stopped 3D Viewport bag replay');
      const rep = await fetchROS2BagReplayStatus();
      setReplayStatus(rep);
    } catch (e: any) {
      onShowToast(`Stop replay failed: ${e.message}`);
    }
  };

  const handleExportWorkspace = () => {
    const url = getROS2PackageExportUrl();
    const link = document.createElement('a');
    link.href = url;
    link.download = 'roboforge_bridge_ros2_ws.zip';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    onShowToast('Downloading colcon-ready ROS 2 package: roboforge_bridge_ros2_ws.zip');
  };

  const handleDeploySwarm = async () => {
    try {
      await deploySwarm(swarmFormation, swarmCount, swarmSpacing);
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

  const loadSwarmStatus = async () => {
    try {
      const res = await fetchSwarmStatus();
      setSwarmStatus(res);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', color: 'var(--text-primary)', fontSize: 12 }}>
      {/* Real rosbridge connection status */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', padding: '6px 12px',
        borderBottom: '1px solid var(--border-subtle)',
        background: bridge?.running ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)',
      }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700 }}>
          <span style={{ width: 8, height: 8, borderRadius: 4, background: bridge?.running ? '#3fb27f' : '#e5484d' }} />
          rosbridge {bridge?.running ? 'LIVE' : 'OFFLINE'}
        </span>
        {bridge?.running ? (
          <>
            <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: 4 }}>{bridge.url}</code>
            <button
              onClick={() => { navigator.clipboard?.writeText(bridge.url); onShowToast('Copied rosbridge URL'); }}
              style={{ fontSize: 10.5, padding: '2px 8px', borderRadius: 4, border: '1px solid var(--border-subtle)', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer' }}
            >Copy</button>
            <span style={{ color: 'var(--text-secondary)' }}>Clients: <b style={{ color: '#ecebe6' }}>{bridge.clients}</b></span>
            <span style={{ color: 'var(--text-secondary)' }}>/cmd_vel received: <b style={{ color: '#ecebe6' }}>{bridge.cmd_vel_received}</b></span>
            <span style={{ color: 'var(--text-secondary)' }}>Subscribed: <b style={{ color: '#ecebe6' }}>{(bridge.subscriptions || []).join(', ') || 'none'}</b></span>
            <span style={{ color: 'var(--text-muted)', fontSize: 10.5 }}>
              Foxglove: Open connection → Rosbridge → {bridge.url} · Python: roslibpy.Ros('localhost', {bridge.port})
            </span>
          </>
        ) : (
          <span style={{ color: '#f4a1a4' }}>{bridge?.error || 'Backend not reachable.'}</span>
        )}
      </div>
      {/* Top Banner / Sub-tab Selector */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '6px 10px',
        backgroundColor: 'var(--bg-panel)',
        borderBottom: '1px solid var(--border-subtle)',
        gap: 8,
        flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Radio size={14} color="#e8622c" />
          <span style={{ fontWeight: 600, color: '#e8622c' }}>ROS 2 Robotics Bridge</span>
          <span style={{
            fontSize: 10,
            padding: '1px 6px',
            borderRadius: 10,
            backgroundColor: 'rgba(232, 98, 44, 0.15)',
            color: '#e8622c',
            fontFamily: 'var(--font-mono)'
          }}>
            DOMAIN 0 • /roboforge_bridge_node
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

          {replayStatus.is_playing && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 10,
              padding: '1px 8px',
              borderRadius: 10,
              backgroundColor: 'rgba(16, 185, 129, 0.2)',
              color: 'var(--accent-green)',
              fontWeight: 600
            }}>
              <Play size={10} />
              <span>REPLAYING 3D ({replayStatus.playback_progress}%)</span>
              <button
                onClick={handleStopReplay}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--accent-red)',
                  cursor: 'pointer',
                  padding: 0,
                  display: 'flex',
                  alignItems: 'center'
                }}
                title="Stop Replay"
              >
                <StopCircle size={12} />
              </button>
            </div>
          )}
        </div>

        {/* Subtab Buttons & Export Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {[
            { id: 'cli', label: 'CLI Shell', icon: <Terminal size={12} /> },
            { id: 'arm', label: 'Arm & Joints', icon: <Sliders size={12} /> },
            { id: 'nav2', label: 'Nav2 Goal', icon: <Navigation size={12} /> },
            { id: 'topics', label: 'Topics & Echo', icon: <Radio size={12} /> },
            { id: 'teleop', label: '/cmd_vel Drive', icon: <Send size={12} /> },
            { id: 'tf', label: 'TF2 Tree', icon: <Share2 size={12} /> },
            { id: 'bags', label: 'Bags & Replay', icon: <Database size={12} /> },
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
                gap: 4,
                padding: '3px 8px',
                borderRadius: 4,
                border: 'none',
                backgroundColor: subTab === tab.id ? 'var(--accent-soft)' : 'var(--bg-primary)',
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

          {/* Export ROS 2 Workspace Button */}
          <button
            onClick={handleExportWorkspace}
            className="rf-btn"
            style={{
              padding: '3px 8px',
              backgroundColor: 'rgba(34, 197, 94, 0.15)',
              border: '1px solid rgba(34, 197, 94, 0.4)',
              color: 'var(--accent-green)',
              fontWeight: 600,
              fontSize: 11,
              marginLeft: 4
            }}
            title="Download colcon-ready ROS 2 Workspace Package (.zip)"
          >
            <Download size={12} />
            <span>Export Package</span>
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 8 }}>
        {/* ========================================== */}
        {/* --- 1. INTERACTIVE ROS 2 CLI SHELL ---     */}
        {/* ========================================== */}
        {subTab === 'cli' && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            backgroundColor: '#0c0d0e',
            borderRadius: 6,
            border: '1px solid #1b1e20',
            overflow: 'hidden'
          }}>
            {/* Quick Command Toolbar */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 10px',
              backgroundColor: '#101213',
              borderBottom: '1px solid #1b1e20',
              overflowX: 'auto'
            }}>
              <span style={{ fontSize: 10, color: '#a9aca7', fontWeight: 600, whiteSpace: 'nowrap' }}>QUICK COMMANDS:</span>
              {[
                'ros2 topic list -t',
                'ros2 topic echo /odom --once',
                'ros2 topic echo /joint_states --once',
                'ros2 node list',
                'ros2 node info /roboforge_physics_bridge',
                'ros2 service call /reset_simulation',
                'ros2 doctor',
                'ros2 bag list',
                'ros2 help'
              ].map((cmd) => (
                <button
                  key={cmd}
                  onClick={() => handleRunCLI(cmd)}
                  style={{
                    backgroundColor: '#1b1e20',
                    border: '1px solid #363b3e',
                    color: '#e8622c',
                    padding: '2px 8px',
                    borderRadius: 3,
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    cursor: 'pointer',
                    whiteSpace: 'nowrap'
                  }}
                >
                  {cmd}
                </button>
              ))}
              <button
                onClick={() => setCliHistory([])}
                style={{
                  backgroundColor: 'transparent',
                  border: '1px solid #6b6e6a',
                  color: '#a9aca7',
                  padding: '2px 6px',
                  borderRadius: 3,
                  fontSize: 10,
                  cursor: 'pointer',
                  marginLeft: 'auto'
                }}
                title="Clear Output"
              >
                Clear
              </button>
            </div>

            {/* Terminal History Output */}
            <div style={{
              flex: 1,
              overflowY: 'auto',
              padding: 12,
              fontFamily: 'Consolas, Monaco, "Courier New", monospace',
              fontSize: 11,
              lineHeight: 1.5,
              display: 'flex',
              flexDirection: 'column',
              gap: 10
            }}>
              {cliHistory.map((item, idx) => (
                <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ color: '#3fb27f' }}>user@roboforge:~$</span>
                    <span style={{ color: '#f4f3ef', fontWeight: 600 }}>{item.cmd}</span>
                    <span style={{ color: '#8a8d89', fontSize: 10, marginLeft: 'auto' }}>{item.time}</span>
                  </div>
                  <pre style={{
                    margin: 0,
                    padding: '6px 10px',
                    borderRadius: 4,
                    backgroundColor: item.exitCode === 0 ? 'rgba(30, 41, 59, 0.5)' : 'rgba(239, 68, 68, 0.1)',
                    color: item.exitCode === 0 ? '#d6d5cf' : '#f07378',
                    borderLeft: item.exitCode === 0 ? '3px solid #e8622c' : '3px solid #e5484d',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    fontFamily: 'inherit'
                  }}>
                    {item.output}
                  </pre>
                </div>
              ))}
              <div ref={cliTerminalEndRef} />
            </div>

            {/* Terminal Input Line */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              padding: '6px 10px',
              backgroundColor: '#0b1120',
              borderTop: '1px solid #1b1e20',
              gap: 8
            }}>
              <span style={{ color: '#3fb27f', fontFamily: 'monospace', fontWeight: 600 }}>user@roboforge:~$</span>
              <input
                type="text"
                value={cliInput}
                onChange={(e) => setCliInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleRunCLI();
                  } else if (e.key === 'ArrowUp') {
                    if (commandHistory.length > 0) {
                      const nextIdx = historyNavIdx === -1 ? commandHistory.length - 1 : Math.max(0, historyNavIdx - 1);
                      setHistoryNavIdx(nextIdx);
                      setCliInput(commandHistory[nextIdx]);
                    }
                  } else if (e.key === 'ArrowDown') {
                    if (historyNavIdx >= 0 && historyNavIdx < commandHistory.length - 1) {
                      const nextIdx = historyNavIdx + 1;
                      setHistoryNavIdx(nextIdx);
                      setCliInput(commandHistory[nextIdx]);
                    } else {
                      setHistoryNavIdx(-1);
                      setCliInput('');
                    }
                  }
                }}
                placeholder="Type any ROS 2 command (e.g. 'ros2 topic list', 'ros2 topic echo /odom', 'ros2 doctor')..."
                style={{
                  flex: 1,
                  backgroundColor: 'transparent',
                  border: 'none',
                  outline: 'none',
                  color: '#f4f3ef',
                  fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                  fontSize: 12
                }}
                autoFocus
              />
              <button
                onClick={() => handleRunCLI()}
                className="rf-btn rf-btn-primary"
                style={{ padding: '3px 10px', fontSize: 11 }}
              >
                Execute
              </button>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* --- 2. ARM & MULTI-DOF JOINT CONTROLLER TAB ---      */}
        {/* ==================================================== */}
        {subTab === 'arm' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 12, height: '100%' }}>
            {/* Left: Joint Sliders Panel */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
              overflowY: 'auto'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: 6 }}>
                <div>
                  <span style={{ fontWeight: 600, fontSize: 12 }}>ROS 2 Joint Actuation (/joint_commands)</span>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    std_msgs/msg/Float64MultiArray • Direct MuJoCo physics servo control
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <button
                    onClick={handleSyncJointsFromSim}
                    className="rf-btn"
                    style={{ padding: '3px 8px', fontSize: 10 }}
                    title="Read current joint positions from physics simulation"
                  >
                    <RotateCcw size={11} />
                    <span>Sync From Physics</span>
                  </button>

                  <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={autoStreamJoints}
                      onChange={(e) => setAutoStreamJoints(e.target.checked)}
                    />
                    <span>Live Drag Stream</span>
                  </label>
                </div>
              </div>

              {/* Individual Joint Sliders */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, flex: 1, overflowY: 'auto' }}>
                {!hasArm && (
                  <div style={{
                    padding: '20px 16px', textAlign: 'center', color: 'var(--text-muted)',
                    border: '1px dashed var(--border-subtle)', borderRadius: 6, fontSize: 12, lineHeight: 1.6,
                  }}>
                    No articulated robot in the scene.<br />
                    Import a URDF or ask the assistant to <strong>"spawn a 4 joint arm"</strong>,
                    then its real joints will appear here.
                  </div>
                )}
                {activeArmJoints.map((j) => {
                  const targetVal = jointTargets[j.name] ?? j.current;
                  const degVal = ((targetVal * 180) / Math.PI).toFixed(1);
                  const isPrismatic = j.type === 'prismatic';

                  return (
                    <div
                      key={j.name}
                      style={{
                        padding: '8px 10px',
                        backgroundColor: 'var(--bg-primary)',
                        borderRadius: 4,
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 6
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: 600, color: '#e8622c', fontFamily: 'var(--font-mono)' }}>
                          {j.name}
                        </span>
                        <div style={{ display: 'flex', gap: 6, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                          <span style={{ color: 'var(--accent-orange)' }}>
                            {targetVal.toFixed(3)} {isPrismatic ? 'm' : 'rad'}
                          </span>
                          {!isPrismatic && (
                            <span style={{ color: 'var(--text-muted)' }}>({degVal}°)</span>
                          )}
                          <span style={{ color: '#3fb27f', fontSize: 10 }}>
                            [Actual: {j.current.toFixed(3)}]
                          </span>
                        </div>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 9, color: 'var(--text-muted)', width: 36, textAlign: 'right' }}>
                          {j.lower.toFixed(2)}
                        </span>
                        <input
                          type="range"
                          min={j.lower}
                          max={j.upper}
                          step={isPrismatic ? 0.002 : 0.01}
                          value={targetVal}
                          onChange={(e) => handleJointSliderChange(j.name, parseFloat(e.target.value))}
                          style={{ flex: 1 }}
                        />
                        <span style={{ fontSize: 9, color: 'var(--text-muted)', width: 36 }}>
                          {j.upper.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Publish Action */}
              <div style={{ display: 'flex', gap: 8, paddingTop: 6, borderTop: '1px solid var(--border-subtle)' }}>
                <button
                  onClick={() => handlePublishJointTargets()}
                  className="rf-btn rf-btn-primary"
                  style={{ flex: 1, padding: '7px 12px', justifyContent: 'center' }}
                >
                  <Send size={13} />
                  <span>Publish to /joint_commands</span>
                </button>
                <button
                  onClick={() => handleApplyArmPreset('home')}
                  className="rf-btn"
                  style={{ padding: '7px 12px' }}
                >
                  Reset Home
                </button>
              </div>
            </div>

            {/* Right: Presets & Pose Library */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}>
              <div>
                <span style={{ fontWeight: 600, fontSize: 11 }}>MoveIt 2 Style Pose Presets</span>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                  One-click trajectory poses published directly to ROS 2
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                {[
                  { id: 'home', label: 'Home (0°)', desc: 'Zero all joints' },
                  { id: 'ready', label: 'Ready Pose', desc: 'Standard standby pose' },
                  { id: 'reach', label: 'Reach Forward', desc: 'Full frontal reach' },
                  { id: 'pick', label: 'Pick Ground', desc: 'Downward low grasp' },
                  { id: 'place_left', label: 'Place Left (+90°)', desc: 'Left bin deposit' },
                  { id: 'place_right', label: 'Place Right (-90°)', desc: 'Right bin deposit' },
                ].map((p) => (
                  <button
                    key={p.id}
                    onClick={() => handleApplyArmPreset(p.id as any)}
                    style={{
                      padding: '8px 10px',
                      backgroundColor: 'var(--bg-primary)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 4,
                      textAlign: 'left',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 2
                    }}
                  >
                    <span style={{ fontWeight: 600, color: '#e8622c', fontSize: 11 }}>{p.label}</span>
                    <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>{p.desc}</span>
                  </button>
                ))}
              </div>

              {/* Model & Actuator Telemetry */}
              <div style={{
                backgroundColor: 'var(--bg-primary)',
                borderRadius: 4,
                padding: 10,
                border: '1px solid var(--border-subtle)',
                marginTop: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
                fontSize: 10
              }}>
                <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>ACTIVE ACTUATOR TELEMETRY</span>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Robot Model:</span>
                  <span style={{ fontWeight: 600 }}>{project?.generic_robot?.name || project?.robot?.name || 'Articulated Manipulator'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Controlled DOF:</span>
                  <span style={{ fontWeight: 600, color: '#e8622c' }}>{activeArmJoints.length} Axis</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Feedback Topic:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>/joint_states (50Hz)</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* --- 3. NAV2 AUTONOMOUS GOAL DISPATCHER TAB ---       */}
        {/* ==================================================== */}
        {subTab === 'nav2' && (
          <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: 12, height: '100%' }}>
            {/* Target Coordinate Inputs */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}>
              <div>
                <span style={{ fontWeight: 600, fontSize: 12 }}>Nav2 Goal Dispatcher (/goal_pose)</span>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                  geometry_msgs/msg/PoseStamped • Plans collision-free trajectory in 3D physics
                </div>
              </div>

              {/* Coordinates */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                <div>
                  <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>X Position (m)</label>
                  <input
                    type="number"
                    step="0.2"
                    value={navX}
                    onChange={(e) => setNavX(parseFloat(e.target.value) || 0)}
                    style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border-subtle)', backgroundColor: 'var(--bg-primary)', color: '#fff' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>Y Position (m)</label>
                  <input
                    type="number"
                    step="0.2"
                    value={navY}
                    onChange={(e) => setNavY(parseFloat(e.target.value) || 0)}
                    style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border-subtle)', backgroundColor: 'var(--bg-primary)', color: '#fff' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>Yaw Angle (deg)</label>
                  <input
                    type="number"
                    step="15"
                    value={navYawDeg}
                    onChange={(e) => setNavYawDeg(parseFloat(e.target.value) || 0)}
                    style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border-subtle)', backgroundColor: 'var(--bg-primary)', color: '#fff' }}
                  />
                </div>
              </div>

              {/* Waypoint Presets */}
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: 6 }}>
                  PRESET FIELD WAYPOINTS:
                </span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                  {[
                    { name: 'Spawn Origin', x: 0.0, y: 0.0, yaw: 0 },
                    { name: 'Target Alpha', x: 2.0, y: 2.0, yaw: 45 },
                    { name: 'Target Beta', x: -2.0, y: 2.0, yaw: 135 },
                    { name: 'Cargo Bay', x: 2.5, y: -1.5, yaw: 0 },
                    { name: 'Perimeter Station', x: -2.5, y: -2.5, yaw: -90 },
                    { name: 'Far Corner', x: 3.5, y: 3.5, yaw: 90 },
                  ].map((wp) => (
                    <button
                      key={wp.name}
                      onClick={() => {
                        setNavX(wp.x);
                        setNavY(wp.y);
                        setNavYawDeg(wp.yaw);
                        handleDispatchNavGoal(wp.x, wp.y, (wp.yaw * Math.PI) / 180);
                      }}
                      style={{
                        padding: '6px 8px',
                        backgroundColor: 'var(--bg-primary)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: 4,
                        textAlign: 'left',
                        cursor: 'pointer',
                        display: 'flex',
                        flexDirection: 'column'
                      }}
                    >
                      <span style={{ fontWeight: 600, color: '#e8622c', fontSize: 10 }}>{wp.name}</span>
                      <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>({wp.x.toFixed(1)}, {wp.y.toFixed(1)})</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Action */}
              <button
                onClick={() => handleDispatchNavGoal()}
                className="rf-btn rf-btn-primary"
                style={{ padding: '8px 12px', justifyContent: 'center', marginTop: 'auto' }}
              >
                <Navigation size={13} />
                <span>Dispatch Goal (/goal_pose)</span>
              </button>
            </div>

            {/* Navigation Status & Telemetry Card */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}>
              <span style={{ fontWeight: 600, fontSize: 11 }}>Nav2 Autonomous Guidance Status</span>

              <div style={{
                padding: 10,
                backgroundColor: 'var(--bg-primary)',
                borderRadius: 4,
                border: '1px solid var(--border-subtle)',
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                color: '#e8622c'
              }}>
                {navStatus}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8, fontSize: 11 }}>
                <div style={{ padding: 8, backgroundColor: 'var(--bg-primary)', borderRadius: 4 }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block' }}>Current Odometry (X, Y)</span>
                  <span style={{ fontWeight: 600, color: '#f4f3ef' }}>
                    {simState?.robot ? `${simState.robot.position[0].toFixed(2)}m, ${simState.robot.position[1].toFixed(2)}m` : '0.00m, 0.00m'}
                  </span>
                </div>
                <div style={{ padding: 8, backgroundColor: 'var(--bg-primary)', borderRadius: 4 }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block' }}>Linear Velocity</span>
                  <span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>
                    {simState?.robot ? `${simState.robot.linear_speed.toFixed(2)} m/s` : '0.00 m/s'}
                  </span>
                </div>
              </div>

              {/* Serialized Nav2 message sample */}
              <div style={{ marginTop: 'auto' }}>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: 4 }}>
                  OUTGOING ROS 2 MESSAGE SCHEMA (PoseStamped):
                </span>
                <pre style={{
                  margin: 0,
                  padding: 8,
                  backgroundColor: '#0c0d0e',
                  borderRadius: 4,
                  fontSize: 10,
                  fontFamily: 'monospace',
                  color: '#a9aca7',
                  overflowX: 'auto'
                }}>
{`header: { frame_id: "map", stamp: { sec: ${Math.floor(simState?.sim_time || 0)} } }
pose:
  position: { x: ${navX.toFixed(2)}, y: ${navY.toFixed(2)}, z: 0.0 }
  orientation: { yaw_deg: ${navYawDeg.toFixed(1)}° }`}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* ========================================== */}
        {/* --- 4. TOPICS & LIVE ECHO TAB ---          */}
        {/* ========================================== */}
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
                        backgroundColor: isSelected ? 'rgba(232, 98, 44, 0.15)' : 'var(--bg-primary)',
                        border: isSelected ? '1px solid rgba(232, 98, 44, 0.4)' : '1px solid transparent',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontWeight: 600, color: isSelected ? '#e8622c' : 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {t.topic}
                        </span>
                        <span style={{
                          fontSize: 9,
                          padding: '0 4px',
                          borderRadius: 3,
                          backgroundColor: t.mode === 'publication' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(255, 255, 255, 0.057)',
                          color: t.mode === 'publication' ? 'var(--accent-green)' : '#a9aca7'
                        }}>
                          {t.mode === 'publication' ? 'PUB' : 'SUB'}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
                        <span>{t.type}</span>
                        <span>{typeof t.rate_hz === 'number' ? `${t.rate_hz} Hz` : t.rate_hz}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Topic Inspector & Manual Publisher */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 10,
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
              overflow: 'hidden'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontWeight: 600, color: '#e8622c', fontFamily: 'var(--font-mono)' }}>
                    {selectedTopic || 'No topic selected'}
                  </span>
                  {selectedTopic && (
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      ({topics.find(t => t.topic === selectedTopic)?.type || 'unknown type'})
                    </span>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={autoEcho}
                      onChange={(e) => setAutoEcho(e.target.checked)}
                    />
                    <span>Auto-Echo (1Hz)</span>
                  </label>
                  <button
                    onClick={() => echoTopic(selectedTopic)}
                    disabled={loadingTopic}
                    className="rf-btn"
                    style={{ padding: '2px 8px', fontSize: 10 }}
                  >
                    <RefreshCw size={10} className={loadingTopic ? 'animate-spin' : ''} />
                    <span>Echo Once</span>
                  </button>
                </div>
              </div>

              {/* Message Payload JSON Viewer */}
              <div style={{
                flex: 1,
                overflowY: 'auto',
                backgroundColor: '#0c0d0e',
                borderRadius: 4,
                border: '1px solid var(--border-subtle)',
                padding: 10,
                fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                fontSize: 11
              }}>
                {loadingTopic && !topicData ? (
                  <div style={{ color: 'var(--text-muted)' }}>Echoing message...</div>
                ) : topicData ? (
                  <pre style={{ margin: 0, color: '#d6d5cf', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {JSON.stringify(topicData, null, 2)}
                  </pre>
                ) : (
                  <div style={{ color: 'var(--text-muted)' }}>No message data received for this topic.</div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ========================================== */}
        {/* --- 5. /cmd_vel TELEOP DRIVER TAB ---      */}
        {/* ========================================== */}
        {subTab === 'teleop' && (
          <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 12, height: '100%' }}>
            {/* Control Pad */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}>
              <div>
                <span style={{ fontWeight: 600, fontSize: 12 }}>ROS 2 Twist Publisher (/cmd_vel)</span>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                  geometry_msgs/msg/Twist • Mobile base steering controller
                </div>
              </div>

              {/* Directional Pad */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 54px)',
                gridTemplateRows: 'repeat(3, 54px)',
                gap: 6,
                justifyContent: 'center',
                margin: '10px 0'
              }}>
                <div />
                <button
                  onClick={() => handlePublishCmdVel(1.0 * speedMultiplier, 0.0)}
                  className="rf-btn"
                  style={{ justifyContent: 'center' }}
                  title="Forward"
                >
                  <ArrowUp size={18} />
                </button>
                <div />
                <button
                  onClick={() => handlePublishCmdVel(0.0, 1.2 * speedMultiplier)}
                  className="rf-btn"
                  style={{ justifyContent: 'center' }}
                  title="Turn Left"
                >
                  <ArrowLeft size={18} />
                </button>
                <button
                  onClick={() => handlePublishCmdVel(0.0, 0.0)}
                  className="rf-btn rf-btn-danger"
                  style={{ justifyContent: 'center' }}
                  title="Stop"
                >
                  <StopCircle size={18} />
                </button>
                <button
                  onClick={() => handlePublishCmdVel(0.0, -1.2 * speedMultiplier)}
                  className="rf-btn"
                  style={{ justifyContent: 'center' }}
                  title="Turn Right"
                >
                  <ArrowRight size={18} />
                </button>
                <div />
                <button
                  onClick={() => handlePublishCmdVel(-1.0 * speedMultiplier, 0.0)}
                  className="rf-btn"
                  style={{ justifyContent: 'center' }}
                  title="Backward"
                >
                  <ArrowDown size={18} />
                </button>
                <div />
              </div>

              {/* Speed Multiplier */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                  <span>Velocity Scale:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>{speedMultiplier.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="3.0"
                  step="0.2"
                  value={speedMultiplier}
                  onChange={(e) => setSpeedMultiplier(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              {/* Keyboard Instructions */}
              <div style={{
                padding: 8,
                backgroundColor: 'var(--bg-primary)',
                borderRadius: 4,
                fontSize: 10,
                color: 'var(--text-muted)'
              }}>
                Tip: You can also drive using <kbd style={{ background: '#363b3e', padding: '1px 4px', borderRadius: 2 }}>W</kbd> <kbd style={{ background: '#363b3e', padding: '1px 4px', borderRadius: 2 }}>A</kbd> <kbd style={{ background: '#363b3e', padding: '1px 4px', borderRadius: 2 }}>S</kbd> <kbd style={{ background: '#363b3e', padding: '1px 4px', borderRadius: 2 }}>D</kbd> keys in the 3D viewport.
              </div>
            </div>

            {/* Published Twist Telemetry */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 10
            }}>
              <span style={{ fontWeight: 600, fontSize: 11 }}>Active /cmd_vel Telemetry</span>
              <div style={{
                flex: 1,
                backgroundColor: '#0c0d0e',
                borderRadius: 4,
                padding: 10,
                fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                fontSize: 11,
                color: '#d6d5cf'
              }}>
                <pre style={{ margin: 0 }}>
{`linear:
  x: ${linearX.toFixed(2)} m/s
  y: 0.00 m/s
  z: 0.00 m/s
angular:
  x: 0.00 rad/s
  y: 0.00 rad/s
  z: ${angularZ.toFixed(2)} rad/s`}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* ========================================== */}
        {/* --- 6. TF2 TRANSFORM TREE TAB ---          */}
        {/* ========================================== */}
        {subTab === 'tf' && (
          <div style={{
            backgroundColor: 'var(--bg-panel)',
            borderRadius: 6,
            border: '1px solid var(--border-subtle)',
            padding: 12,
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            gap: 10
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <span style={{ fontWeight: 600, fontSize: 12 }}>TF2 Coordinate Frame Hierarchy</span>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                  tf2_msgs/msg/TFMessage • Active kinematic transform tree
                </div>
              </div>
              <button onClick={loadTf} className="rf-btn" style={{ padding: '3px 8px' }}>
                <RefreshCw size={11} className={loadingTf ? 'animate-spin' : ''} />
                <span>Refresh Frames</span>
              </button>
            </div>

            <div style={{
              flex: 1,
              overflowY: 'auto',
              backgroundColor: '#0c0d0e',
              borderRadius: 4,
              padding: 10,
              fontFamily: 'Consolas, Monaco, "Courier New", monospace',
              fontSize: 11,
              color: '#d6d5cf'
            }}>
              {tfData?.transforms ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {tfData.transforms.map((tf: any, i: number) => (
                    <div
                      key={i}
                      style={{
                        padding: 8,
                        backgroundColor: '#101213',
                        borderRadius: 4,
                        border: '1px solid #1b1e20'
                      }}
                    >
                      <div style={{ color: '#e8622c', fontWeight: 600, marginBottom: 4 }}>
                        {tf.header?.frame_id} &rarr; {tf.child_frame_id}
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 10 }}>
                        <div>
                          <span style={{ color: 'var(--text-muted)' }}>Translation: </span>
                          <span>[{tf.transform?.translation?.x?.toFixed(3)}, {tf.transform?.translation?.y?.toFixed(3)}, {tf.transform?.translation?.z?.toFixed(3)}]</span>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-muted)' }}>Rotation (quat): </span>
                          <span>[{tf.transform?.rotation?.x?.toFixed(3)}, {tf.transform?.rotation?.y?.toFixed(3)}, {tf.transform?.rotation?.z?.toFixed(3)}, {tf.transform?.rotation?.w?.toFixed(3)}]</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)' }}>No TF data received.</div>
              )}
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* --- 7. BAG RECORDER & 3D VIEWPORT REPLAY TAB ---     */}
        {/* ==================================================== */}
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
              <div>
                <span style={{ fontWeight: 600, fontSize: 12 }}>ROS 2 Bag Recorder & Player</span>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                  rosbag2 format • Replay saved bags directly into 3D physics
                </div>
              </div>

              {/* Start / Stop Recording */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div>
                  <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>
                    Dataset / Bag Name
                  </label>
                  <input
                    type="text"
                    value={newBagName}
                    disabled={bagStatus.is_recording}
                    onChange={(e) => setNewBagName(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '5px 8px',
                      borderRadius: 4,
                      border: '1px solid var(--border-subtle)',
                      backgroundColor: 'var(--bg-primary)',
                      color: '#fff'
                    }}
                  />
                </div>

                {!bagStatus.is_recording ? (
                  <button
                    onClick={handleStartRecording}
                    className="rf-btn rf-btn-danger"
                    style={{ padding: '8px 12px', justifyContent: 'center' }}
                  >
                    <Square size={13} fill="currentColor" />
                    <span>Start Recording Telemetry</span>
                  </button>
                ) : (
                  <button
                    onClick={handleStopRecording}
                    className="rf-btn"
                    style={{
                      padding: '8px 12px',
                      justifyContent: 'center',
                      backgroundColor: 'var(--accent-orange)',
                      color: '#000',
                      fontWeight: 600
                    }}
                  >
                    <StopCircle size={14} />
                    <span>Stop Recording & Save Bag</span>
                  </button>
                )}
              </div>

              {/* Replay Speed Multiplier */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                  <span>3D Replay Speed:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: '#e8622c' }}>{replaySpeed}x</span>
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  {[0.5, 1.0, 2.0, 5.0].map((s) => (
                    <button
                      key={s}
                      onClick={() => setReplaySpeed(s)}
                      style={{
                        flex: 1,
                        padding: '4px 0',
                        borderRadius: 3,
                        border: replaySpeed === s ? '1px solid #e8622c' : '1px solid var(--border-subtle)',
                        backgroundColor: replaySpeed === s ? 'rgba(232, 98, 44, 0.15)' : 'var(--bg-primary)',
                        color: replaySpeed === s ? '#e8622c' : 'var(--text-secondary)',
                        fontSize: 10,
                        cursor: 'pointer'
                      }}
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Saved Bags Table */}
            <div style={{
              backgroundColor: 'var(--bg-panel)',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              padding: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
              overflow: 'hidden'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: 11 }}>Saved Bag Datasets ({savedBags.length})</span>
                <button onClick={loadBags} className="rf-btn" style={{ padding: '2px 6px' }}>
                  <RefreshCw size={11} />
                </button>
              </div>

              <div style={{ flex: 1, overflowY: 'auto' }}>
                {savedBags.length === 0 ? (
                  <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: 30, fontSize: 11 }}>
                    No recorded ROS 2 Bags yet. Click "Start Recording Telemetry" to capture a dataset.
                  </div>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', textAlign: 'left' }}>
                        <th style={{ padding: '4px 6px' }}>Bag Name</th>
                        <th style={{ padding: '4px 6px' }}>Messages</th>
                        <th style={{ padding: '4px 6px' }}>Duration</th>
                        <th style={{ padding: '4px 6px' }}>Format</th>
                        <th style={{ padding: '4px 6px', textAlign: 'right' }}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {savedBags.map((bag) => (
                        <tr key={bag.bag_id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                          <td style={{ padding: '6px', fontWeight: 600, color: '#e8622c' }}>
                            {bag.bag_name}
                          </td>
                          <td style={{ padding: '6px' }}>{bag.total_messages}</td>
                          <td style={{ padding: '6px' }}>{bag.sim_duration_sec?.toFixed(1)}s</td>
                          <td style={{ padding: '6px', color: 'var(--text-muted)' }}>{bag.storage_format}</td>
                          <td style={{ padding: '6px', textAlign: 'right' }}>
                            <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                              <button
                                onClick={() => handleReplayBag(bag.bag_id)}
                                className="rf-btn rf-btn-primary"
                                style={{ padding: '2px 8px', fontSize: 10 }}
                                title="Replay bag telemetry live into 3D simulation"
                              >
                                <Play size={10} />
                                <span>Replay in 3D</span>
                              </button>
                              <button
                                onClick={() => handleDeleteBag(bag.bag_id)}
                                style={{
                                  background: 'none',
                                  border: 'none',
                                  color: 'var(--accent-red)',
                                  cursor: 'pointer'
                                }}
                                title="Delete Bag"
                              >
                                <Trash2 size={12} />
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

        {/* ========================================== */}
        {/* --- 8. SWARM FLEET ORCHESTRATOR TAB ---   */}
        {/* ========================================== */}
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
                <Users size={14} color="#e8622c" />
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
                        border: swarmFormation === f.id ? '1px solid #e8622c' : '1px solid var(--border-subtle)',
                        backgroundColor: swarmFormation === f.id ? 'rgba(232, 98, 44, 0.15)' : 'var(--bg-primary)',
                        color: swarmFormation === f.id ? '#e8622c' : 'var(--text-secondary)',
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
                    <span style={{ fontFamily: 'var(--font-mono)', color: '#e8622c' }}>{swarmCount} agents</span>
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
                      <span style={{ fontWeight: 600, color: '#e8622c', textTransform: 'uppercase' }}>
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
                    {swarmStatus.config.followers.map((f) => (
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
                        <span style={{ color: '#e8622c', fontWeight: 600 }}>{f.name}</span>
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
