import React, { useState, useEffect } from 'react';
import {
  Play, Copy, Layers, Target, Clock, ShieldAlert,
  ArrowRight, CheckCircle2, XCircle, AlertCircle, RefreshCw, BarChart2,
  GitCompare, ExternalLink, Eye, Sparkles, BrainCircuit, Sliders, Plus, Settings2, Save, X,
  Bot, Award, Zap, Lightbulb, CheckSquare, StopCircle, CornerDownRight
} from 'lucide-react';
import {
  ExperimentConfig, ExperimentRunResult, RunSummary, RunComparison, Project,
  AvoidancePolicy, PolicyTrainingResult, RLTrainingConfig, RLTrainingStatus,
  RLModelMetadata, EvaluationSuiteResult, AIAssistantConfigResponse, HypothesisRefinementResult
} from '../types';
import {
  fetchScenarios, duplicateScenario, loadScenarioToScene,
  saveScenarioFromActive, runExperiment, runLiveExperiment, fetchRuns, fetchRun, compareRuns,
  saveScenario, updateScenario, trainAvoidancePolicy,
  startRLTraining, getRLTrainingStatus, cancelRLTraining, fetchRLModels,
  evaluateRLPolicy, executeRLPolicyLive, interpretAIPrompt, applyAIConfig, refineHypothesis
} from '../services/api';
import { AIAssistantChat } from './AIAssistantChat';

interface ExperimentPanelProps {
  currentProject: Project | null;
  onShowToast: (msg: string) => void;
  onProjectLoaded?: () => void;
  onSelectTarget?: (pos: [number, number, number] | null) => void;
  onSetTrail?: (trail: Array<[number, number, number]> | null) => void;
  activeLiveRun?: ExperimentRunResult | null;
  targetPosition?: [number, number, number] | null;
}

const defaultPolicy: AvoidancePolicy = {
  enabled: true,
  safety_margin: 0.75,
  repulsion_gain: 2.2,
  tangential_gain: 1.8,
  max_turn_rate: 3.2,
  slowdown_distance: 0.55,
};

export const ExperimentPanel: React.FC<ExperimentPanelProps> = ({
  currentProject,
  onShowToast,
  onProjectLoaded,
  onSelectTarget,
  onSetTrail,
  activeLiveRun,
  targetPosition,
}) => {
  const [scenarios, setScenarios] = useState<ExperimentConfig[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('');
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [activeRun, setActiveRun] = useState<ExperimentRunResult | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);

  // Dynamic Space Division / Priority Layout Mode
  const [layoutMode, setLayoutMode] = useState<'balanced' | 'metrics' | 'scenarios'>('balanced');

  // Policy Training & Tuning States
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingResult, setTrainingResult] = useState<PolicyTrainingResult | null>(null);
  const [showTrainingModal, setShowTrainingModal] = useState<boolean>(false);
  const [showPolicyTuner, setShowPolicyTuner] = useState<boolean>(false);
  const [policyParams, setPolicyParams] = useState<AvoidancePolicy>(defaultPolicy);

  // Custom Scenario Creation State
  const [showNewScenarioModal, setShowNewScenarioModal] = useState<boolean>(false);
  const [newScenarioForm, setNewScenarioForm] = useState({
    name: 'Obstacle Course Challenge',
    targetX: 2.5,
    targetY: 0.5,
    timeout: 10.0,
    seed: 42,
    safetyMargin: 0.85,
    repulsionGain: 2.5,
  });

  // Runner inputs
  const [customSeed, setCustomSeed] = useState<number>(42);
  const [customDuration, setCustomDuration] = useState<number>(8.0);

  // Comparison & Center View Modes
  const [compareRunAId, setCompareRunAId] = useState<string>('');
  const [compareRunBId, setCompareRunBId] = useState<string>('');
  const [comparisonResult, setComparisonResult] = useState<RunComparison | null>(null);
  const [viewMode, setViewMode] = useState<'metrics' | 'compare' | 'rl_training' | 'benchmark' | 'ai_assistant' | 'hypotheses'>('metrics');

  // RL Training State
  const [rlModels, setRlModels] = useState<RLModelMetadata[]>([]);
  const [activeRLJob, setActiveRLJob] = useState<RLTrainingStatus | null>(null);
  const [rlForm, setRlForm] = useState({
    modelName: 'PPO-DiffDrive-Agent',
    totalTimesteps: 5000,
    seed: 42,
    randomizeScenarios: true,
    learningRate: 0.0003,
    batchSize: 64,
    customRewardCode: `# Custom Python Reward Function (Executed in MuJoCo step loop)
def compute_reward(state, action, info):
    dist = info.get("distance_to_goal", 0.0)
    prev_dist = info.get("prev_distance", dist)
    angle_diff = info.get("angle_diff", 0.0)

    # 1. Proximity progress towards target beacon
    r_progress = 6.0 * (prev_dist - dist)

    # 2. Heading alignment
    r_heading = 0.3 * math.cos(angle_diff)

    # 3. Action effort penalty
    r_step = -0.02
    r_effort = -0.01 * float(action[0]**2 + action[1]**2)

    # 4. Terminal outcomes
    if info.get("is_collision"):
        return -50.0, True
    if info.get("is_success"):
        return 100.0, True

    return r_progress + r_heading + r_step + r_effort, False`,
  });

  // Benchmark Evaluation State
  const [benchmarkResult, setBenchmarkResult] = useState<EvaluationSuiteResult | null>(null);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [selectedEvalModelId, setSelectedEvalModelId] = useState<string>('');

  // AI Assistant State
  const [aiPrompt, setAiPrompt] = useState<string>('Create a tight slalom challenge with 3 tall cylinders and evaluate PPO vs APF');
  const [aiResponse, setAiResponse] = useState<AIAssistantConfigResponse | null>(null);
  const [isInterpreting, setIsInterpreting] = useState<boolean>(false);

  // Evidence-Based Hypothesis Refinement State
  const [hypoResult, setHypoResult] = useState<HypothesisRefinementResult | null>(null);
  const [isRefining, setIsRefining] = useState<boolean>(false);

  const loadScenariosList = async () => {
    try {
      const list = await fetchScenarios();
      setScenarios(list);
      if (list.length > 0 && !selectedScenarioId) {
        setSelectedScenarioId(list[0].id);
        setCustomSeed(list[0].seed);
        setCustomDuration(list[0].duration);
        setPolicyParams(list[0].task.policy || defaultPolicy);
        if (onSelectTarget) {
          onSelectTarget(list[0].task.target_position);
        }
      }
    } catch (e: any) {
      console.error(e);
    }
  };

  const loadRunsList = async () => {
    try {
      const list = await fetchRuns();
      setRuns(list);
      if (list.length >= 2 && !compareRunAId && !compareRunBId) {
        setCompareRunAId(list[1].run_id);
        setCompareRunBId(list[0].run_id);
      }
    } catch (e: any) {
      console.error(e);
    }
  };

  const loadRLModelsList = async () => {
    try {
      const models = await fetchRLModels();
      setRlModels(models);
      if (models.length > 0 && !selectedEvalModelId) {
        setSelectedEvalModelId(models[0].model_id);
      }
    } catch (e: any) {
      console.error('Failed to load RL models:', e);
    }
  };

  useEffect(() => {
    loadScenariosList();
    loadRunsList();
    loadRLModelsList();
  }, []);

  // Poll RL training job status
  useEffect(() => {
    if (!activeRLJob || activeRLJob.status === 'completed' || activeRLJob.status === 'cancelled' || activeRLJob.status === 'failed') {
      return;
    }
    const interval = setInterval(async () => {
      try {
        const updated = await getRLTrainingStatus(activeRLJob.job_id);
        setActiveRLJob(updated);
        if (updated.status === 'completed') {
          onShowToast(`RL Training complete: ${updated.model_name} saved!`);
          loadRLModelsList();
        } else if (updated.status === 'failed') {
          onShowToast(`RL Training failed: ${updated.error || 'Unknown error'}`);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 800);
    return () => clearInterval(interval);
  }, [activeRLJob]);

  const handleStartRLTraining = async () => {
    try {
      onShowToast(`Launching background PPO training (${rlForm.totalTimesteps} steps)...`);
      const status = await startRLTraining({
        model_name: rlForm.modelName,
        total_timesteps: rlForm.totalTimesteps,
        seed: rlForm.seed,
        randomize_scenarios: rlForm.randomizeScenarios,
        target_position: targetPosition || [2.5, 0.5, 0.25],
        scenario_id: selectedScenarioId || undefined,
        custom_reward_code: rlForm.customRewardCode,
        hyperparameters: {
          learning_rate: rlForm.learningRate,
          batch_size: rlForm.batchSize,
        },
      });
      setActiveRLJob(status);
      setViewMode('rl_training');
    } catch (e: any) {
      onShowToast(`Failed to start RL training: ${e.message}`);
    }
  };

  const handleCancelRLTraining = async () => {
    if (!activeRLJob) return;
    try {
      await cancelRLTraining(activeRLJob.job_id);
      onShowToast('RL training cancellation requested.');
    } catch (e: any) {
      onShowToast(`Cancellation error: ${e.message}`);
    }
  };

  const handleExecuteRLModelLive = async (modelId: string) => {
    try {
      onShowToast(`Loading PPO policy '${modelId}' and launching live 3D run...`);
      await executeRLPolicyLive(modelId, selectedScenarioId, targetPosition || undefined, 14.0);
      if (onProjectLoaded) onProjectLoaded();
    } catch (e: any) {
      onShowToast(`Live execution failed: ${e.message}`);
    }
  };

  const handleRunBenchmarkSuite = async (modelId?: string) => {
    const targetModel = modelId || selectedEvalModelId || (rlModels[0]?.model_id);
    if (!targetModel) {
      onShowToast('Please select or train an RL model first to evaluate.');
      return;
    }
    setIsEvaluating(true);
    onShowToast(`Running benchmark evaluation suite on held-out layouts...`);
    try {
      const res = await evaluateRLPolicy(targetModel);
      setBenchmarkResult(res);
      setViewMode('benchmark');
      onShowToast(`Evaluation complete: ${res.verdict}`);
    } catch (e: any) {
      onShowToast(`Evaluation failed: ${e.message}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleInterpretAIPrompt = async () => {
    if (!aiPrompt.trim()) return;
    setIsInterpreting(true);
    onShowToast('AI Assistant: Synthesizing and validating robotics experiment...');
    try {
      const res = await interpretAIPrompt(aiPrompt, selectedScenarioId);
      setAiResponse(res);
      onShowToast(`Generated: ${res.interpreted_goal}`);
    } catch (e: any) {
      onShowToast(`AI interpretation failed: ${e.message}`);
    } finally {
      setIsInterpreting(false);
    }
  };

  const handleApplyAIConfig = async () => {
    if (!aiResponse) return;
    try {
      onShowToast('Applying validated AI configuration to 3D simulation scene...');
      await applyAIConfig(aiResponse.suggested_configuration);
      await loadScenariosList();
      if (onProjectLoaded) onProjectLoaded();
      if (aiResponse.suggested_configuration?.task?.target_position && onSelectTarget) {
        onSelectTarget(aiResponse.suggested_configuration.task.target_position);
      }
      onShowToast('Configuration applied safely. Ready to run in 3D viewport!');
    } catch (e: any) {
      onShowToast(`Failed to apply configuration: ${e.message}`);
    }
  };

  const handleRunHypothesisRefinement = async () => {
    setIsRefining(true);
    onShowToast(`Formulating empirical hypothesis and executing controlled rerun...`);
    try {
      const res = await refineHypothesis(selectedScenarioId || undefined, 'collision_rate');
      setHypoResult(res);
      setViewMode('hypotheses');
      onShowToast(`Hypothesis ${res.outcome_verdict}: ${res.empirical_evidence}`);
      await loadScenariosList();
      await loadRunsList();
    } catch (e: any) {
      onShowToast(`Hypothesis refinement error: ${e.message}`);
    } finally {
      setIsRefining(false);
    }
  };

  // Sync active live experiment completion
  useEffect(() => {
    if (activeLiveRun) {
      setActiveRun(activeLiveRun);
      setViewMode('metrics');
      setIsRunning(false);
      loadRunsList();
      if (onSetTrail && activeLiveRun.measurements) {
        onSetTrail(activeLiveRun.measurements.map((m) => m.position));
      }
    }
  }, [activeLiveRun]);

  const handleScenarioChange = (id: string) => {
    setSelectedScenarioId(id);
    const scen = scenarios.find((s) => s.id === id);
    if (scen) {
      setCustomSeed(scen.seed);
      setCustomDuration(scen.duration);
      setPolicyParams(scen.task.policy || defaultPolicy);
      if (onSelectTarget) {
        onSelectTarget(scen.task.target_position);
      }
      if (onSetTrail) {
        onSetTrail(null);
      }
    }
  };

  const handleRunLive = async () => {
    if (!selectedScenarioId) return;
    setIsRunning(true);
    try {
      const scen = scenarios.find((s) => s.id === selectedScenarioId);
      if (scen && onSelectTarget) {
        onSelectTarget(scen.task.target_position);
      }
      if (onSetTrail) {
        onSetTrail(null);
      }
      onShowToast(`Starting live 3D experiment with seed ${customSeed}...`);
      await runLiveExperiment({
        scenario_id: selectedScenarioId,
        seed: customSeed,
        duration: customDuration,
      });
      if (onProjectLoaded) onProjectLoaded();
    } catch (e: any) {
      onShowToast(`Failed to start live experiment: ${e.message}`);
      setIsRunning(false);
    }
  };

  const handleTrainPolicy = async () => {
    if (!selectedScenarioId) return;
    setIsTraining(true);
    onShowToast('Training avoidance policy across parameter candidates...');
    try {
      const res = await trainAvoidancePolicy(selectedScenarioId);
      setTrainingResult(res);
      setShowTrainingModal(true);
      onShowToast(res.message);
      await loadScenariosList();
      setPolicyParams(res.best_policy);
    } catch (e: any) {
      onShowToast(`Policy training failed: ${e.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleSavePolicyTuning = async () => {
    const scen = scenarios.find((s) => s.id === selectedScenarioId);
    if (!scen) return;
    try {
      const updated: ExperimentConfig = {
        ...scen,
        task: {
          ...scen.task,
          policy: policyParams,
        },
      };
      await updateScenario(scen.id, updated);
      onShowToast('Obstacle avoidance policy updated for scenario');
      await loadScenariosList();
    } catch (e: any) {
      onShowToast(`Failed to update policy: ${e.message}`);
    }
  };

  const handleCreateCustomScenario = async () => {
    try {
      const newScen: ExperimentConfig = {
        schema_version: '1.0.0',
        id: `scen_${Date.now().toString(36)}`,
        name: newScenarioForm.name || 'Custom Arena Challenge',
        description: `Custom scenario targeting [${newScenarioForm.targetX}, ${newScenarioForm.targetY}]`,
        seed: newScenarioForm.seed,
        robot: currentProject?.robot || {
          id: 'robot_default',
          name: 'ForgeBot-DD1',
          initial_position: [0, 0, 0.07],
          initial_yaw: 0,
          chassis_size: [0.3, 0.2, 0.1],
          wheel_radius: 0.04,
          wheel_width: 0.02,
          wheel_base: 0.24,
          caster_radius: 0.02,
          chassis_mass: 4.0,
          wheel_mass: 0.3,
          max_linear_velocity: 1.5,
          max_angular_velocity: 3.0,
        },
        environment: currentProject?.objects || [],
        settings: currentProject?.settings || {
          timestep: 0.005,
          speed_multiplier: 1.0,
          gravity: [0, 0, -9.81],
        },
        task: {
          target_position: [newScenarioForm.targetX, newScenarioForm.targetY, 0.25],
          distance_threshold: 0.25,
          max_allowed_collisions: 0,
          timeout: newScenarioForm.timeout,
          control_mode: 'goal_seeking',
          policy: {
            enabled: true,
            safety_margin: newScenarioForm.safetyMargin,
            repulsion_gain: newScenarioForm.repulsionGain,
            tangential_gain: 2.0,
            max_turn_rate: 3.5,
            slowdown_distance: 0.6,
          },
        },
        duration: newScenarioForm.timeout,
        sample_rate_hz: 20.0,
        created_at: Date.now() / 1000,
      };
      await saveScenario(newScen);
      onShowToast(`Created scenario '${newScen.name}'`);
      setShowNewScenarioModal(false);
      await loadScenariosList();
      setSelectedScenarioId(newScen.id);
      if (onSelectTarget) onSelectTarget(newScen.task.target_position);
    } catch (e: any) {
      onShowToast(`Failed to create scenario: ${e.message}`);
    }
  };

  const handleRunExperiment = async () => {
    if (!selectedScenarioId) return;
    setIsRunning(true);
    try {
      onShowToast(`Running headless experiment with seed ${customSeed}...`);
      const result = await runExperiment({
        scenario_id: selectedScenarioId,
        seed: customSeed,
        duration: customDuration,
      });
      setActiveRun(result);
      setViewMode('metrics');
      if (onSetTrail && result.measurements) {
        onSetTrail(result.measurements.map((m) => m.position));
      }
      onShowToast(`Experiment completed: ${result.metrics.status_reason}`);
      await loadRunsList();
    } catch (e: any) {
      onShowToast(`Run failed: ${e.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  const handleDuplicateScenario = async () => {
    if (!selectedScenarioId) return;
    try {
      const dup = await duplicateScenario(selectedScenarioId);
      onShowToast(`Duplicated as '${dup.name}'`);
      await loadScenariosList();
      setSelectedScenarioId(dup.id);
    } catch (e: any) {
      onShowToast(`Failed to duplicate: ${e.message}`);
    }
  };

  const handleLoadToScene = async () => {
    if (!selectedScenarioId) return;
    try {
      await loadScenarioToScene(selectedScenarioId);
      onShowToast('Scenario loaded into active scene');
      if (onProjectLoaded) onProjectLoaded();
    } catch (e: any) {
      onShowToast(`Failed to load scenario: ${e.message}`);
    }
  };

  const handleSaveActiveAsScenario = async () => {
    if (!currentProject) return;
    const name = window.prompt('Enter Scenario Name:', `${currentProject.name} Scenario`);
    if (!name) return;
    try {
      const scen = await saveScenarioFromActive({
        name,
        seed: customSeed,
        timeout: customDuration,
      });
      onShowToast(`Created scenario '${scen.name}'`);
      await loadScenariosList();
      setSelectedScenarioId(scen.id);
    } catch (e: any) {
      onShowToast(`Failed to save: ${e.message}`);
    }
  };

  const handleCompareRuns = async () => {
    if (!compareRunAId || !compareRunBId) {
      onShowToast('Select two runs to compare');
      return;
    }
    try {
      const comp = await compareRuns(compareRunAId, compareRunBId);
      setComparisonResult(comp);
      setViewMode('compare');
      onShowToast('Runs compared successfully');
    } catch (e: any) {
      onShowToast(`Comparison failed: ${e.message}`);
    }
  };

  const selectedScenario = scenarios.find((s) => s.id === selectedScenarioId);

  // SVG Plots for Active Run
  const renderDistancePlot = (measurements: any[]) => {
    if (!measurements || measurements.length === 0) return null;
    const target = selectedScenario?.task.target_position || [2.0, 0.0, 0.25];
    const targetX = target[0];
    const targetY = target[1];

    const distPoints = measurements.map((m) => {
      const dx = m.position[0] - targetX;
      const dy = m.position[1] - targetY;
      return { time: m.sim_time, dist: Math.hypot(dx, dy) };
    });

    const maxDist = Math.max(3.0, ...distPoints.map((p) => p.dist));
    const width = 240;
    const height = 75;
    const pad = 12;

    const pointsStr = distPoints
      .map((p, idx) => {
        const x = pad + (idx / Math.max(1, distPoints.length - 1)) * (width - pad * 2);
        const y = height - pad - (p.dist / maxDist) * (height - pad * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

    return (
      <svg width={width} height={height} style={{ background: 'var(--bg-canvas)', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
        {/* Threshold line */}
        <line
          x1={pad}
          y1={height - pad - (0.25 / maxDist) * (height - pad * 2)}
          x2={width - pad}
          y2={height - pad - (0.25 / maxDist) * (height - pad * 2)}
          stroke="var(--accent-green)"
          strokeDasharray="3 3"
          strokeWidth="1"
        />
        <polyline
          fill="none"
          stroke="var(--accent-blue)"
          strokeWidth="1.8"
          points={pointsStr}
        />
        <text x={pad + 2} y={pad + 2} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)">
          Target Dist (m)
        </text>
        <text x={width - pad - 20} y={height - pad - 2} fill="var(--text-muted)" fontSize="8" textAnchor="end" fontFamily="var(--font-mono)">
          {distPoints[distPoints.length - 1]?.dist.toFixed(2)}m
        </text>
      </svg>
    );
  };

  const renderSpeedPlot = (measurements: any[]) => {
    if (!measurements || measurements.length === 0) return null;
    const maxSpeed = Math.max(1.8, ...measurements.map((m) => m.linear_speed));
    const width = 240;
    const height = 75;
    const pad = 12;

    const pointsStr = measurements
      .map((m, idx) => {
        const x = pad + (idx / Math.max(1, measurements.length - 1)) * (width - pad * 2);
        const y = height - pad - (m.linear_speed / maxSpeed) * (height - pad * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

    return (
      <svg width={width} height={height} style={{ background: 'var(--bg-canvas)', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
        <polyline
          fill="none"
          stroke="var(--accent-orange)"
          strokeWidth="1.8"
          points={pointsStr}
        />
        <text x={pad + 2} y={pad + 2} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)">
          Speed (m/s)
        </text>
        <text x={width - pad - 2} y={height - pad - 2} fill="var(--text-muted)" fontSize="8" textAnchor="end" fontFamily="var(--font-mono)">
          Max: {maxSpeed.toFixed(2)}
        </text>
      </svg>
    );
  };

  const renderRLTrainingView = () => {
    const isJobActive = Boolean(activeRLJob && (activeRLJob.status === 'training' || activeRLJob.status === 'queued'));
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Header Banner */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg-card)', padding: '8px 12px', borderRadius: 6 }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
              <BrainCircuit size={16} color="var(--accent-purple, #a855f7)" />
              <span>Real PPO Reinforcement Learning Manager</span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
              Authentic Gymnasium + Stable-Baselines3 PPO worker interacting directly with MuJoCo differential-drive physics.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <button className="rf-btn" onClick={loadRLModelsList} style={{ fontSize: 10, padding: '3px 8px' }}>
              <RefreshCw size={11} />
              <span>Refresh Models</span>
            </button>
          </div>
        </div>

        {/* Active Job Progress Banner */}
        {activeRLJob && (
          <div style={{ background: 'var(--bg-card)', padding: 10, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                {activeRLJob.status === 'training' ? (
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-blue)', animation: 'pulse 1.5s infinite' }} />
                ) : activeRLJob.status === 'completed' ? (
                  <CheckCircle2 size={14} color="var(--accent-green)" />
                ) : activeRLJob.status === 'cancelled' ? (
                  <AlertCircle size={14} color="var(--accent-yellow)" />
                ) : (
                  <XCircle size={14} color="var(--accent-red)" />
                )}
                <span style={{ fontWeight: 600, fontSize: 11 }}>
                  Job: {activeRLJob.model_name}
                </span>
                <span style={{ fontSize: 9, padding: '1px 5px', borderRadius: 3, background: 'var(--bg-canvas)', color: 'var(--text-muted)' }}>
                  {activeRLJob.status.toUpperCase()}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                {isJobActive ? (
                  <button
                    className="rf-btn"
                    onClick={handleCancelRLTraining}
                    style={{ fontSize: 10, padding: '2px 8px', color: 'var(--accent-red)' }}
                  >
                    <StopCircle size={11} />
                    <span>Cancel Training</span>
                  </button>
                ) : activeRLJob.status === 'completed' ? (
                  <>
                    <button
                      className="rf-btn rf-btn-primary"
                      onClick={() => handleExecuteRLModelLive(activeRLJob.job_id)}
                      style={{ fontSize: 10, padding: '2px 8px' }}
                    >
                      <Play size={11} />
                      <span>Run Live in 3D</span>
                    </button>
                    <button
                      className="rf-btn"
                      onClick={() => handleRunBenchmarkSuite(activeRLJob.job_id)}
                      style={{ fontSize: 10, padding: '2px 8px' }}
                    >
                      <Award size={11} />
                      <span>Benchmark</span>
                    </button>
                  </>
                ) : null}
              </div>
            </div>

            {/* Progress Bar */}
            <div style={{ marginBottom: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>
                <span>Timesteps Progress</span>
                <span>
                  {activeRLJob.metrics?.timestep?.toLocaleString() || 0} / {activeRLJob.metrics?.total_timesteps?.toLocaleString() || 5000} (
                  {activeRLJob.metrics?.percent_complete?.toFixed(1) || 0}%)
                </span>
              </div>
              <div style={{ width: '100%', height: 6, background: 'var(--bg-canvas)', borderRadius: 3, overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${Math.min(100, Math.max(0, activeRLJob.metrics?.percent_complete || 0))}%`,
                    height: '100%',
                    background: activeRLJob.status === 'completed' ? 'var(--accent-green)' : 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
            </div>

            {/* Metrics Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6, fontSize: 10 }}>
              <div style={{ background: 'var(--bg-canvas)', padding: '4px 6px', borderRadius: 4 }}>
                <div style={{ color: 'var(--text-muted)' }}>Mean Reward</div>
                <div style={{ fontWeight: 600, color: (activeRLJob.metrics?.mean_reward || 0) > 0 ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                  {activeRLJob.metrics?.mean_reward !== undefined ? activeRLJob.metrics.mean_reward.toFixed(2) : '0.00'}
                </div>
              </div>
              <div style={{ background: 'var(--bg-canvas)', padding: '4px 6px', borderRadius: 4 }}>
                <div style={{ color: 'var(--text-muted)' }}>Success Rate</div>
                <div style={{ fontWeight: 600, color: 'var(--accent-green)' }}>
                  {activeRLJob.metrics?.success_rate !== undefined ? `${(activeRLJob.metrics.success_rate * 100).toFixed(0)}%` : '0%'}
                </div>
              </div>
              <div style={{ background: 'var(--bg-canvas)', padding: '4px 6px', borderRadius: 4 }}>
                <div style={{ color: 'var(--text-muted)' }}>Collision Rate</div>
                <div style={{ fontWeight: 600, color: (activeRLJob.metrics?.collision_rate || 0) > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                  {activeRLJob.metrics?.collision_rate !== undefined ? `${(activeRLJob.metrics.collision_rate * 100).toFixed(0)}%` : '0%'}
                </div>
              </div>
              <div style={{ background: 'var(--bg-canvas)', padding: '4px 6px', borderRadius: 4 }}>
                <div style={{ color: 'var(--text-muted)' }}>Physics FPS</div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                  {activeRLJob.metrics?.fps ? Math.round(activeRLJob.metrics.fps) : 0} fps
                </div>
              </div>
              <div style={{ background: 'var(--bg-canvas)', padding: '4px 6px', borderRadius: 4 }}>
                <div style={{ color: 'var(--text-muted)' }}>Elapsed Time</div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                  {activeRLJob.metrics?.elapsed_seconds ? `${activeRLJob.metrics.elapsed_seconds.toFixed(1)}s` : '0.0s'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Training Configuration Panel */}
        <div style={{ background: 'var(--bg-card)', padding: 10, borderRadius: 6, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)' }}>Configure New PPO Training Run</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
            <div>
              <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Model Name:</label>
              <input
                type="text"
                className="rf-input"
                value={rlForm.modelName}
                onChange={(e) => setRlForm({ ...rlForm, modelName: e.target.value })}
                style={{ width: '100%', padding: '4px 6px', fontSize: 11 }}
              />
            </div>
            <div>
              <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Total Timesteps:</label>
              <input
                type="number"
                step="1000"
                className="rf-input"
                value={rlForm.totalTimesteps}
                onChange={(e) => setRlForm({ ...rlForm, totalTimesteps: parseInt(e.target.value) || 2000 })}
                style={{ width: '100%', padding: '4px 6px', fontSize: 11 }}
              />
            </div>
            <div>
              <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Random Seed:</label>
              <input
                type="number"
                className="rf-input"
                value={rlForm.seed}
                onChange={(e) => setRlForm({ ...rlForm, seed: parseInt(e.target.value) || 42 })}
                style={{ width: '100%', padding: '4px 6px', fontSize: 11 }}
              />
            </div>
          </div>

          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Quick Timestep Presets:</span>
            {[2000, 5000, 15000, 30000].map((steps) => (
              <button
                key={steps}
                type="button"
                className="rf-btn"
                onClick={() => setRlForm({ ...rlForm, totalTimesteps: steps })}
                style={{ fontSize: 9, padding: '2px 6px' }}
              >
                {steps.toLocaleString()} steps
              </button>
            ))}
          </div>

          {/* Custom Python Reward Function Script Editor (Replacing Hardcoded Values) */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 4 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>Custom Python Reward Script (Executed per Physics Step)</span>
              </div>
              <div style={{ display: 'flex', gap: 4 }}>
                <button
                  type="button"
                  className="rf-btn"
                  onClick={() => setRlForm({
                    ...rlForm,
                    customRewardCode: `# Preset: Dense Progress + Heading + Goal\ndef compute_reward(state, action, info):\n    dist = info.get("distance_to_goal", 0.0)\n    prev_dist = info.get("prev_distance", dist)\n    angle_diff = info.get("angle_diff", 0.0)\n    r_progress = 6.0 * (prev_dist - dist)\n    r_heading = 0.3 * math.cos(angle_diff)\n    r_step = -0.02\n    if info.get("is_collision"):\n        return -50.0, True\n    if info.get("is_success"):\n        return 100.0, True\n    return r_progress + r_heading + r_step, False`
                  })}
                  style={{ fontSize: 9, padding: '2px 6px' }}
                >
                  Dense Progress
                </button>
                <button
                  type="button"
                  className="rf-btn"
                  onClick={() => setRlForm({
                    ...rlForm,
                    customRewardCode: `# Preset: Sparse Goal-Seeking with Step Penalty\ndef compute_reward(state, action, info):\n    if info.get("is_collision"):\n        return -100.0, True\n    if info.get("is_success"):\n        return 500.0, True\n    return -0.1, False`
                  })}
                  style={{ fontSize: 9, padding: '2px 6px' }}
                >
                  Sparse Goal
                </button>
                <button
                  type="button"
                  className="rf-btn"
                  onClick={() => setRlForm({
                    ...rlForm,
                    customRewardCode: `# Preset: High-Safety Obstacle Repulsion\ndef compute_reward(state, action, info):\n    dist = info.get("distance_to_goal", 0.0)\n    prev_dist = info.get("prev_distance", dist)\n    r_progress = 5.0 * (prev_dist - dist)\n    if info.get("is_collision"):\n        return -200.0, True\n    if info.get("is_success"):\n        return 150.0, True\n    effort_penalty = -0.05 * float(action[1]**2)\n    return r_progress + effort_penalty - 0.02, False`
                  })}
                  style={{ fontSize: 9, padding: '2px 6px' }}
                >
                  High Safety
                </button>
              </div>
            </div>

            <textarea
              className="rf-input"
              value={rlForm.customRewardCode}
              onChange={(e) => setRlForm({ ...rlForm, customRewardCode: e.target.value })}
              rows={9}
              style={{
                fontFamily: 'var(--font-mono, monospace)',
                fontSize: 11,
                lineHeight: 1.5,
                background: '#0d1117',
                color: '#e6edf3',
                border: '1px solid #30363d',
                borderRadius: 6,
                padding: '8px 10px',
                resize: 'vertical',
                width: '100%',
              }}
              spellCheck={false}
            />
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
              Variables available: <code>state</code> (robot pose), <code>action</code> ([linear_v, angular_w]), <code>info</code> (distance_to_goal, prev_distance, angle_diff, is_collision, is_success). Return float or tuple <code>(reward, terminated)</code>.
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'var(--bg-canvas)', padding: '6px 8px', borderRadius: 4 }}>
            <input
              type="checkbox"
              id="rand_scen"
              checked={rlForm.randomizeScenarios}
              onChange={(e) => setRlForm({ ...rlForm, randomizeScenarios: e.target.checked })}
            />
            <label htmlFor="rand_scen" style={{ fontSize: 10, color: 'var(--text-secondary)', cursor: 'pointer' }}>
              <strong>Domain Randomization:</strong> Randomize obstacle positions &amp; target beacon each episode to learn robust obstacle avoidance.
            </label>
          </div>

          <button
            className="rf-btn rf-btn-primary"
            onClick={handleStartRLTraining}
            disabled={isJobActive}
            style={{
              padding: '6px 12px',
              fontWeight: 600,
              justifyContent: 'center',
              background: 'linear-gradient(135deg, #7c3aed, #4f46e5)',
              border: 'none',
              boxShadow: '0 2px 8px rgba(124, 58, 237, 0.3)',
            }}
          >
            <BrainCircuit size={13} />
            <span>{isJobActive ? 'Training in Progress...' : 'Launch Background PPO Training'}</span>
          </button>
        </div>

        {/* Saved Checkpoints Registry */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span>Saved PPO Policy Checkpoints ({rlModels.length})</span>
            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Persisted in models/rl/</span>
          </div>

          {rlModels.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '16px 10px', color: 'var(--text-muted)', background: 'var(--bg-card)', borderRadius: 6, fontSize: 11 }}>
              No trained PPO models yet. Launch a training run above to generate persistent policy checkpoints.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {rlModels.map((m) => (
                <div
                  key={m.model_id}
                  style={{
                    background: 'var(--bg-card)',
                    padding: '8px 10px',
                    borderRadius: 6,
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span>{m.name}</span>
                      <span style={{ fontSize: 9, padding: '1px 5px', borderRadius: 3, background: 'rgba(59, 130, 246, 0.15)', color: 'var(--accent-blue)' }}>
                        PPO-MlpPolicy
                      </span>
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                      Trained: {m.timesteps_trained.toLocaleString()} steps &bull; Reward: {m.mean_reward.toFixed(2)} &bull; Success: {(m.success_rate * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      className="rf-btn rf-btn-primary"
                      onClick={() => handleExecuteRLModelLive(m.model_id)}
                      style={{ fontSize: 10, padding: '3px 8px' }}
                      title="Load weights into MuJoCo simulation and view real robot movement in 3D"
                    >
                      <Play size={11} fill="#fff" />
                      <span>Execute in 3D</span>
                    </button>
                    <button
                      className="rf-btn"
                      onClick={() => handleRunBenchmarkSuite(m.model_id)}
                      style={{ fontSize: 10, padding: '3px 8px' }}
                      title="Evaluate against APF & Baseline across held-out scenarios"
                    >
                      <Award size={11} />
                      <span>Benchmark</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderBenchmarkView = () => {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg-card)', padding: '8px 12px', borderRadius: 6 }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
              <Award size={16} color="var(--accent-green)" />
              <span>Rigorous Policy Benchmark Suite</span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
              Standardized evaluation on held-out unseen scenarios comparing PPO vs Artificial Potential Field (APF) vs Baseline.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <select
              className="rf-select"
              value={selectedEvalModelId}
              onChange={(e) => setSelectedEvalModelId(e.target.value)}
              style={{ fontSize: 11, padding: '3px 6px' }}
            >
              {rlModels.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.name} ({m.timesteps_trained.toLocaleString()} steps)
                </option>
              ))}
              {rlModels.length === 0 && <option value="">No PPO Models Saved</option>}
            </select>
            <button
              className="rf-btn rf-btn-primary"
              onClick={() => handleRunBenchmarkSuite(selectedEvalModelId)}
              disabled={isEvaluating || rlModels.length === 0}
              style={{ fontSize: 11, padding: '4px 10px', fontWeight: 600 }}
            >
              <Play size={11} fill="#fff" />
              <span>{isEvaluating ? 'Evaluating Benchmark...' : 'Run Benchmark Suite'}</span>
            </button>
          </div>
        </div>

        {/* Evaluation Output */}
        {!benchmarkResult ? (
          <div style={{ textAlign: 'center', padding: '30px 10px', color: 'var(--text-muted)', background: 'var(--bg-card)', borderRadius: 6 }}>
            <div>Select a trained PPO model above and click <strong>Run Benchmark Suite</strong>.</div>
            <div style={{ fontSize: 10, marginTop: 4 }}>
              The suite executes identical evaluation runs on 3 held-out layouts (Slalom, Corridor, Direct Block) and computes 95% confidence intervals.
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {/* Verdict Banner */}
            <div style={{ background: 'var(--bg-card)', padding: '10px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 3, background: 'var(--accent-green)', color: '#000', fontWeight: 700 }}>
                  EMPIRICAL VERDICT
                </span>
                <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)' }}>
                  {benchmarkResult.verdict}
                </span>
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                Evaluated Model: <strong>{benchmarkResult.model_id}</strong> &bull; Completed at {new Date(benchmarkResult.timestamp * 1000).toLocaleTimeString()} &bull; 95% Wilson Confidence Intervals applied.
              </div>
            </div>

            {/* Benchmark Comparison Table */}
            <div style={{ background: 'var(--bg-card)', padding: 10, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 8, color: 'var(--text-primary)' }}>
                Held-Out Benchmark Performance Across All Controllers
              </div>
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '6px 8px' }}>Controller</th>
                    <th style={{ padding: '6px 8px' }}>Success Rate (95% CI)</th>
                    <th style={{ padding: '6px 8px' }}>Collision Rate</th>
                    <th style={{ padding: '6px 8px' }}>Mean Episode Time</th>
                    <th style={{ padding: '6px 8px' }}>Avg Path Length</th>
                    <th style={{ padding: '6px 8px' }}>Training Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarkResult.benchmark_results.map((b) => (
                    <tr key={b.name} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>
                        <span style={{
                          color: b.policy_type === 'PPO' ? 'var(--accent-purple, #a855f7)' : b.policy_type === 'APF' ? 'var(--accent-blue)' : 'var(--text-muted)'
                        }}>
                          {b.name}
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px' }}>
                        <span style={{ fontWeight: 600, color: b.success_rate >= 0.7 ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                          {(b.success_rate * 100).toFixed(1)}%
                        </span>
                        {b.confidence_interval_95?.success_rate_margin !== undefined && (
                          <span style={{ fontSize: 9, color: 'var(--text-muted)', marginLeft: 4 }}>
                            [&plusmn;{(b.confidence_interval_95.success_rate_margin * 100).toFixed(1)}%]
                          </span>
                        )}
                      </td>
                      <td style={{ padding: '6px 8px', color: b.collision_rate > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                        {(b.collision_rate * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '6px 8px' }}>
                        {b.avg_episode_time.toFixed(2)}s
                        {b.confidence_interval_95?.duration_margin_sec !== undefined && (
                          <span style={{ fontSize: 9, color: 'var(--text-muted)', marginLeft: 4 }}>
                            [&plusmn;{b.confidence_interval_95.duration_margin_sec.toFixed(2)}s]
                          </span>
                        )}
                      </td>
                      <td style={{ padding: '6px 8px' }}>
                        {b.avg_path_length.toFixed(2)}m
                      </td>
                      <td style={{ padding: '6px 8px', fontSize: 10, color: 'var(--text-muted)' }}>
                        {b.training_cost_timesteps ? `${b.training_cost_timesteps.toLocaleString()} steps` : '0 steps (Rule-based)'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Statistical Notes */}
              <div style={{ marginTop: 8, fontSize: 10, color: 'var(--text-secondary)', background: 'var(--bg-canvas)', padding: '6px 8px', borderRadius: 4 }}>
                <strong>Empirical Rigor:</strong> Each controller was evaluated on identical held-out scenarios (<code>HeldOut-Slalom</code>, <code>HeldOut-Corridor</code>, <code>HeldOut-DirectBlock</code>) with identical physical seeds. Wilson score intervals quantify statistical uncertainty for discrete outcomes.
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderAIAssistantView = () => {
    return (
      <AIAssistantChat
        onShowToast={onShowToast}
        selectedScenarioId={selectedScenarioId}
        targetPosition={targetPosition}
        onProjectLoaded={onProjectLoaded}
        onSelectTarget={onSelectTarget}
        onSwitchView={setViewMode}
      />
    );
  };


  const renderHypothesesView = () => {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg-card)', padding: '8px 12px', borderRadius: 6 }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
              <Lightbulb size={16} color="var(--accent-yellow)" />
              <span>Evidence-Based Policy Refinement Engine</span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
              Inspects failures, formulates a single controlled hypothesis, reruns with identical seed, and records empirical results without fabrication.
            </div>
          </div>
          <button
            className="rf-btn rf-btn-primary"
            onClick={handleRunHypothesisRefinement}
            disabled={isRefining}
            style={{ fontSize: 11, padding: '4px 10px', fontWeight: 600 }}
          >
            <Lightbulb size={12} />
            <span>{isRefining ? 'Testing Hypothesis...' : 'Formulate & Test Hypothesis'}</span>
          </button>
        </div>

        {/* Hypothesis Results */}
        {!hypoResult ? (
          <div style={{ textAlign: 'center', padding: '30px 10px', color: 'var(--text-muted)', background: 'var(--bg-card)', borderRadius: 6 }}>
            <div>Click <strong>Formulate &amp; Test Hypothesis</strong> to inspect recent runs and test a controlled physical improvement.</div>
            <div style={{ fontSize: 10, marginTop: 4 }}>
              The engine changes exactly ONE parameter (e.g. safety margin or repulsion gain) and reruns the exact same physics seed to measure empirical effect.
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {/* Verdict Banner */}
            <div style={{ background: 'var(--bg-card)', padding: '10px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span style={{
                  fontSize: 10,
                  padding: '2px 6px',
                  borderRadius: 3,
                  background: hypoResult.outcome_verdict === 'SUPPORTED' ? 'var(--accent-green)' : hypoResult.outcome_verdict === 'REFUTED' ? 'var(--accent-red)' : 'var(--accent-yellow)',
                  color: '#000',
                  fontWeight: 700,
                }}>
                  HYPOTHESIS {hypoResult.outcome_verdict}
                </span>
                <span style={{ fontWeight: 700, fontSize: 12, color: 'var(--text-primary)' }}>
                  {hypoResult.hypothesis}
                </span>
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                Observed Issue: <em>{hypoResult.observed_failure}</em> &bull; Parameter: <code>{hypoResult.controlled_parameter}</code> ({String(hypoResult.baseline_value)} &rarr; {String(hypoResult.proposed_value)})
              </div>
            </div>

            {/* Controlled Comparison Card */}
            <div style={{ background: 'var(--bg-card)', padding: 10, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 8 }}>
                Controlled Isolation: Baseline Run vs Refinement Run (Identical Seed)
              </div>
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '6px 8px' }}>Metric</th>
                    <th style={{ padding: '6px 8px' }}>Baseline ({hypoResult.baseline_run_id})</th>
                    <th style={{ padding: '6px 8px' }}>Refinement ({hypoResult.refinement_run_id})</th>
                    <th style={{ padding: '6px 8px' }}>Empirical Change</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '6px 8px' }}>Total Collisions</td>
                    <td style={{ padding: '6px 8px', color: hypoResult.baseline_collisions > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                      {hypoResult.baseline_collisions}
                    </td>
                    <td style={{ padding: '6px 8px', color: hypoResult.refinement_collisions > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                      {hypoResult.refinement_collisions}
                    </td>
                    <td style={{ padding: '6px 8px', fontWeight: 600, color: hypoResult.refinement_collisions < hypoResult.baseline_collisions ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
                      {hypoResult.refinement_collisions - hypoResult.baseline_collisions <= 0 ? `${hypoResult.refinement_collisions - hypoResult.baseline_collisions}` : `+${hypoResult.refinement_collisions - hypoResult.baseline_collisions}`}
                    </td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '6px 8px' }}>Task Success</td>
                    <td style={{ padding: '6px 8px' }}>{hypoResult.baseline_success ? 'Success' : 'Failed'}</td>
                    <td style={{ padding: '6px 8px', color: hypoResult.refinement_success ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {hypoResult.refinement_success ? 'Success' : 'Failed'}
                    </td>
                    <td style={{ padding: '6px 8px' }}>
                      {hypoResult.refinement_success && !hypoResult.baseline_success ? '+Improved' : hypoResult.refinement_success ? 'Maintained' : 'No Change'}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '6px 8px' }}>Episode Duration</td>
                    <td style={{ padding: '6px 8px' }}>{hypoResult.baseline_time.toFixed(2)}s</td>
                    <td style={{ padding: '6px 8px' }}>{hypoResult.refinement_time.toFixed(2)}s</td>
                    <td style={{ padding: '6px 8px' }}>
                      {(hypoResult.refinement_time - hypoResult.baseline_time) > 0 ? `+${(hypoResult.refinement_time - hypoResult.baseline_time).toFixed(2)}s` : `${(hypoResult.refinement_time - hypoResult.baseline_time).toFixed(2)}s`}
                    </td>
                  </tr>
                </tbody>
              </table>

              {/* Empirical Evidence Description */}
              <div style={{ marginTop: 10, fontSize: 10, color: 'var(--text-secondary)', background: 'var(--bg-canvas)', padding: '6px 8px', borderRadius: 4 }}>
                <strong>Empirical Evidence:</strong> {hypoResult.empirical_evidence}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const gridTemplate = layoutMode === 'metrics'
    ? '220px 1fr 220px'
    : layoutMode === 'scenarios'
    ? '380px 1fr 220px'
    : '280px 1fr 280px';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: gridTemplate, gap: 10, height: '100%', overflow: 'hidden', fontSize: 12, transition: 'grid-template-columns 0.2s ease' }}>
      {/* 1. SCENARIO CONFIGURATION & RUNNER */}
      <div style={{ background: 'var(--bg-panel)', padding: 10, borderRadius: 6, border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontWeight: 600, fontSize: 12, display: 'flex', alignItems: 'center', gap: 5 }}>
            <Layers size={14} color="var(--accent-blue)" />
            Scenario Configuration
          </span>
          <div style={{ display: 'flex', gap: 4 }}>
            <button
              className="rf-btn"
              onClick={() => setShowNewScenarioModal(true)}
              title="Create New Custom Scenario"
              style={{ fontSize: 10, padding: '2px 6px', color: 'var(--accent-green)', fontWeight: 600 }}
            >
              <Plus size={11} />
              <span>New</span>
            </button>
            <button
              className="rf-btn"
              onClick={handleSaveActiveAsScenario}
              title="Save current 3D scene as a scenario"
              style={{ fontSize: 10, padding: '2px 6px' }}
            >
              + Scene
            </button>
          </div>
        </div>

        <div>
          <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>Select Scenario:</label>
          <select
            className="rf-select"
            value={selectedScenarioId}
            onChange={(e) => handleScenarioChange(e.target.value)}
            style={{ width: '100%', padding: '4px 6px' }}
          >
            {scenarios.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} (Seed {s.seed})
              </option>
            ))}
          </select>
        </div>

        {selectedScenario && (
          <div style={{ background: 'var(--bg-card)', padding: '6px 8px', borderRadius: 4, fontSize: 11, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ color: 'var(--text-secondary)' }}>{selectedScenario.description}</div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: 10 }}>
              <span>Target: [{selectedScenario.task.target_position.slice(0, 2).join(', ')}]</span>
              <span>Tol: {selectedScenario.task.distance_threshold}m</span>
            </div>
          </div>
        )}

        {/* Avoidance Policy Tuner Section */}
        <div style={{ background: 'var(--bg-card)', padding: '6px 8px', borderRadius: 4, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 10, fontWeight: 600, color: '#a855f7', display: 'flex', alignItems: 'center', gap: 4 }}>
              <BrainCircuit size={12} />
              <span>Avoidance Policy</span>
            </span>
            <button
              className="rf-btn"
              onClick={() => setShowPolicyTuner(!showPolicyTuner)}
              style={{ fontSize: 9, padding: '1px 5px' }}
            >
              <Sliders size={10} />
              <span>{showPolicyTuner ? 'Hide' : 'Tune'}</span>
            </button>
          </div>

          <div style={{ fontSize: 9, color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
            <span>Clearance: {policyParams.safety_margin.toFixed(2)}m</span>
            <span>Repulse: {policyParams.repulsion_gain.toFixed(1)}x</span>
            <span>Swerve: {policyParams.tangential_gain.toFixed(1)}x</span>
          </div>

          {showPolicyTuner && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4, paddingTop: 4, borderTop: '1px solid var(--border-subtle)' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)' }}>
                  <span>Safety Margin</span>
                  <span>{policyParams.safety_margin.toFixed(2)}m</span>
                </div>
                <input
                  type="range"
                  min="0.3"
                  max="1.5"
                  step="0.05"
                  value={policyParams.safety_margin}
                  onChange={(e) => setPolicyParams({ ...policyParams, safety_margin: parseFloat(e.target.value) })}
                  style={{ width: '100%', height: 4 }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)' }}>
                  <span>Repulsion Force</span>
                  <span>{policyParams.repulsion_gain.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="5.0"
                  step="0.1"
                  value={policyParams.repulsion_gain}
                  onChange={(e) => setPolicyParams({ ...policyParams, repulsion_gain: parseFloat(e.target.value) })}
                  style={{ width: '100%', height: 4 }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)' }}>
                  <span>Tangential Swerve</span>
                  <span>{policyParams.tangential_gain.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="4.0"
                  step="0.1"
                  value={policyParams.tangential_gain}
                  onChange={(e) => setPolicyParams({ ...policyParams, tangential_gain: parseFloat(e.target.value) })}
                  style={{ width: '100%', height: 4 }}
                />
              </div>

              <button
                className="rf-btn rf-btn-primary"
                onClick={handleSavePolicyTuning}
                style={{ fontSize: 10, padding: '3px 6px', marginTop: 2, justifyContent: 'center' }}
              >
                <Save size={11} />
                <span>Save Policy Tuning</span>
              </button>
            </div>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
          <div>
            <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Seed:</label>
            <input
              type="number"
              className="rf-input"
              value={customSeed}
              onChange={(e) => setCustomSeed(parseInt(e.target.value) || 0)}
              style={{ width: '100%', padding: '3px 6px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Duration (s):</label>
            <input
              type="number"
              className="rf-input"
              step="0.5"
              value={customDuration}
              onChange={(e) => setCustomDuration(parseFloat(e.target.value) || 1.0)}
              style={{ width: '100%', padding: '3px 6px' }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 'auto' }}>
          <div style={{ display: 'flex', gap: 4 }}>
            <button
              className="rf-btn rf-btn-primary"
              onClick={handleRunLive}
              disabled={isRunning || isTraining || !selectedScenarioId}
              style={{
                flex: 1,
                padding: '6px 8px',
                fontWeight: 600,
                background: 'linear-gradient(135deg, #0284c7, #2563eb)',
                color: '#fff',
                border: 'none',
              }}
              title="Execute live in 3D viewport with real-time autonomous steering and visual trajectory"
            >
              <Play size={13} fill="#fff" />
              <span>{isRunning ? 'Running in 3D...' : 'Run in 3D Viewport'}</span>
            </button>
            <button
              className="rf-btn"
              onClick={handleRunExperiment}
              disabled={isRunning || isTraining || !selectedScenarioId}
              style={{ padding: '6px 8px', fontSize: 11 }}
              title="Fast Headless Run"
            >
              <span>Fast Run</span>
            </button>
          </div>

          <button
            className="rf-btn"
            onClick={handleTrainPolicy}
            disabled={isRunning || isTraining || !selectedScenarioId}
            style={{
              padding: '6px 8px',
              fontWeight: 600,
              background: 'linear-gradient(135deg, #7c3aed, #4f46e5)',
              color: '#fff',
              border: 'none',
              boxShadow: '0 2px 8px rgba(124, 58, 237, 0.25)',
              justifyContent: 'center',
            }}
            title="Train and auto-optimize obstacle avoidance parameters across multiple simulation passes"
          >
            <BrainCircuit size={13} />
            <span>{isTraining ? 'Training Robot...' : 'Train Avoidance Policy'}</span>
          </button>

          <div style={{ display: 'flex', gap: 4 }}>
            <button className="rf-btn" onClick={handleDuplicateScenario} title="Duplicate Scenario" style={{ flex: 1, padding: '4px 6px', fontSize: 10 }}>
              <Copy size={11} />
              <span>Duplicate</span>
            </button>
            <button className="rf-btn" onClick={handleLoadToScene} title="Load to 3D Scene" style={{ flex: 1, padding: '4px 6px', fontSize: 10 }}>
              <ExternalLink size={11} />
              <span>Load Scene</span>
            </button>
          </div>
        </div>
      </div>

      {/* 2. CENTER: METRICS VIEW OR COMPARISON */}
      <div style={{ background: 'var(--bg-panel)', padding: 10, borderRadius: 6, border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: 6 }}>
          <div style={{ display: 'flex', gap: 4, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              className={`rf-btn ${viewMode === 'metrics' ? 'rf-btn-primary' : ''}`}
              onClick={() => setViewMode('metrics')}
              style={{ fontSize: 11, padding: '2px 8px' }}
            >
              <BarChart2 size={12} />
              <span>Metrics &amp; Plots</span>
            </button>
            <button
              className={`rf-btn ${viewMode === 'compare' ? 'rf-btn-primary' : ''}`}
              onClick={() => setViewMode('compare')}
              style={{ fontSize: 11, padding: '2px 8px' }}
            >
              <GitCompare size={12} />
              <span>Run Comparison</span>
            </button>
            <button
              className={`rf-btn ${viewMode === 'rl_training' ? 'rf-btn-primary' : ''}`}
              onClick={() => { setViewMode('rl_training'); loadRLModelsList(); }}
              style={{ fontSize: 11, padding: '2px 8px', borderColor: viewMode === 'rl_training' ? 'var(--accent-purple, #a855f7)' : undefined }}
            >
              <BrainCircuit size={12} color={viewMode === 'rl_training' ? '#fff' : 'var(--accent-purple, #a855f7)'} />
              <span>PPO Training</span>
            </button>
            <button
              className={`rf-btn ${viewMode === 'benchmark' ? 'rf-btn-primary' : ''}`}
              onClick={() => setViewMode('benchmark')}
              style={{ fontSize: 11, padding: '2px 8px', borderColor: viewMode === 'benchmark' ? 'var(--accent-green)' : undefined }}
            >
              <Award size={12} color={viewMode === 'benchmark' ? '#fff' : 'var(--accent-green)'} />
              <span>Policy Benchmark</span>
            </button>
            <button
              className={`rf-btn ${viewMode === 'ai_assistant' ? 'rf-btn-primary' : ''}`}
              onClick={() => setViewMode('ai_assistant')}
              style={{ fontSize: 11, padding: '2px 8px', borderColor: viewMode === 'ai_assistant' ? 'var(--accent-blue)' : undefined }}
            >
              <Sparkles size={12} color={viewMode === 'ai_assistant' ? '#fff' : 'var(--accent-blue)'} />
              <span>AI Assistant</span>
            </button>
            <button
              className={`rf-btn ${viewMode === 'hypotheses' ? 'rf-btn-primary' : ''}`}
              onClick={() => setViewMode('hypotheses')}
              style={{ fontSize: 11, padding: '2px 8px', borderColor: viewMode === 'hypotheses' ? 'var(--accent-yellow)' : undefined }}
            >
              <Lightbulb size={12} color={viewMode === 'hypotheses' ? '#fff' : 'var(--accent-yellow)'} />
              <span>Hypothesis</span>
            </button>

            {/* Layout Priority Mode Switcher */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 3, background: 'var(--bg-canvas)', padding: '2px 4px', borderRadius: 4, border: '1px solid var(--border-subtle)', marginLeft: 8 }}>
              <span style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', marginRight: 2 }}>Priority:</span>
              <button
                className="rf-btn"
                onClick={() => setLayoutMode('scenarios')}
                style={{
                  fontSize: 9,
                  padding: '1px 5px',
                  backgroundColor: layoutMode === 'scenarios' ? 'var(--accent-blue)' : 'transparent',
                  color: layoutMode === 'scenarios' ? '#fff' : 'var(--text-muted)'
                }}
                title="Prioritize Scenario Setup & Tuning Space"
              >
                Scenarios
              </button>
              <button
                className="rf-btn"
                onClick={() => setLayoutMode('balanced')}
                style={{
                  fontSize: 9,
                  padding: '1px 5px',
                  backgroundColor: layoutMode === 'balanced' ? 'var(--accent-blue)' : 'transparent',
                  color: layoutMode === 'balanced' ? '#fff' : 'var(--text-muted)'
                }}
                title="Balanced 3-Column Space"
              >
                Balanced
              </button>
              <button
                className="rf-btn"
                onClick={() => setLayoutMode('metrics')}
                style={{
                  fontSize: 9,
                  padding: '1px 5px',
                  backgroundColor: layoutMode === 'metrics' ? 'var(--accent-blue)' : 'transparent',
                  color: layoutMode === 'metrics' ? '#fff' : 'var(--text-muted)'
                }}
                title="Prioritize Metrics & Real-time Charts Space"
              >
                Metrics
              </button>
            </div>
          </div>

          {activeRun && viewMode === 'metrics' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Run: <strong>{activeRun.run_id}</strong> (Seed {activeRun.seed})
              </span>
              <button
                className="rf-btn"
                style={{ fontSize: 10, padding: '2px 6px', display: 'flex', alignItems: 'center', gap: 4 }}
                onClick={() => {
                  if (activeRun.measurements && onSetTrail) {
                    onSetTrail(activeRun.measurements.map((m) => m.position));
                    onShowToast('Trajectory trail displayed in 3D Viewport');
                  }
                }}
                title="Display 3D trajectory path in viewport"
              >
                <Eye size={11} color="var(--accent-blue)" />
                <span>Show Trail</span>
              </button>
            </div>
          )}
        </div>

        {viewMode === 'metrics' && (
          activeRun ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {/* Status Header */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg-card)', padding: '6px 10px', borderRadius: 5 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  {activeRun.metrics.success ? (
                    <CheckCircle2 size={16} color="var(--accent-green)" />
                  ) : activeRun.status === 'timeout' ? (
                    <AlertCircle size={16} color="var(--accent-yellow)" />
                  ) : (
                    <XCircle size={16} color="var(--accent-red)" />
                  )}
                  <span style={{ fontWeight: 600 }}>{activeRun.experiment_name}</span>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>v{activeRun.config_version}</span>
                </div>
                <div style={{ fontSize: 11, color: activeRun.metrics.success ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                  {activeRun.metrics.status_reason}
                </div>
              </div>

              {/* Metric Cards Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6 }}>
                <div style={{ background: 'var(--bg-card)', padding: 6, borderRadius: 4 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Duration</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{activeRun.metrics.total_duration}s</div>
                </div>
                <div style={{ background: 'var(--bg-card)', padding: 6, borderRadius: 4 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Final Dist</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: activeRun.metrics.final_distance_to_target <= 0.25 ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                    {activeRun.metrics.final_distance_to_target}m
                  </div>
                </div>
                <div style={{ background: 'var(--bg-card)', padding: 6, borderRadius: 4 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Min Dist</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{activeRun.metrics.min_distance_to_target}m</div>
                </div>
                <div style={{ background: 'var(--bg-card)', padding: 6, borderRadius: 4 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Path Traveled</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{activeRun.metrics.total_distance_traveled}m</div>
                </div>
                <div style={{ background: 'var(--bg-card)', padding: 6, borderRadius: 4 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Collisions</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: activeRun.metrics.total_collisions > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                    {activeRun.metrics.total_collisions}
                  </div>
                </div>
              </div>

              {/* Time Series Plots */}
              <div style={{ display: 'flex', gap: 10 }}>
                {renderDistancePlot(activeRun.measurements)}
                {renderSpeedPlot(activeRun.measurements)}
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '30px 10px', color: 'var(--text-muted)' }}>
              Select a scenario and click <strong>Run Experiment</strong> to record deterministic physics metrics and plots.
            </div>
          )
        )}

        {viewMode === 'compare' && (
          comparisonResult ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ background: 'var(--bg-card)', padding: '6px 10px', borderRadius: 4, fontSize: 11, color: 'var(--text-primary)' }}>
                <strong>Comparison Summary:</strong> {comparisonResult.summary}
              </div>

              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '4px 6px' }}>Metric</th>
                    <th style={{ padding: '4px 6px' }}>Run A ({comparisonResult.run_a_seed})</th>
                    <th style={{ padding: '4px 6px' }}>Run B ({comparisonResult.run_b_seed})</th>
                    <th style={{ padding: '4px 6px' }}>Delta (&Delta; B - A)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '4px 6px' }}>Duration</td>
                    <td style={{ padding: '4px 6px' }}>{comparisonResult.run_a_metrics.total_duration}s</td>
                    <td style={{ padding: '4px 6px' }}>{comparisonResult.run_b_metrics.total_duration}s</td>
                    <td style={{ padding: '4px 6px', color: comparisonResult.metric_deltas.duration_delta <= 0 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                      {comparisonResult.metric_deltas.duration_delta > 0 ? `+${comparisonResult.metric_deltas.duration_delta}` : comparisonResult.metric_deltas.duration_delta}s
                    </td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '4px 6px' }}>Final Distance</td>
                    <td style={{ padding: '4px 6px' }}>{comparisonResult.run_a_metrics.final_distance_to_target}m</td>
                    <td style={{ padding: '4px 6px' }}>{comparisonResult.run_b_metrics.final_distance_to_target}m</td>
                    <td style={{ padding: '4px 6px', color: comparisonResult.metric_deltas.final_distance_delta < 0 ? 'var(--accent-green)' : 'var(--accent-orange)' }}>
                      {comparisonResult.metric_deltas.final_distance_delta > 0 ? `+${comparisonResult.metric_deltas.final_distance_delta}` : comparisonResult.metric_deltas.final_distance_delta}m
                    </td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '4px 6px' }}>Path Traveled</td>
                    <td style={{ padding: '4px 6px' }}>{comparisonResult.run_a_metrics.total_distance_traveled}m</td>
                    <td style={{ padding: '4px 6px' }}>{comparisonResult.run_b_metrics.total_distance_traveled}m</td>
                    <td style={{ padding: '4px 6px' }}>
                      {comparisonResult.metric_deltas.total_distance_delta > 0 ? `+${comparisonResult.metric_deltas.total_distance_delta}` : comparisonResult.metric_deltas.total_distance_delta}m
                    </td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '4px 6px' }}>Collisions</td>
                    <td style={{ padding: '4px 6px' }}>{comparisonResult.run_a_metrics.total_collisions}</td>
                    <td style={{ padding: '4px 6px' }}>{comparisonResult.run_b_metrics.total_collisions}</td>
                    <td style={{ padding: '4px 6px', color: comparisonResult.metric_deltas.collisions_delta <= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {comparisonResult.metric_deltas.collisions_delta > 0 ? `+${comparisonResult.metric_deltas.collisions_delta}` : comparisonResult.metric_deltas.collisions_delta}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '4px 6px' }}>Success</td>
                    <td style={{ padding: '4px 6px', color: comparisonResult.run_a_metrics.success ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {comparisonResult.run_a_metrics.success ? 'Yes' : 'No'}
                    </td>
                    <td style={{ padding: '4px 6px', color: comparisonResult.run_b_metrics.success ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {comparisonResult.run_b_metrics.success ? 'Yes' : 'No'}
                    </td>
                    <td style={{ padding: '4px 6px' }}>-</td>
                  </tr>
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '30px 10px', color: 'var(--text-muted)' }}>
              Select two runs in the right panel and click <strong>Compare Runs</strong>.
            </div>
          )
        )}

        {viewMode === 'rl_training' && renderRLTrainingView()}
        {viewMode === 'benchmark' && renderBenchmarkView()}
        {viewMode === 'ai_assistant' && renderAIAssistantView()}
        {viewMode === 'hypotheses' && renderHypothesesView()}
      </div>

      {/* 3. RIGHT: RUN HISTORY & COMPARISON SELECTOR */}
      <div style={{ background: 'var(--bg-panel)', padding: 10, borderRadius: 6, border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontWeight: 600, fontSize: 12 }}>Run History ({runs.length})</span>
          <button className="rf-btn" onClick={loadRunsList} title="Refresh Runs" style={{ padding: '2px 5px' }}>
            <RefreshCw size={11} />
          </button>
        </div>

        {/* Compare Selectors */}
        <div style={{ background: 'var(--bg-card)', padding: 6, borderRadius: 4, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>Compare Two Runs:</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <select
              className="rf-select"
              value={compareRunAId}
              onChange={(e) => setCompareRunAId(e.target.value)}
              style={{ padding: '2px 4px', fontSize: 10 }}
            >
              <option value="">Select Run A</option>
              {runs.map((r) => (
                <option key={`a_${r.run_id}`} value={r.run_id}>
                  {r.experiment_name} (Seed {r.seed}) - {r.run_id}
                </option>
              ))}
            </select>
            <select
              className="rf-select"
              value={compareRunBId}
              onChange={(e) => setCompareRunBId(e.target.value)}
              style={{ padding: '2px 4px', fontSize: 10 }}
            >
              <option value="">Select Run B</option>
              {runs.map((r) => (
                <option key={`b_${r.run_id}`} value={r.run_id}>
                  {r.experiment_name} (Seed {r.seed}) - {r.run_id}
                </option>
              ))}
            </select>
          </div>
          <button
            className="rf-btn rf-btn-primary"
            onClick={handleCompareRuns}
            disabled={!compareRunAId || !compareRunBId}
            style={{ padding: '3px 6px', fontSize: 10, justifyContent: 'center' }}
          >
            <GitCompare size={11} />
            <span>Compare Runs</span>
          </button>
        </div>

        {/* Past Runs Scroll List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, overflowY: 'auto' }}>
          {runs.map((r) => (
            <div
              key={r.run_id}
              onClick={async () => {
                const fullRun = await fetchRun(r.run_id);
                setActiveRun(fullRun);
                setViewMode('metrics');
                if (onSetTrail && fullRun.measurements) {
                  onSetTrail(fullRun.measurements.map((m) => m.position));
                }
                const matchedScen = scenarios.find((s) => s.id === fullRun.experiment_id);
                if (matchedScen && onSelectTarget) {
                  onSelectTarget(matchedScen.task.target_position);
                }
              }}
              style={{
                padding: '4px 6px',
                borderRadius: 4,
                background: activeRun?.run_id === r.run_id ? 'var(--bg-card-selected)' : 'var(--bg-card)',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: 10,
              }}
            >
              <div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{r.experiment_name}</div>
                <div style={{ color: 'var(--text-muted)' }}>
                  Seed {r.seed} &bull; {r.metrics?.total_duration}s &bull; {r.metrics?.final_distance_to_target}m
                </div>
              </div>
              <span style={{
                color: r.metrics?.success ? 'var(--accent-green)' : 'var(--accent-yellow)',
                fontWeight: 600,
                fontSize: 9,
              }}>
                {r.metrics?.success ? 'SUCCESS' : r.status.toUpperCase()}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Policy Training Results Modal */}
      {showTrainingModal && trainingResult && (
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
          zIndex: 1000,
        }}>
          <div style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 8,
            width: 480,
            maxWidth: '90vw',
            padding: 16,
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
            boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                <BrainCircuit size={16} color="#a855f7" />
                <span>Policy Optimization & Training Report</span>
              </span>
              <button className="rf-btn" onClick={() => setShowTrainingModal(false)} style={{ padding: '2px 5px' }}>
                <X size={13} />
              </button>
            </div>

            <div style={{
              background: trainingResult.trained_collisions === 0 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(59, 130, 246, 0.15)',
              border: `1px solid ${trainingResult.trained_collisions === 0 ? 'var(--accent-green)' : 'var(--accent-blue)'}`,
              borderRadius: 6,
              padding: '8px 12px',
              fontSize: 11,
            }}>
              <div style={{ fontWeight: 600, color: trainingResult.trained_collisions === 0 ? 'var(--accent-green)' : 'var(--accent-blue)' }}>
                {trainingResult.message}
              </div>
              <div style={{ marginTop: 4, display: 'flex', gap: 16, color: 'var(--text-secondary)' }}>
                <span>Baseline Collisions: <strong>{trainingResult.baseline_collisions}</strong></span>
                <span>Optimal Collisions: <strong style={{ color: trainingResult.trained_collisions === 0 ? 'var(--accent-green)' : 'inherit' }}>{trainingResult.trained_collisions}</strong></span>
              </div>
            </div>

            {/* Iterations Table */}
            <div style={{ maxHeight: 160, overflowY: 'auto', border: '1px solid var(--border-subtle)', borderRadius: 4 }}>
              <table style={{ width: '100%', fontSize: 10, textAlign: 'left', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-card)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '4px 6px' }}>#</th>
                    <th style={{ padding: '4px 6px' }}>Clearance</th>
                    <th style={{ padding: '4px 6px' }}>Repulse</th>
                    <th style={{ padding: '4px 6px' }}>Swerve</th>
                    <th style={{ padding: '4px 6px' }}>Collisions</th>
                    <th style={{ padding: '4px 6px' }}>Time</th>
                    <th style={{ padding: '4px 6px' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingResult.iterations.map((it) => (
                    <tr
                      key={it.iteration}
                      style={{
                        borderTop: '1px solid var(--border-subtle)',
                        background: it.policy.safety_margin === trainingResult.best_policy.safety_margin ? 'rgba(168, 85, 247, 0.15)' : 'transparent',
                      }}
                    >
                      <td style={{ padding: '4px 6px' }}>{it.iteration}</td>
                      <td style={{ padding: '4px 6px' }}>{it.policy.safety_margin.toFixed(2)}m</td>
                      <td style={{ padding: '4px 6px' }}>{it.policy.repulsion_gain.toFixed(1)}x</td>
                      <td style={{ padding: '4px 6px' }}>{it.policy.tangential_gain.toFixed(1)}x</td>
                      <td style={{ padding: '4px 6px', fontWeight: 600, color: it.collisions === 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {it.collisions}
                      </td>
                      <td style={{ padding: '4px 6px' }}>{it.time_to_goal ? `${it.time_to_goal.toFixed(2)}s` : '-'}</td>
                      <td style={{ padding: '4px 6px', color: it.success ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
                        {it.success ? 'SUCCESS' : 'FAILED'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 4 }}>
              <button className="rf-btn" onClick={() => setShowTrainingModal(false)} style={{ padding: '5px 10px' }}>
                Close
              </button>
              <button
                className="rf-btn rf-btn-primary"
                onClick={() => {
                  setShowTrainingModal(false);
                  handleRunLive();
                }}
                style={{
                  padding: '5px 12px',
                  fontWeight: 600,
                  background: 'linear-gradient(135deg, #0284c7, #2563eb)',
                }}
              >
                <Play size={13} fill="#fff" />
                <span>Test Trained Policy in 3D</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* New Scenario Creation Modal */}
      {showNewScenarioModal && (
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
          zIndex: 1000,
        }}>
          <div style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 8,
            width: 440,
            maxWidth: '90vw',
            padding: 16,
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
            boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Plus size={16} color="var(--accent-green)" />
                <span>Create New Simulation Scenario</span>
              </span>
              <button className="rf-btn" onClick={() => setShowNewScenarioModal(false)} style={{ padding: '2px 5px' }}>
                <X size={13} />
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div>
                <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Scenario Name:</label>
                <input
                  type="text"
                  className="rf-input"
                  value={newScenarioForm.name}
                  onChange={(e) => setNewScenarioForm({ ...newScenarioForm, name: e.target.value })}
                  style={{ width: '100%', padding: '4px 6px' }}
                />
              </div>

              {/* Target Coordinates with 3D Sync & Presets */}
              <div style={{ background: 'var(--bg-canvas)', padding: 10, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <Target size={13} />
                    <span>Target Waypoint Location</span>
                  </span>
                  {targetPosition && (
                    <button
                      type="button"
                      className="rf-btn"
                      onClick={() => setNewScenarioForm({ ...newScenarioForm, targetX: targetPosition[0], targetY: targetPosition[1] })}
                      style={{ fontSize: 10, padding: '2px 6px', color: 'var(--accent-blue)' }}
                      title="Copy coordinates from active 3D beacon"
                    >
                      Use 3D Beacon ({targetPosition[0].toFixed(1)}, {targetPosition[1].toFixed(1)})
                    </button>
                  )}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
                  <div>
                    <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Target X (m):</label>
                    <input
                      type="number"
                      step="0.2"
                      className="rf-input"
                      value={newScenarioForm.targetX}
                      onChange={(e) => setNewScenarioForm({ ...newScenarioForm, targetX: parseFloat(e.target.value) || 0 })}
                      style={{ width: '100%', padding: '4px 6px' }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Target Y (m):</label>
                    <input
                      type="number"
                      step="0.2"
                      className="rf-input"
                      value={newScenarioForm.targetY}
                      onChange={(e) => setNewScenarioForm({ ...newScenarioForm, targetY: parseFloat(e.target.value) || 0 })}
                      style={{ width: '100%', padding: '4px 6px' }}
                    />
                  </div>
                </div>

                {/* Quick Presets */}
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', alignSelf: 'center' }}>Presets:</span>
                  <button
                    type="button"
                    className="rf-btn"
                    onClick={() => setNewScenarioForm({ ...newScenarioForm, targetX: 2.0, targetY: 0.0 })}
                    style={{ fontSize: 10, padding: '2px 6px' }}
                  >
                    Ahead (+2m)
                  </button>
                  <button
                    type="button"
                    className="rf-btn"
                    onClick={() => setNewScenarioForm({ ...newScenarioForm, targetX: 2.5, targetY: 0.8 })}
                    style={{ fontSize: 10, padding: '2px 6px' }}
                  >
                    Slalom (+2.5, +0.8)
                  </button>
                  <button
                    type="button"
                    className="rf-btn"
                    onClick={() => setNewScenarioForm({ ...newScenarioForm, targetX: 2.0, targetY: -1.2 })}
                    style={{ fontSize: 10, padding: '2px 6px' }}
                  >
                    Left (+2, -1.2)
                  </button>
                  <button
                    type="button"
                    className="rf-btn"
                    onClick={() => setNewScenarioForm({ ...newScenarioForm, targetX: 3.5, targetY: 0.0 })}
                    style={{ fontSize: 10, padding: '2px 6px' }}
                  >
                    Far (+3.5m)
                  </button>
                </div>

                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6, fontStyle: 'italic' }}>
                  💡 Tip: You can also drag the green beacon directly in the 3D viewport using the Move gizmo!
                </div>
              </div>

              {/* Scene Obstacles Summary */}
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', background: 'var(--bg-canvas)', padding: '6px 10px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                🧱 <strong>Included Obstacles:</strong> All <strong>{currentProject?.objects?.length || 0} active scene obstacles</strong> will be saved into this new scenario.
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <div>
                  <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Seed:</label>
                  <input
                    type="number"
                    className="rf-input"
                    value={newScenarioForm.seed}
                    onChange={(e) => setNewScenarioForm({ ...newScenarioForm, seed: parseInt(e.target.value) || 0 })}
                    style={{ width: '100%', padding: '4px 6px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Timeout (s):</label>
                  <input
                    type="number"
                    step="0.5"
                    className="rf-input"
                    value={newScenarioForm.timeout}
                    onChange={(e) => setNewScenarioForm({ ...newScenarioForm, timeout: parseFloat(e.target.value) || 5.0 })}
                    style={{ width: '100%', padding: '4px 6px' }}
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <div>
                  <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Initial Safety Margin (m):</label>
                  <input
                    type="number"
                    step="0.05"
                    className="rf-input"
                    value={newScenarioForm.safetyMargin}
                    onChange={(e) => setNewScenarioForm({ ...newScenarioForm, safetyMargin: parseFloat(e.target.value) || 0.5 })}
                    style={{ width: '100%', padding: '4px 6px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Repulsion Gain:</label>
                  <input
                    type="number"
                    step="0.1"
                    className="rf-input"
                    value={newScenarioForm.repulsionGain}
                    onChange={(e) => setNewScenarioForm({ ...newScenarioForm, repulsionGain: parseFloat(e.target.value) || 1.0 })}
                    style={{ width: '100%', padding: '4px 6px' }}
                  />
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 4 }}>
              <button className="rf-btn" onClick={() => setShowNewScenarioModal(false)} style={{ padding: '5px 10px' }}>
                Cancel
              </button>
              <button
                className="rf-btn rf-btn-primary"
                onClick={handleCreateCustomScenario}
                style={{ padding: '5px 14px', fontWeight: 600 }}
              >
                Create Scenario
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
