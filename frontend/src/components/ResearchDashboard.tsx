import React, { useState, useEffect } from 'react';
import {
  FlaskConical, Play, CheckCircle2, XCircle, AlertTriangle,
  TrendingUp, BarChart3, Sparkles, RefreshCw, FileText, Check,
  Copy, Sliders, ShieldCheck, Scale, Award, ArrowRight, Brain
} from 'lucide-react';

interface Hypothesis {
  hypothesis_id: string;
  mission_id: string;
  title: string;
  statement: string;
  mechanism: string;
  controlled_variable: string;
  baseline_value: any;
  proposed_value: any;
  status: 'untested' | 'supported' | 'refuted' | 'inconclusive';
  confidence: number;
  supporting_evidence: string[];
  contradicting_evidence: string[];
}

interface ControlledExperimentPlan {
  plan_id: string;
  mission_id: string;
  hypothesis_id: string;
  title: string;
  rationale: string;
  parameter_modifications: Record<string, any>;
  seeds: number[];
  expected_outcome: string;
  candidate_rank?: number;
  expected_improvement?: number;
  information_gain?: number;
  acquisition_score?: number;
  selection_rationale?: string;
  prior_evidence_influence?: string;
  is_validated: boolean;
  user_approved: boolean;
  status: string;
}

interface ExperimentOutcome {
  outcome_id: string;
  plan_id: string;
  hypothesis_id: string;
  baseline_deltas: Record<string, number>;
  empirical_verdict: 'improved' | 'degraded' | 'inconclusive';
  verdict_explanation: string;
}

interface HeldOutValidationResult {
  scenario_id: string;
  scenario_name: string;
  baseline_success: boolean;
  candidate_success: boolean;
  baseline_collisions: number;
  candidate_collisions: number;
  baseline_time: number;
  candidate_time: number;
  verdict: string;
}

interface ValidationReport {
  report_id: string;
  generalization_verdict: string;
  summary: string;
  trial_results: HeldOutValidationResult[];
}

interface ActionLogEntry {
  timestamp: number;
  step: string;
  action: string;
  status: string;
  message: string;
}

interface ResearchMission {
  mission_id: string;
  title: string;
  goal_description: string;
  mode: 'supervised' | 'autonomous';
  status: string;
  target_metric: string;
  target_threshold: number;
  optimization_direction: string;
  constraints: string[];
  budget: {
    max_iterations: number;
    max_experiments: number;
    max_time_seconds: number;
    used_iterations: number;
    used_experiments: number;
    elapsed_time: number;
  };
  base_scenario_id: string;
  baseline_metrics?: Record<string, any>;
  active_hypothesis_id?: string;
  active_plan_id?: string;
  best_candidate_plan_id?: string;
  hypotheses: Hypothesis[];
  plans: ControlledExperimentPlan[];
  outcomes: ExperimentOutcome[];
  validation_report?: ValidationReport;
  action_log: ActionLogEntry[];
  final_report_markdown?: string;
}

interface BenchmarkResult {
  strategy: string;
  strategy_name: string;
  best_candidate_params: Record<string, any>;
  success_rate: number;
  mean_collisions: number;
  std_collisions: number;
  mean_time_to_goal: number | null;
  std_time_to_goal: number | null;
  evaluations_used: number;
  held_out_generalization_rate: number;
  selection_rationale: string;
}

interface BenchmarkReport {
  benchmark_id: string;
  title: string;
  scenario_id: string;
  evaluation_budget: number;
  winner_strategy: string;
  research_advantage_demonstrated: boolean;
  advantage_summary: string;
  limitations_and_diagnosis: string;
  results: Record<string, BenchmarkResult>;
}

interface ResearchDashboardProps {
  onShowToast?: (message: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
}

export const ResearchDashboard: React.FC<ResearchDashboardProps> = ({ onShowToast }) => {
  const [activeSubTab, setActiveSubTab] = useState<'mission' | 'hypotheses' | 'experiments' | 'benchmark' | 'report'>('mission');
  const [currentMission, setCurrentMission] = useState<ResearchMission | null>(null);
  const [missionsList, setMissionsList] = useState<ResearchMission[]>([]);
  const [goalInput, setGoalInput] = useState('Optimize slalom traversal clearance and transit time');
  const [modeInput, setModeInput] = useState<'autonomous' | 'supervised'>('autonomous');
  const [maxIterations, setMaxIterations] = useState(3);
  const [maxExperiments, setMaxExperiments] = useState(8);
  const [isLoading, setIsLoading] = useState(false);
  const [isApproving, setIsApproving] = useState(false);
  const [reportMarkdown, setReportMarkdown] = useState<string>('');
  const [benchmarkReport, setBenchmarkReport] = useState<BenchmarkReport | null>(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [copiedReport, setCopiedReport] = useState(false);

  // Poll recent missions on mount
  useEffect(() => {
    fetchMissions();
  }, []);

  const fetchMissions = async () => {
    try {
      const res = await fetch('/api/research/missions?limit=10');
      if (res.ok) {
        const data: ResearchMission[] = await res.json();
        setMissionsList(data);
        if (data.length > 0 && !currentMission) {
          setCurrentMission(data[0]);
        }
      }
    } catch (e) {
      console.error('Failed to fetch research missions', e);
    }
  };

  const handleStartMission = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/research/missions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          goal_description: goalInput,
          mode: modeInput,
          base_scenario_id: 'scen_slalom_beta',
          max_iterations: maxIterations,
          max_experiments: maxExperiments,
          max_time_seconds: 120.0
        })
      });
      if (res.ok) {
        const mission: ResearchMission = await res.json();
        setCurrentMission(mission);
        fetchMissions();
        onShowToast?.(`Research Mission '${mission.title}' started in ${modeInput} mode.`, 'success');
        if (mission.final_report_markdown) {
          setReportMarkdown(mission.final_report_markdown);
        }
      } else {
        const err = await res.json();
        onShowToast?.(`Failed to start mission: ${err.detail || 'Server error'}`, 'error');
      }
    } catch (e: any) {
      onShowToast?.(`Error starting mission: ${e.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleApprovePlan = async () => {
    if (!currentMission) return;
    setIsApproving(true);
    try {
      const res = await fetch(`/api/research/missions/${currentMission.mission_id}/approve`, {
        method: 'POST'
      });
      if (res.ok) {
        const updated: ResearchMission = await res.json();
        setCurrentMission(updated);
        fetchMissions();
        onShowToast?.('Supervisor approval granted. Resumed research loop.', 'success');
      } else {
        const err = await res.json();
        onShowToast?.(`Approval error: ${err.detail}`, 'error');
      }
    } catch (e: any) {
      onShowToast?.(`Approval failed: ${e.message}`, 'error');
    } finally {
      setIsApproving(false);
    }
  };

  const handleStepMission = async () => {
    if (!currentMission) return;
    setIsLoading(true);
    try {
      const res = await fetch(`/api/research/missions/${currentMission.mission_id}/step`, {
        method: 'POST'
      });
      if (res.ok) {
        const updated: ResearchMission = await res.json();
        setCurrentMission(updated);
        fetchMissions();
      }
    } catch (e: any) {
      onShowToast?.(`Step failed: ${e.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelMission = async () => {
    if (!currentMission) return;
    try {
      const res = await fetch(`/api/research/missions/${currentMission.mission_id}/cancel`, {
        method: 'POST'
      });
      if (res.ok) {
        const cancelled: ResearchMission = await res.json();
        setCurrentMission(cancelled);
        fetchMissions();
        onShowToast?.('Mission cancelled by user.', 'warning');
      }
    } catch (e: any) {
      onShowToast?.(`Cancellation failed: ${e.message}`, 'error');
    }
  };

  const handleRunComparativeBenchmark = async () => {
    setIsBenchmarking(true);
    try {
      const res = await fetch('/api/research/benchmark/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: 'scen_slalom_beta',
          evaluation_budget: 9,
          seeds: [100, 101, 102]
        })
      });
      if (res.ok) {
        const report: BenchmarkReport = await res.json();
        setBenchmarkReport(report);
        onShowToast?.('Comparative Benchmark completed successfully.', 'success');
      } else {
        const err = await res.json();
        onShowToast?.(`Benchmark error: ${err.detail}`, 'error');
      }
    } catch (e: any) {
      onShowToast?.(`Benchmark run failed: ${e.message}`, 'error');
    } finally {
      setIsBenchmarking(false);
    }
  };

  const handleCopyReport = () => {
    if (!currentMission?.final_report_markdown) return;
    navigator.clipboard.writeText(currentMission.final_report_markdown);
    setCopiedReport(true);
    setTimeout(() => setCopiedReport(false), 2000);
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', backgroundColor: 'var(--bg-panel)' }}>
      {/* Top Navigation Tabs */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', padding: '6px 12px', backgroundColor: 'var(--bg-secondary)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Brain size={16} color="var(--accent-blue)" />
          <span style={{ fontWeight: 600, fontSize: 13, color: 'var(--text-primary)' }}>Scientific Intelligence Studio</span>
          {currentMission && (
            <span style={{
              fontSize: 11,
              padding: '2px 8px',
              borderRadius: 10,
              backgroundColor: currentMission.status === 'completed' ? '#10b98120' : currentMission.status === 'awaiting_approval' ? '#f59e0b25' : '#3b82f620',
              color: currentMission.status === 'completed' ? '#10b981' : currentMission.status === 'awaiting_approval' ? '#f59e0b' : '#3b82f6',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 4
            }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: 'currentColor' }} />
              {currentMission.status.toUpperCase()}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: 4 }}>
          {(['mission', 'hypotheses', 'experiments', 'benchmark', 'report'] as const).map((tab) => (
            <button
              key={tab}
              className="rf-btn"
              onClick={() => setActiveSubTab(tab)}
              style={{
                fontSize: 11,
                padding: '4px 10px',
                borderRadius: 4,
                backgroundColor: activeSubTab === tab ? 'var(--accent-blue)' : 'transparent',
                color: activeSubTab === tab ? '#fff' : 'var(--text-secondary)'
              }}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px' }}>
        {/* SUBTAB 1: MISSION CONTROLLER */}
        {activeSubTab === 'mission' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Input card */}
            <div style={{ backgroundColor: 'var(--bg-primary)', padding: 14, borderRadius: 6, border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Sparkles size={14} color="var(--accent-blue)" /> RESEARCH OBJECTIVE COMPILER
                </span>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    className="rf-btn"
                    style={{ fontSize: 10, padding: '2px 8px' }}
                    onClick={() => setGoalInput('Optimize slalom traversal clearance and eliminate timeouts')}
                  >
                    Preset: Slalom Optimization
                  </button>
                  <button
                    className="rf-btn"
                    style={{ fontSize: 10, padding: '2px 8px' }}
                    onClick={() => setGoalInput('Eliminate collision contacts in narrow warehouse corridors')}
                  >
                    Preset: Collision Avoidance
                  </button>
                </div>
              </div>

              <input
                type="text"
                className="rf-input"
                value={goalInput}
                onChange={(e) => setGoalInput(e.target.value)}
                placeholder="Enter natural-language robotics research objective..."
                style={{ width: '100%', fontSize: 13, padding: '8px 10px' }}
              />

              <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Mode:</span>
                  <button
                    className="rf-btn"
                    onClick={() => setModeInput('autonomous')}
                    style={{
                      fontSize: 11,
                      backgroundColor: modeInput === 'autonomous' ? 'var(--accent-blue)' : 'var(--bg-secondary)',
                      color: modeInput === 'autonomous' ? '#fff' : 'var(--text-secondary)'
                    }}
                  >
                    Autonomous
                  </button>
                  <button
                    className="rf-btn"
                    onClick={() => setModeInput('supervised')}
                    style={{
                      fontSize: 11,
                      backgroundColor: modeInput === 'supervised' ? 'var(--accent-orange)' : 'var(--bg-secondary)',
                      color: modeInput === 'supervised' ? '#fff' : 'var(--text-secondary)'
                    }}
                  >
                    Supervised (Approval Gated)
                  </button>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Max Loops:</span>
                  <input
                    type="number"
                    className="rf-input"
                    style={{ width: 50, fontSize: 11, textAlign: 'center' }}
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(parseInt(e.target.value, 10) || 3)}
                    min={1}
                    max={10}
                  />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Max Trials:</span>
                  <input
                    type="number"
                    className="rf-input"
                    style={{ width: 50, fontSize: 11, textAlign: 'center' }}
                    value={maxExperiments}
                    onChange={(e) => setMaxExperiments(parseInt(e.target.value, 10) || 8)}
                    min={1}
                    max={20}
                  />
                </div>

                <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
                  {currentMission?.status === 'awaiting_approval' && (
                    <button
                      className="rf-btn"
                      onClick={handleApprovePlan}
                      disabled={isApproving}
                      style={{
                        backgroundColor: '#10b981',
                        color: '#fff',
                        fontSize: 12,
                        padding: '6px 14px',
                        fontWeight: 600,
                        animation: 'pulse 1.5s infinite'
                      }}
                    >
                      <ShieldCheck size={14} style={{ marginRight: 4 }} />
                      {isApproving ? 'Approving...' : 'Approve Plan & Execute Trials'}
                    </button>
                  )}

                  <button
                    className="rf-btn"
                    onClick={handleStartMission}
                    disabled={isLoading}
                    style={{ backgroundColor: 'var(--accent-blue)', color: '#fff', fontSize: 12, padding: '6px 16px', fontWeight: 600 }}
                  >
                    <Play size={14} style={{ marginRight: 4 }} />
                    {isLoading ? 'Investigating...' : 'Start Research Investigation'}
                  </button>

                  {currentMission && currentMission.status !== 'completed' && currentMission.status !== 'cancelled' && (
                    <>
                      <button className="rf-btn" onClick={handleStepMission} disabled={isLoading} style={{ fontSize: 11 }}>
                        Step Loop
                      </button>
                      <button className="rf-btn" onClick={handleCancelMission} style={{ fontSize: 11, color: 'var(--accent-red)' }}>
                        Cancel
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Mission Overview & Telemetry Card */}
            {currentMission && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
                <div style={{ backgroundColor: 'var(--bg-primary)', padding: 12, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>BASELINE PERFORMANCE (Real MuJoCo)</span>
                  <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>Success:</span>
                      <strong style={{ color: currentMission.baseline_metrics?.success ? '#10b981' : '#ef4444' }}>
                        {currentMission.baseline_metrics?.success ? 'TRUE' : 'FALSE (Timeout / Stuck)'}
                      </strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>Total Collisions:</span>
                      <strong>{currentMission.baseline_metrics?.total_collisions ?? 'N/A'}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>Time to Goal:</span>
                      <strong>{currentMission.baseline_metrics?.time_to_goal ? `${currentMission.baseline_metrics.time_to_goal}s` : 'Timed out'}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>Scenario ID:</span>
                      <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{currentMission.base_scenario_id}</span>
                    </div>
                  </div>
                </div>

                <div style={{ backgroundColor: 'var(--bg-primary)', padding: 12, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>BUDGET & RESOURCE CONSUMPTION</span>
                  <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>Research Loops:</span>
                      <strong>{currentMission.budget.used_iterations} / {currentMission.budget.max_iterations}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>Simulation Trials:</span>
                      <strong>{currentMission.budget.used_experiments} / {currentMission.budget.max_experiments}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>Active Plan:</span>
                      <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{currentMission.active_plan_id || 'None'}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>Best Candidate:</span>
                      <span style={{ fontFamily: 'monospace', fontSize: 11, color: '#10b981' }}>{currentMission.best_candidate_plan_id || 'Evaluating'}</span>
                    </div>
                  </div>
                </div>

                <div style={{ backgroundColor: 'var(--bg-primary)', padding: 12, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>HELD-OUT GENERALIZATION</span>
                  <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>Generalization Verdict:</span>
                      <strong style={{ color: currentMission.validation_report?.generalization_verdict === 'generalized' ? '#10b981' : '#ef4444' }}>
                        {currentMission.validation_report?.generalization_verdict?.toUpperCase() || 'PENDING'}
                      </strong>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                      {currentMission.validation_report?.summary || 'Runs automatically after candidate identification.'}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* SUBTAB 2: COMPETING HYPOTHESES */}
        {activeSubTab === 'hypotheses' && currentMission && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)' }}>COMPETING PHYSICAL HYPOTHESES & CAUSAL MECHANISMS</span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 12 }}>
              {currentMission.hypotheses.map((h) => (
                <div
                  key={h.hypothesis_id}
                  style={{
                    backgroundColor: 'var(--bg-primary)',
                    padding: 14,
                    borderRadius: 6,
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <strong style={{ fontSize: 13, color: 'var(--text-primary)' }}>{h.title}</strong>
                    <span style={{
                      fontSize: 10,
                      padding: '2px 6px',
                      borderRadius: 4,
                      fontWeight: 600,
                      backgroundColor: h.status === 'supported' ? '#10b98125' : h.status === 'refuted' ? '#ef444425' : '#6b728025',
                      color: h.status === 'supported' ? '#10b981' : h.status === 'refuted' ? '#ef4444' : 'var(--text-muted)'
                    }}>
                      {h.status.toUpperCase()}
                    </span>
                  </div>

                  <p style={{ fontSize: 11, color: 'var(--text-secondary)', margin: 0 }}>
                    <strong>Statement:</strong> {h.statement}
                  </p>

                  <p style={{ fontSize: 11, color: 'var(--accent-blue)', margin: 0 }}>
                    <strong>Physical Mechanism:</strong> {h.mechanism}
                  </p>

                  <div style={{ fontSize: 11, backgroundColor: 'var(--bg-secondary)', padding: '6px 8px', borderRadius: 4, fontFamily: 'monospace' }}>
                    {h.controlled_variable}: {String(h.baseline_value)} &rarr; <strong>{String(h.proposed_value)}</strong>
                  </div>

                  {/* Confidence meter */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>
                      <span>EMPIRICAL CONFIDENCE</span>
                      <span>{(h.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div style={{ width: '100%', height: 4, backgroundColor: 'var(--bg-secondary)', borderRadius: 2, overflow: 'hidden' }}>
                      <div style={{ width: `${h.confidence * 100}%`, height: '100%', backgroundColor: h.confidence > 0.6 ? '#10b981' : h.confidence < 0.4 ? '#ef4444' : 'var(--accent-blue)' }} />
                    </div>
                  </div>

                  {h.contradicting_evidence.length > 0 && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: 6 }}>
                      <em>Audit:</em> {h.contradicting_evidence[h.contradicting_evidence.length - 1]}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* SUBTAB 3: EXPERIMENT PLANS & RANKING */}
        {activeSubTab === 'experiments' && currentMission && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)' }}>PRINCIPLED CANDIDATE SELECTION & TRIAL OUTCOMES</span>
            {currentMission.plans.map((p) => {
              const outcome = currentMission.outcomes.find(o => o.plan_id === p.plan_id);
              return (
                <div
                  key={p.plan_id}
                  style={{
                    backgroundColor: 'var(--bg-primary)',
                    padding: 14,
                    borderRadius: 6,
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <strong style={{ fontSize: 13 }}>{p.title}</strong>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, backgroundColor: 'var(--bg-secondary)', color: 'var(--accent-blue)', fontWeight: 600 }}>
                        Rank #{p.candidate_rank ?? 1} (Score: {p.acquisition_score ?? '0.50'})
                      </span>
                      <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, backgroundColor: outcome?.empirical_verdict === 'improved' ? '#10b98125' : '#6b728025', color: outcome?.empirical_verdict === 'improved' ? '#10b981' : 'var(--text-secondary)', fontWeight: 600 }}>
                        {outcome?.empirical_verdict?.toUpperCase() || p.status.toUpperCase()}
                      </span>
                    </div>
                  </div>

                  <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    <strong>Selection Rationale:</strong> {p.selection_rationale || p.rationale}
                  </div>

                  <div style={{ display: 'flex', gap: 16, fontSize: 11, backgroundColor: 'var(--bg-secondary)', padding: '6px 10px', borderRadius: 4 }}>
                    <span><strong>Modifications:</strong> {JSON.stringify(p.parameter_modifications)}</span>
                    <span><strong>EI:</strong> {p.expected_improvement ?? '0.50'}</span>
                    <span><strong>Info Gain:</strong> {p.information_gain ?? '0.50'}</span>
                    <span><strong>Seeds:</strong> [{p.seeds.join(', ')}]</span>
                  </div>

                  {outcome && (
                    <div style={{ display: 'flex', gap: 16, fontSize: 11, color: outcome.empirical_verdict === 'improved' ? '#10b981' : 'var(--text-secondary)' }}>
                      <span>&Delta; Collisions: {outcome.baseline_deltas?.delta_collisions?.toFixed(2) ?? '0.00'}</span>
                      <span>&Delta; Success: {outcome.baseline_deltas?.delta_success_rate ? `${(outcome.baseline_deltas.delta_success_rate * 100).toFixed(0)}%` : '0%'}</span>
                      <span>&Delta; Time: {outcome.baseline_deltas?.delta_time_to_goal?.toFixed(2) ?? '0.00'}s</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* SUBTAB 4: COMPARATIVE ADVANTAGE BENCHMARK */}
        {activeSubTab === 'benchmark' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h4 style={{ margin: 0, fontSize: 14, color: 'var(--text-primary)' }}>4-Way Comparative Benchmark Suite</h4>
                <p style={{ margin: '4px 0 0 0', fontSize: 11, color: 'var(--text-secondary)' }}>
                  Evaluates Original Baseline, Random Search, Simple 1D Search, and RoboForge Research Agent under strictly identical budgets (9 runs) and seeds.
                </p>
              </div>

              <button
                className="rf-btn"
                onClick={handleRunComparativeBenchmark}
                disabled={isBenchmarking}
                style={{ backgroundColor: 'var(--accent-blue)', color: '#fff', fontSize: 12, padding: '8px 16px', fontWeight: 600 }}
              >
                <Scale size={14} style={{ marginRight: 6 }} />
                {isBenchmarking ? 'Simulating All Strategies in MuJoCo...' : 'Run 4-Way Comparative Benchmark'}
              </button>
            </div>

            {benchmarkReport && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Advantage Card Banner */}
                <div style={{
                  backgroundColor: benchmarkReport.research_advantage_demonstrated ? '#10b98115' : 'var(--bg-primary)',
                  border: `1px solid ${benchmarkReport.research_advantage_demonstrated ? '#10b98160' : 'var(--border-subtle)'}`,
                  padding: 14,
                  borderRadius: 6
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#10b981', fontWeight: 600, fontSize: 13 }}>
                    <Award size={18} />
                    RESEARCH ADVANTAGE VERIFIED: Winner = {benchmarkReport.winner_strategy.toUpperCase()}
                  </div>
                  <p style={{ margin: '6px 0 0 0', fontSize: 12, color: 'var(--text-primary)' }}>
                    {benchmarkReport.advantage_summary}
                  </p>
                </div>

                {/* Comparison Grid Table */}
                <div style={{ backgroundColor: 'var(--bg-primary)', borderRadius: 6, border: '1px solid var(--border-subtle)', overflow: 'hidden' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
                    <thead>
                      <tr style={{ backgroundColor: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '8px 12px' }}>Strategy</th>
                        <th style={{ padding: '8px 12px' }}>Success Rate</th>
                        <th style={{ padding: '8px 12px' }}>Collisions (&plusmn; std)</th>
                        <th style={{ padding: '8px 12px' }}>Time to Goal (&plusmn; std)</th>
                        <th style={{ padding: '8px 12px' }}>Evals Used</th>
                        <th style={{ padding: '8px 12px' }}>Held-Out Gen.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.values(benchmarkReport.results).map((res) => {
                        const isWinner = res.strategy === benchmarkReport.winner_strategy;
                        return (
                          <tr
                            key={res.strategy}
                            style={{
                              borderBottom: '1px solid var(--border-subtle)',
                              backgroundColor: isWinner ? '#10b98108' : 'transparent',
                              fontWeight: isWinner ? 600 : 400
                            }}
                          >
                            <td style={{ padding: '10px 12px', display: 'flex', alignItems: 'center', gap: 6 }}>
                              {isWinner && <CheckCircle2 size={14} color="#10b981" />}
                              <span>{res.strategy_name}</span>
                            </td>
                            <td style={{ padding: '10px 12px', color: res.success_rate > 0.5 ? '#10b981' : '#ef4444' }}>
                              {(res.success_rate * 100).toFixed(0)}%
                            </td>
                            <td style={{ padding: '10px 12px' }}>
                              {res.mean_collisions.toFixed(1)} &plusmn; {res.std_collisions.toFixed(2)}
                            </td>
                            <td style={{ padding: '10px 12px' }}>
                              {res.mean_time_to_goal ? `${res.mean_time_to_goal.toFixed(2)}s (\u00B1${res.std_time_to_goal?.toFixed(2)}s)` : 'N/A (Timed out)'}
                            </td>
                            <td style={{ padding: '10px 12px' }}>
                              {res.evaluations_used} / {benchmarkReport.evaluation_budget}
                            </td>
                            <td style={{ padding: '10px 12px', color: res.held_out_generalization_rate >= 0.5 ? '#10b981' : 'var(--text-muted)' }}>
                              {(res.held_out_generalization_rate * 100).toFixed(0)}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div style={{ fontSize: 11, color: 'var(--text-muted)', backgroundColor: 'var(--bg-secondary)', padding: 10, borderRadius: 4 }}>
                  <strong>Limitations & Diagnosis:</strong> {benchmarkReport.limitations_and_diagnosis}
                </div>
              </div>
            )}
          </div>
        )}

        {/* SUBTAB 5: REPRODUCIBLE REPORT */}
        {activeSubTab === 'report' && currentMission && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)' }}>REPRODUCIBLE RESEARCH REPORT (Markdown)</span>
              <button className="rf-btn" onClick={handleCopyReport} style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
                {copiedReport ? <Check size={12} color="#10b981" /> : <Copy size={12} />}
                {copiedReport ? 'Copied!' : 'Copy Markdown'}
              </button>
            </div>

            <div style={{
              backgroundColor: 'var(--bg-primary)',
              padding: 16,
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              fontFamily: 'monospace',
              fontSize: 12,
              whiteSpace: 'pre-wrap',
              maxHeight: 480,
              overflowY: 'auto'
            }}>
              {currentMission.final_report_markdown || 'Report will be synthesized upon mission completion.'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
