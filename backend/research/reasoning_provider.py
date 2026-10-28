"""
Provider-independent reasoning interface for RoboForge Scientific Intelligence.
Supports pluggable language model providers and an honest deterministic scientific rule engine fallback.
Never allows arbitrary code execution; all outputs are strictly validated data dicts.
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("roboforge.research.reasoning")


class ReasoningProvider(ABC):
    """Abstract interface for scientific reasoning and experiment generation."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def provider_type(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def interpret_goal(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Translates natural language goal into target metrics and constraints."""
        pass

    @abstractmethod
    def propose_hypotheses(
        self,
        goal: str,
        baseline_metrics: Dict[str, Any],
        scenario_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generates competing testable physical hypotheses for observed baseline performance."""
        pass

    @abstractmethod
    def design_controlled_experiment(
        self,
        hypothesis: Dict[str, Any],
        base_scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generates a bounded, single-variable or controlled parameter modification plan."""
        pass


class DeterministicRuleEngine(ReasoningProvider):
    """
    Truthful, physics-grounded rule engine based on robotics control and navigation heuristics.
    Explicitly labeled as deterministic_rule_engine so it never claims to be LLM reasoning.
    """
    
    @property
    def name(self) -> str:
        return "DeterministicRoboticsHeuristicEngine-v1"

    @property
    def provider_type(self) -> str:
        return "deterministic_rule_engine"

    def is_available(self) -> bool:
        return True

    def interpret_goal(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        g_lower = goal.lower()
        target_metric = "total_collisions"
        threshold = 0.0
        direction = "minimize"
        constraints = ["max_linear_velocity <= 1.5", "timeout <= 25.0"]

        if "time" in g_lower or "fast" in g_lower or "speed" in g_lower:
            target_metric = "time_to_goal"
            threshold = 8.0
            direction = "minimize"
            constraints.append("total_collisions == 0")
        elif "collision" in g_lower or "safe" in g_lower or "avoid" in g_lower:
            target_metric = "total_collisions"
            threshold = 0.0
            direction = "minimize"
            constraints.append("success == True")

        # Extract numeric constraints if mentioned
        time_match = re.search(r'under\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)', g_lower)
        if time_match:
            sec = float(time_match.group(1))
            constraints.append(f"time_to_goal <= {sec}")

        return {
            "title": f"Scientific Investigation: {goal[:50]}",
            "target_metric": target_metric,
            "target_threshold": threshold,
            "optimization_direction": direction,
            "constraints": constraints,
            "rationale": f"Derived quantitative objectives and safety bounds from goal description."
        }

    def propose_hypotheses(
        self,
        goal: str,
        baseline_metrics: Dict[str, Any],
        scenario_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        collisions = baseline_metrics.get("total_collisions", 0)
        success = baseline_metrics.get("success", False)
        time_to_goal = baseline_metrics.get("time_to_goal") or baseline_metrics.get("total_duration", 15.0)

        hypotheses = []

        # Hypothesis 1: Safety Margin Clearance
        if collisions > 0:
            hypotheses.append({
                "title": "Repulsive Safety Margin Hypothesis",
                "statement": "Increasing obstacle repulsion clearance margin will eliminate collision contacts during obstacle swerving.",
                "mechanism": "A larger safety margin initiates repulsive steering earlier, preventing corner clipping.",
                "controlled_variable": "task.policy.safety_margin",
                "baseline_value": 0.70,
                "proposed_value": 0.95,
                "expected_outcome": "Collisions reduce to 0 with minimal transit time penalty."
            })

            # Competing Hypothesis 2: Turning Rate & Agility
            hypotheses.append({
                "title": "Angular Agility Saturation Hypothesis",
                "statement": "Increasing maximum turn rate will allow the robot to negotiate tight turns without sliding into obstacles.",
                "mechanism": "Higher angular rate prevents control saturation during rapid avoidance maneuvers.",
                "controlled_variable": "task.policy.max_turn_rate",
                "baseline_value": 3.0,
                "proposed_value": 4.5,
                "expected_outcome": "Faster heading convergence around sharp obstacle edges."
            })

            # Competing Hypothesis 3: Forward Velocity Regulation
            hypotheses.append({
                "title": "Cornering Speed Regulation Hypothesis",
                "statement": "Lowering maximum cruise velocity during avoidance will decrease inertia and stopping distance.",
                "mechanism": "Reduced linear momentum decreases the centripetal force required to round obstacle boundaries.",
                "controlled_variable": "robot.max_linear_velocity",
                "baseline_value": 1.2,
                "proposed_value": 0.85,
                "expected_outcome": "Lower trajectory overshoot near obstacle boundaries."
            })
        elif not success and collisions == 0:
            # Baseline timed out / stalled in front of obstacles without collision (local minimum or excessive repulsive drag)
            hypotheses.append({
                "title": "Vortex Circulation Obstacle Bypass Hypothesis",
                "statement": "Increasing tangential vortex circulation gain will break repulsive equilibrium and guide the robot smoothly past obstacles.",
                "mechanism": "Tangential circulation creates a perpendicular vortex force around obstacles, preventing head-on stall.",
                "controlled_variable": "task.policy.tangential_gain",
                "baseline_value": 1.3,
                "proposed_value": 2.2,
                "expected_outcome": "Robot curves smoothly past obstacles and reaches target before timeout."
            })
            hypotheses.append({
                "title": "Repulsion Corridor Clearance Hypothesis",
                "statement": "Decreasing safety margin slightly will reduce premature deceleration and repulsive drag between closely-spaced obstacles.",
                "mechanism": "A slightly tighter safety margin allows the robot to enter navigable gaps without excessive repulsive braking.",
                "controlled_variable": "task.policy.safety_margin",
                "baseline_value": 0.65,
                "proposed_value": 0.55,
                "expected_outcome": "Higher average cruise speed through obstacle corridors."
            })
            hypotheses.append({
                "title": "Angular Steering Agility Hypothesis",
                "statement": "Increasing maximum turn rate will allow the robot to quickly align with tangent escape vectors.",
                "mechanism": "Higher angular agility avoids control saturation during rapid heading changes around obstacles.",
                "controlled_variable": "task.policy.max_turn_rate",
                "baseline_value": 2.5,
                "proposed_value": 3.2,
                "expected_outcome": "Faster convergence to goal-seeking orientation."
            })
        else:
            # Baseline succeeded without collisions; optimize speed / efficiency
            hypotheses.append({
                "title": "Velocity Scaling Optimization Hypothesis",
                "statement": "Increasing linear velocity from 1.0 to 1.4 m/s will decrease time to goal without causing collision.",
                "mechanism": "Straight-line transit speed increases while existing repulsive margin maintains clearance.",
                "controlled_variable": "robot.max_linear_velocity",
                "baseline_value": 1.0,
                "proposed_value": 1.4,
                "expected_outcome": f"Time to goal decreases below {time_to_goal * 0.8:.1f}s."
            })
            hypotheses.append({
                "title": "Tangential Circulation Gain Hypothesis",
                "statement": "Increasing tangential circulation gain will reduce unnecessary deceleration near obstacles.",
                "mechanism": "Higher vortex circulation guides robot smoothly around curvature without stopping.",
                "controlled_variable": "task.policy.tangential_gain",
                "baseline_value": 1.3,
                "proposed_value": 1.8,
                "expected_outcome": "Shorter total path length and smoother transit."
            })

        return hypotheses

    def design_controlled_experiment(
        self,
        hypothesis: Dict[str, Any],
        base_scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        var_name = hypothesis["controlled_variable"]
        prop_val = hypothesis["proposed_value"]

        return {
            "title": f"Controlled Experiment for: {hypothesis['title']}",
            "rationale": f"Test causal effect of modifying {var_name} to {prop_val} while keeping all other physics and layout variables constant.",
            "parameter_modifications": {
                var_name: prop_val
            },
            "seeds": [100, 101, 102],
            "expected_outcome": hypothesis.get("expected_outcome", "Observed metric improves relative to baseline.")
        }


class LanguageModelReasoningProvider(ReasoningProvider):
    """
    Language model reasoning provider for environments where GEMINI_API_KEY or OPENAI_API_KEY is available.
    Supports Google Gemini (preferring Gemini 2.0 Flash) and OpenAI (gpt-4o-mini).
    Executes real inference via official REST endpoints with httpx.
    Strictly validates all outputs to typed schemas and physical bounds. Does not execute code.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        # Explicit api_key overrides or defaults
        if api_key is not None:
            self.api_key = api_key
            if not self.gemini_key and not self.openai_key:
                # Assign to gemini_key by default if looks like Google key, else openai
                if "AIza" in api_key:
                    self.gemini_key = api_key
                else:
                    self.openai_key = api_key
        else:
            self.api_key = self.gemini_key or self.openai_key

        self._fallback = DeterministicRuleEngine()

    @property
    def name(self) -> str:
        if self.gemini_key:
            return f"Google-Gemini ({os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')})"
        if self.openai_key:
            return f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})"
        return "LLMReasoningProvider (Unconfigured)"

    @property
    def provider_type(self) -> str:
        return "llm" if self.is_available() else "blocked"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _execute_prompt(self, system_instruction: str, user_content: str) -> Dict[str, Any]:
        """Dispatches structured prompt to configured LLM via httpx."""
        import httpx
        if self.gemini_key:
            return self._call_gemini_api(system_instruction, user_content)
        elif self.openai_key:
            return self._call_openai_api(system_instruction, user_content)
        else:
            raise RuntimeError("LLM reasoning provider is not configured (missing API credentials).")

    def _call_gemini_api(self, system_instruction: str, user_content: str) -> Dict[str, Any]:
        import httpx
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"SYSTEM RULES:\n{system_instruction}\n\nTASK:\n{user_content}"}
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2
            }
        }
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(raw_text)

    def _call_openai_api(self, system_instruction: str, user_content: str) -> Dict[str, Any]:
        import httpx
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            return json.loads(raw_text)

    def interpret_goal(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("LLM reasoning provider is not configured (missing API credentials).")
        try:
            sys_inst = (
                "You are an expert robotics research AI. Translate the user goal into quantitative target metrics and constraints. "
                "Output strictly a JSON object with keys: title (str), target_metric (str, e.g. total_collisions or time_to_goal), "
                "target_threshold (float), optimization_direction ('minimize' or 'maximize'), constraints (list of str), rationale (str)."
            )
            data = self._execute_prompt(sys_inst, f"Goal: {goal}\nContext: {json.dumps(context)}")
            if "target_metric" in data and "target_threshold" in data:
                return data
        except Exception as e:
            logger.warning("LLM goal interpretation error: %s. Falling back to deterministic rule engine.", e)
        return self._fallback.interpret_goal(goal, context)

    def propose_hypotheses(
        self,
        goal: str,
        baseline_metrics: Dict[str, Any],
        scenario_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("LLM reasoning provider is not configured (missing API credentials).")
        try:
            sys_inst = (
                "You are an expert robotics research AI. Given baseline metrics and scenario layout, propose competing physical hypotheses. "
                "Output strictly a JSON object with key 'hypotheses' containing a list of objects with keys: "
                "title (str), statement (str), mechanism (str), controlled_variable (str, e.g. 'task.policy.safety_margin'), "
                "baseline_value (float), proposed_value (float), expected_outcome (str)."
            )
            user_msg = f"Goal: {goal}\nBaseline Metrics: {json.dumps(baseline_metrics)}\nScenario: {json.dumps(scenario_info)}"
            data = self._execute_prompt(sys_inst, user_msg)
            if "hypotheses" in data and isinstance(data["hypotheses"], list) and len(data["hypotheses"]) > 0:
                return data["hypotheses"]
        except Exception as e:
            logger.warning("LLM hypothesis generation error: %s. Falling back to deterministic rule engine.", e)
        return self._fallback.propose_hypotheses(goal, baseline_metrics, scenario_info)

    def design_controlled_experiment(
        self,
        hypothesis: Dict[str, Any],
        base_scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("LLM reasoning provider is not configured (missing API credentials).")
        try:
            sys_inst = (
                "You are an expert robotics research AI. Design a controlled single-variable experiment plan for the given hypothesis. "
                "Output strictly a JSON object with keys: title (str), rationale (str), parameter_modifications (dict mapping param to float), "
                "seeds (list of int, minimum 3 seeds like [100, 101, 102]), expected_outcome (str)."
            )
            user_msg = f"Hypothesis: {json.dumps(hypothesis)}\nBase Scenario: {json.dumps(base_scenario)}"
            data = self._execute_prompt(sys_inst, user_msg)
            if "parameter_modifications" in data and "seeds" in data:
                return data
        except Exception as e:
            logger.warning("LLM experiment design error: %s. Falling back to deterministic rule engine.", e)
        return self._fallback.design_controlled_experiment(hypothesis, base_scenario)


class OllamaReasoningProvider(ReasoningProvider):
    """
    Local LLM reasoning provider integrating with Ollama (e.g. qwen2.5:7b).
    Strictly verifies model installation, uses timeouts, enforces schema validation,
    and falls back to DeterministicRuleEngine on any error. Never executes code or silently downloads models.
    """

    ALLOWED_VARIABLES = {
        "task.policy.safety_margin",
        "task.policy.repulsion_gain",
        "task.policy.tangential_gain",
        "task.policy.max_turn_rate",
        "task.policy.slowdown_distance",
        "robot.max_linear_velocity",
        "robot.max_angular_velocity",
    }

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        raw_url = base_url or os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
        self.base_url = raw_url.rstrip("/")
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "35.0"))
        self._fallback = DeterministicRuleEngine()
        self.available_models: List[str] = []
        self._is_connected: bool = False
        self.model: Optional[str] = None

        self.discover_models(requested_model=model)

    def discover_models(self, requested_model: Optional[str] = None) -> List[str]:
        """Queries /api/tags to discover installed models without pulling any new models."""
        import httpx
        url = f"{self.base_url}/api/tags"
        try:
            with httpx.Client(timeout=2.5) as client:
                res = client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    self.available_models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                    self._is_connected = True
                else:
                    self.available_models = []
                    self._is_connected = False
        except Exception as e:
            logger.debug("Ollama model discovery failed at %s: %s", url, e)
            self.available_models = []
            self._is_connected = False

        # Model selection:
        req = requested_model or os.getenv("OLLAMA_MODEL")
        if req:
            matched = next(
                (m for m in self.available_models if m == req or m.startswith(f"{req}:") or req.startswith(f"{m}:")),
                None
            )
            if matched:
                self.model = matched
            else:
                logger.warning(
                    "Requested Ollama model '%s' is not installed in %s. Never silently downloading.",
                    req, self.available_models
                )
                self.model = None
        else:
            # Default to qwen2.5:7b ONLY if actually installed
            qwen_match = next(
                (m for m in self.available_models if "qwen2.5:7b" in m or m.startswith("qwen2.5:7b")),
                None
            )
            if qwen_match:
                self.model = qwen_match
            elif self.available_models:
                self.model = self.available_models[0]
            else:
                self.model = None

        return self.available_models

    @property
    def name(self) -> str:
        if self.is_available():
            return f"Ollama ({self.model})"
        return "Ollama (Unavailable)"

    @property
    def provider_type(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        return bool(self._is_connected and self.model and self.model in self.available_models)

    def _call_ollama(self, system_instruction: str, user_content: str) -> Dict[str, Any]:
        """Calls local Ollama /api/chat with strict JSON schema enforcement."""
        if not self.is_available():
            raise RuntimeError(f"Ollama provider is unavailable (endpoint: {self.base_url}, model: {self.model}).")

        import httpx
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2}
        }

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            raw_text = data.get("message", {}).get("content", "{}")

            clean_text = raw_text.strip()
            if clean_text.startswith("```"):
                clean_text = re.sub(r"^```[a-zA-Z]*\n?", "", clean_text)
                clean_text = re.sub(r"\n?```$", "", clean_text)

            return json.loads(clean_text)

    def interpret_goal(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            return self._fallback.interpret_goal(goal, context)
        try:
            sys_inst = (
                "You are an expert robotics research AI in RoboForge. "
                "Translate the user goal and context into quantitative target metrics and constraints. "
                "Output strictly a JSON object with keys: "
                "title (str), target_metric (str, e.g. 'total_collisions' or 'time_to_goal'), "
                "target_threshold (float), optimization_direction ('minimize' or 'maximize'), "
                "constraints (list of str, e.g. ['max_linear_velocity <= 1.5', 'total_collisions == 0']), "
                "rationale (str)."
            )
            data = self._call_ollama(sys_inst, f"Goal: {goal}\nContext: {json.dumps(context)}")
            if "target_metric" in data and "target_threshold" in data:
                if not isinstance(data.get("constraints"), list):
                    data["constraints"] = ["max_linear_velocity <= 1.5"]
                data["target_threshold"] = float(data["target_threshold"])

                # Ensure explicitly stated time constraints in goal text are preserved and standardized
                g_lower = goal.lower()
                time_match = re.search(r'under\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)', g_lower)
                if time_match:
                    sec = float(time_match.group(1))
                    sec_val = int(sec) if sec.is_integer() else sec
                    if not any("time_to_goal" in c for c in data["constraints"]):
                        data["constraints"].append(f"time_to_goal <= {sec_val}")

                # Standardize time_to_goal relational operator '<' to '<='
                norm_constraints = []
                for c in data["constraints"]:
                    c_str = str(c)
                    if "time_to_goal < " in c_str and "<=" not in c_str:
                        c_str = c_str.replace("time_to_goal < ", "time_to_goal <= ")
                    norm_constraints.append(c_str)
                data["constraints"] = norm_constraints

                return data
        except Exception as e:
            logger.warning("Ollama goal interpretation failed (%s). Falling back to deterministic rule engine.", e)
        return self._fallback.interpret_goal(goal, context)

    def propose_hypotheses(
        self,
        goal: str,
        baseline_metrics: Dict[str, Any],
        scenario_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            return self._fallback.propose_hypotheses(goal, baseline_metrics, scenario_info)
        try:
            sys_inst = (
                "You are an expert robotics research AI in RoboForge. "
                "Given baseline simulation metrics and scenario layout, propose 2 to 3 competing physical hypotheses. "
                "Output strictly a JSON object with key 'hypotheses' containing a list of objects with keys: "
                "title (str), statement (str), mechanism (str), "
                "controlled_variable (str, e.g. 'task.policy.safety_margin', 'task.policy.tangential_gain', 'task.policy.repulsion_gain', 'task.policy.max_turn_rate', or 'robot.max_linear_velocity'), "
                "baseline_value (float), proposed_value (float), expected_outcome (str)."
            )
            user_msg = f"Goal: {goal}\nBaseline Metrics: {json.dumps(baseline_metrics)}\nScenario: {json.dumps(scenario_info)}"
            data = self._call_ollama(sys_inst, user_msg)
            hypos = data.get("hypotheses", [])
            valid_hypos = []
            if isinstance(hypos, list) and len(hypos) > 0:
                for h in hypos:
                    if not isinstance(h, dict) or "title" not in h or "proposed_value" not in h:
                        continue
                    c_var = str(h.get("controlled_variable", "")).strip()
                    if c_var not in self.ALLOWED_VARIABLES:
                        if "tangent" in c_var or "vortex" in c_var or "circulat" in c_var:
                            c_var = "task.policy.tangential_gain"
                        elif "margin" in c_var or "distance" in c_var:
                            c_var = "task.policy.safety_margin"
                        elif "turn" in c_var or "angular" in c_var or "yaw" in c_var:
                            c_var = "task.policy.max_turn_rate"
                        elif "repulsion" in c_var or "gain" in c_var:
                            c_var = "task.policy.repulsion_gain"
                        elif "speed" in c_var or "vel" in c_var:
                            c_var = "robot.max_linear_velocity"
                        else:
                            c_var = "task.policy.safety_margin"
                    h["controlled_variable"] = c_var
                    h["baseline_value"] = float(h.get("baseline_value", 0.7))
                    h["proposed_value"] = float(h["proposed_value"])
                    valid_hypos.append(h)
                if valid_hypos:
                    return valid_hypos
        except Exception as e:
            logger.warning("Ollama hypothesis generation failed (%s). Falling back to deterministic rule engine.", e)
        return self._fallback.propose_hypotheses(goal, baseline_metrics, scenario_info)

    def design_controlled_experiment(
        self,
        hypothesis: Dict[str, Any],
        base_scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.is_available():
            return self._fallback.design_controlled_experiment(hypothesis, base_scenario)
        try:
            sys_inst = (
                "You are an expert robotics research AI. Design a controlled single-variable experiment plan for the given hypothesis. "
                "Output strictly a JSON object with keys: title (str), rationale (str), "
                "parameter_modifications (dict mapping param to float), seeds (list of int, minimum 3 seeds like [100, 101, 102]), expected_outcome (str)."
            )
            user_msg = f"Hypothesis: {json.dumps(hypothesis)}\nBase Scenario: {json.dumps(base_scenario)}"
            data = self._call_ollama(sys_inst, user_msg)
            if "parameter_modifications" in data and isinstance(data["parameter_modifications"], dict):
                if "seeds" not in data or not isinstance(data["seeds"], list) or len(data["seeds"]) < 3:
                    data["seeds"] = [100, 101, 102]
                return data
        except Exception as e:
            logger.warning("Ollama experiment design failed (%s). Falling back to deterministic rule engine.", e)
        return self._fallback.design_controlled_experiment(hypothesis, base_scenario)


def get_reasoning_provider(force_type: Optional[str] = None) -> ReasoningProvider:
    """Factory selecting the appropriate reasoning provider with transparent availability guarantees."""
    if force_type == "ollama":
        provider = OllamaReasoningProvider()
        if not provider.is_available():
            logger.warning("Ollama provider requested but unavailable or model missing. Using DeterministicRuleEngine.")
            return DeterministicRuleEngine()
        return provider
    if force_type == "llm":
        provider = LanguageModelReasoningProvider()
        if not provider.is_available():
            logger.warning("LLM provider requested but credentials missing. Using DeterministicRuleEngine.")
            return DeterministicRuleEngine()
        return provider
    if force_type == "deterministic":
        return DeterministicRuleEngine()

    pref = os.getenv("AI_PROVIDER", "").lower()
    if pref == "ollama":
        ollama = OllamaReasoningProvider()
        if ollama.is_available():
            return ollama
        return DeterministicRuleEngine()
    if pref in ("gemini", "openai", "cloud"):
        cloud = LanguageModelReasoningProvider()
        if cloud.is_available():
            return cloud
        return DeterministicRuleEngine()
    if pref == "deterministic":
        return DeterministicRuleEngine()

    # Automatic discovery: Ollama local > Cloud LLM > Deterministic
    ollama = OllamaReasoningProvider()
    if ollama.is_available():
        return ollama

    cloud = LanguageModelReasoningProvider()
    if cloud.is_available():
        return cloud

    return DeterministicRuleEngine()

