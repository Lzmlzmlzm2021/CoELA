"""Single-call decentralized ReAct using CoELA's native candidate compiler."""
import copy
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI
from LLM.LLM import LLM, normalize_agent_role


POLICY_VERSION = 'simple_react_message_cooldown_ascii_v1'
REASON_PATTERN = r'^[ -~]{1,120}$'
MESSAGE_PATTERN = r'^[ -~]{1,240}$'


SYSTEM = """You are one robot in a cooperative transport task. Choose ONE next
high-level action from the current candidate list. Your observations and remembered
objects are local and may be stale. Messages are teammate reports, not verified
physical state. Carriers can hold objects/containers and deliver payloads to the
bed; a scout can only navigate, explore and communicate. The native executor
handles navigation and manipulation over multiple simulation frames.
Use actual held objects, current room and low-level status as feedback. A previous
plan is an attempt, not proof of completion. local_executor_records are the legacy
agent's bookkeeping, NOT evaluator-confirmed delivery. Do not infer success merely
because an object disappeared. Cooperate through explicit message actions.
Return JSON with ALL THREE fields: {"reason":"brief rationale","action_id":"a0","message":"text for teammate"}.
The message field is REQUIRED on every turn. For physical actions use "-"; for
send a message, write the actual useful message to the teammate in printable ASCII.
Use printable ASCII only: reason 1-120 characters and message 1-240 characters.
After sending a message, communication is on cooldown until you attempt a native
non-message action. While it is on cooldown and other actions exist, the message
candidate is withheld. The communication_cooldown context reports this state.
An attempted action need not succeed to restore communication. Communicate
new observations or a concrete coordination request, not repeated readiness.
Do not invent candidate IDs. Select exactly one action, not a multi-step plan."""


def json_safe(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


def load_config(path):
    if not path:
        return {}
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    key = config.get('api_key') or config.get('apikey') or config.get('token')
    if config.get('api_key_file'):
        if key:
            raise ValueError('Use an API key OR api_key_file, not both')
        key_path = Path(os.path.expandvars(config['api_key_file'])).expanduser()
        if not key_path.is_absolute():
            key_path = path.parent / key_path
        key = key_path.read_text(encoding='utf-8').strip()
    return dict(model=config.get('model') or config.get('lm_id'),
                base_url=config.get('base_url') or config.get('api_base'), api_key=key)


class ReActLLM(LLM):
    """Reuse get_available_plans, never instantiate CoELA/V4 model planners."""
    def __init__(self, source, lm_id, prompt_template_path, communication, cot,
                 sampling_parameters, agent_id, agent_role='human', opponent_role='human'):
        if source != 'openai':
            raise ValueError('Simple ReAct requires an OpenAI-compatible endpoint')
        args = sampling_parameters
        config = load_config(getattr(args, 'react_api_config', None))
        self.lm_id = config.get('model') or lm_id
        self.client = OpenAI(api_key=config.get('api_key') or os.getenv('OPENAI_API_KEY', 'EMPTY'),
                             base_url=config.get('base_url') or os.getenv('OPENAI_BASE_URL'),
                             timeout=getattr(args, 'react_timeout', 180), max_retries=0)
        self.max_tokens = getattr(args, 'react_max_tokens', 256)
        self.history_limit = getattr(args, 'react_history', 8)
        self.max_retries = getattr(args, 'react_max_retries', 2)
        self.temperature = getattr(args, 'react_temperature', 0.0)
        self.agent_id = agent_id
        self.agent_role = normalize_agent_role(agent_role)
        self.opponent_role = normalize_agent_role(opponent_role)
        self.communication = communication
        self.peer_consult_protocol = None
        self.peer_decision_card = None
        self.audit_path = None
        self.reset([], {})

    def shared_delivery_target(self):
        return None  # Communication does not grant privileged executor coordinates.

    def reset(self, rooms_name, goal_objects):
        self.rooms = list(rooms_name)
        self.goal_objects = copy.deepcopy(goal_objects)
        self.history = []
        self.previous = None
        self.metrics = Counter()
        self.consecutive_invalid = 0
        self.communication_cooldown = False
        self.pending_non_message_frame = None
        self.peer_decision_card = None
        self.pending_perception_events = []

    def set_output_dir(self, output_dir):
        self.audit_path = Path(output_dir) / f'react_agent_{self.agent_id}.jsonl'

    def _log(self, row):
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(json_safe(row), ensure_ascii=False) + '\n')

    @staticmethod
    def parse(raw, plans):
        text = raw.strip()
        if text.startswith('```') and text.endswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        decision = json.loads(text)
        if not isinstance(decision, dict) or set(decision) != {'reason', 'action_id', 'message'}:
            raise ValueError('Expected exactly reason, action_id and message fields')
        for name, pattern in (('reason', REASON_PATTERN), ('message', MESSAGE_PATTERN)):
            value = decision[name]
            if not isinstance(value, str) or re.fullmatch(pattern, value) is None or not value.strip():
                maximum = 120 if name == 'reason' else 240
                raise ValueError(f'{name} must contain 1 to {maximum} printable ASCII characters')
        key = decision.get('action_id')
        choices = {f'a{i}': plan for i, plan in enumerate(plans)}
        if not isinstance(key, str) or key not in choices:
            raise ValueError('action_id is not a current candidate')
        plan = choices[key]
        if plan == 'send a message':
            message = decision.get('message')
            # Native executor joins words; normalize the exact logged/executed payload.
            message = ' '.join(message.split())
            decision['message'] = message
            plan = 'send a message: ' + message
        return plan, decision

    @staticmethod
    def response_format(plans):
        return {"type": "json_schema", "json_schema": {
            "name": "tdw_react_action", "strict": True, "schema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "reason": {"type": "string", "pattern": REASON_PATTERN},
                    "action_id": {"type": "string", "enum": [f"a{i}" for i in range(len(plans))]},
                    "message": {"type": "string", "pattern": MESSAGE_PATTERN}},
                "required": ["reason", "action_id", "message"]}}}

    def run(self, current_step, current_room, rooms_explored, holding_objects,
            satisfied, object_list, obj_per_room, action_history, dialogue_history,
            opponent_grabbed_objects=None, opponent_last_room=None, feedback=None):
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        # The executor can reject a stale plan and replan within the same frame.
        # Only a later observed non-message action releases communication; merely
        # selecting a physical plan (or making an invalid decision) does not.
        last_action = (feedback or {}).get('last_low_level_action') or {}
        if (self.communication_cooldown and self.pending_non_message_frame is not None
                and current_step > self.pending_non_message_frame
                and last_action.get('type') in (0, 1, 2, 3, 4, 5, 7, 8)):
            self.communication_cooldown = False
            self.pending_non_message_frame = None
        _, _, plans = self.get_available_plans(include_message=self.communication)
        non_message_plans = [plan for plan in plans if plan != 'send a message']
        suppress_message = bool(self.communication_cooldown and non_message_plans
                                and 'send a message' in plans)
        if suppress_message:
            plans = non_message_plans
            self.metrics['message_cooldown_decisions'] += 1
        cooldown = dict(active=self.communication_cooldown,
                        message_suppressed=suppress_message,
                        awaiting_execution_since_frame=self.pending_non_message_frame,
                        release_condition='a later observed native non-message action')
        if self.previous is not None:
            self.history.append(dict(self.previous, observed_at_frame=current_step,
                feedback=json_safe(feedback), held_objects=json_safe(holding_objects),
                current_room=current_room))
            self.history = self.history[-self.history_limit:]
        context = json_safe(dict(goal_objects=self.goal_objects, agent_id=self.agent_id,
            role=self.agent_role, teammate_role=self.opponent_role,
            frame=current_step, current_room=current_room, rooms_explored=rooms_explored,
            held_objects=holding_objects, known_objects=object_list,
            remembered_objects_by_room=obj_per_room, local_executor_records=satisfied,
            last_observed_teammate_payload=opponent_grabbed_objects,
            last_observed_teammate_room=opponent_last_room,
            messages=dialogue_history[-self.history_limit:], recent_history=self.history,
            executor_feedback=feedback,
            policy_version=POLICY_VERSION, communication_cooldown=cooldown,
            available_actions={f'a{i}': p for i, p in enumerate(plans)}))
        messages = [dict(role='system', content=SYSTEM),
                    dict(role='user', content=json.dumps(context, ensure_ascii=False))]
        raw, usage, decision, error = None, None, None, None
        if plans:
            for attempt in range(self.max_retries + 1):
                self.metrics['requests'] += 1
                try:
                    response = self.client.chat.completions.create(model=self.lm_id,
                        messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
                        response_format=self.response_format(plans))
                    raw = response.choices[0].message.content or ''
                    usage = response.usage.model_dump() if response.usage else None
                    if usage:
                        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                            self.metrics[key] += usage.get(key, 0) or 0
                    else:
                        self.metrics['usage_missing'] += 1
                    break
                except Exception as exc:
                    self._log(dict(event='api_error', frame=current_step,
                                   attempt=attempt, error_type=type(exc).__name__))
                    if attempt == self.max_retries:
                        raise RuntimeError(f'ReAct model request failed: {type(exc).__name__}') from exc
                    self.metrics['api_retries'] += 1
                    time.sleep(min(2 ** attempt, 4))
            try:
                plan, decision = self.parse(raw, plans)
                self.consecutive_invalid = 0
                if plan.startswith('send a message:'):
                    self.communication_cooldown = True
                    self.pending_non_message_frame = None
                elif self.communication_cooldown:
                    self.pending_non_message_frame = current_step
            except (ValueError, TypeError) as exc:
                error = str(exc)
                self.metrics['invalid_decisions'] += 1
                self.consecutive_invalid += 1
                self.pending_non_message_frame = None
                plan = 'wait'  # Explicit, logged native one-frame no-op, no heuristic action.
        else:
            plan, error = 'wait', 'no_native_candidates'
            self.metrics['no_candidate_decisions'] += 1
            self.pending_non_message_frame = None
        self.previous = dict(frame=current_step, plan=plan, invalid_decision=error)
        info = dict(plan=plan, decision=decision, num_available_actions=len(plans),
                    raw_response=raw, usage=usage, metrics=dict(self.metrics), error=error,
                    communication_cooldown=dict(cooldown,
                        active_after_decision=self.communication_cooldown,
                        awaiting_execution_after_frame=self.pending_non_message_frame))
        self._log(dict(frame=current_step, model=self.lm_id,
                       settings=dict(policy_version=POLICY_VERSION,
                                     history=self.history_limit, max_tokens=self.max_tokens,
                                     temperature=self.temperature), messages=messages, **info))
        if self.consecutive_invalid >= 3:
            raise RuntimeError('Three consecutive invalid structured decisions; inspect ReAct audit')
        return plan, info
