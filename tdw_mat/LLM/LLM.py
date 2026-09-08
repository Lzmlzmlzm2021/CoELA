import random
import re
import os
from typing import List
import json
import pandas as pd
import backoff
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from openai import OpenAIError

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

_SCOUT_ROLE_ALIASES = {"box", "dog", "scout", "box_scout", "box-scout"}

_MANIPULATION_PATTERN = (
    r"(?:grab(?:bed|bing|s)?|grasp(?:ed|ing|s)?|"
    r"pick(?:ed|ing)?\s+up|carr(?:y|ies|ied|ying)|"
    r"deliver(?:ed|ing|s)?|transport(?:ed|ing|s)?|"
    r"put(?:ting|s)?|drop(?:ped|ping|s)?|place(?:d|ing|s)?|"
    r"hold(?:ing|s)?|held)"
)


def _communication_text(message):
    """Normalize punctuation only for deterministic capability checks."""
    if not isinstance(message, str):
        return ""
    # Keep this source fragment ASCII-safe so Windows PowerShell 5.1 can't
    # rewrite smart punctuation while inspecting or patching it.
    return (message.strip().strip('"').replace("\u2018", "'")
            .replace("\u2019", "'").replace("\u2013", ";")
            .replace("\u2014", ";"))


def guard_communication_message(message, sender_role, receiver_role):
    """Reject messages that assign manipulation to a non-manipulating Scout.

    This is a capability guard, not a semantic quality classifier.  It only
    rejects direct imperatives to a Scout or first-person Scout claims of
    physical manipulation; ordinary reports such as "I recommend you deliver"
    remain valid when addressed to the Human.

    Returns ``(accepted_message, decision)`` where ``accepted_message`` is
    ``None`` on rejection and ``decision`` is safe to serialize in run logs.
    """
    sender_role = normalize_agent_role(sender_role)
    receiver_role = normalize_agent_role(receiver_role)
    text = _communication_text(message)
    decision = {
        "accepted": True,
        "reason": "accepted",
        "sender_role": sender_role,
        "receiver_role": receiver_role,
    }
    if not text:
        decision.update(accepted=False, reason="empty_message")
        return None, decision

    # An imperative clause in a Human->Scout message has the Scout as its
    # implicit subject. Smart dashes are normalized to semicolons above, which
    # catches the observed failure: "... apples there; grab one and deliver".
    if receiver_role == "scout":
        clauses = re.split(r"[\n.!?;]+", text.lower())
        imperative = re.compile(
            rf"^\s*(?:(?:bob|scout|box\s+scout|dog)\b\s*[:,]?\s*)?"
            rf"(?:please\s+)?(?:go\s+)?{_MANIPULATION_PATTERN}\b")
        compound_imperative = re.compile(
            rf"^\s*(?:(?:bob|scout|box\s+scout|dog)\b\s*[:,]?\s*)?"
            rf"(?:please\s+)?(?:go\s+)?(?:find|locate|approach|navigate|"
            rf"move|explore|search)\b[^.!?;]{{0,64}}\b"
            rf"{_MANIPULATION_PATTERN}\b")
        explicit_you = re.compile(
            rf"\b(?:you|you'll|you\s+will|you\s+should|you\s+can|"
            rf"you\s+need\s+to)\b[^.!?;]{{0,48}}\b{_MANIPULATION_PATTERN}\b")
        named_scout = re.compile(
            rf"\b(?:bob|scout|box\s+scout|dog)\b\s*[:,]?\s*"
            rf"(?:please\s+)?{_MANIPULATION_PATTERN}\b")
        if (any(imperative.search(clause) or compound_imperative.search(clause)
                for clause in clauses) or
                explicit_you.search(text.lower()) or
                named_scout.search(text.lower())):
            decision.update(
                accepted=False,
                reason="receiver_scout_manipulation_directive",
            )
            return None, decision

    if sender_role == "scout":
        first_person_claim = re.compile(
            rf"\b(?:i(?:'ll|\s+will|\s+can|\s+should|\s+need\s+to|"
            rf"\s+am\s+going\s+to|'m\s+going\s+to)|let\s+me|"
            rf"we(?:'ll|\s+will|\s+can|\s+should|\s+need\s+to|"
            rf"\s+are\s+going\s+to|'re\s+going\s+to))\s+"
            rf"(?:\w+\s+){{0,3}}?{_MANIPULATION_PATTERN}\b",
            re.IGNORECASE,
        )
        completed_or_current_claim = re.compile(
            rf"\b(?:i|we)\s+(?:(?:already|successfully)\s+)?"
            rf"{_MANIPULATION_PATTERN}\b|"
            rf"\b(?:i|we)(?:'ve|\s+have|\s+had)\s+"
            rf"(?:(?:already|successfully)\s+)?{_MANIPULATION_PATTERN}\b|"
            rf"\b(?:i(?:'m|\s+am)|we(?:'re|\s+are))\s+"
            rf"(?:(?:currently|already|successfully)\s+)?"
            rf"{_MANIPULATION_PATTERN}\b|"
            rf"\b(?:i|we)\s+(?:finished|completed)\s+"
            rf"{_MANIPULATION_PATTERN}\b",
            re.IGNORECASE,
        )
        compound_past_claim = re.compile(
            rf"\b(?:i|we)\s+(?:went|moved|walked|navigated|traveled|"
            rf"travelled)\b[^.!?;]{{0,80}}\b(?:and|then)\s+"
            rf"{_MANIPULATION_PATTERN}\b",
            re.IGNORECASE,
        )
        recommendation = re.compile(
            rf"^\s*(?:i\s+)?(?:recommend|advise|suggest)\b"
            rf"[^.!?;]{{0,80}}\b(?:you|human|alice)\b"
            rf"[^.!?;]{{0,48}}\b{_MANIPULATION_PATTERN}\b",
            re.IGNORECASE,
        )
        for clause in re.split(r"[\n.!?;]+", text):
            if recommendation.search(clause):
                continue
            if (first_person_claim.search(clause) or
                    completed_or_current_claim.search(clause) or
                    compound_past_claim.search(clause)):
                decision.update(
                    accepted=False,
                    reason="sender_scout_self_manipulation_claim",
                )
                return None, decision

    return message, decision


def normalize_agent_role(role):
    """Return the canonical high-level capability role.

    The published TDW-MAT agents are manipulators.  The Human+Box evaluation
    opts agent 1 into the ``scout`` role without changing legacy callers.
    """
    if role is None:
        return "human"
    normalized = str(role).strip().lower()
    return "scout" if normalized in _SCOUT_ROLE_ALIASES else "human"


def plan_allowed_for_role(role, plan):
    """Enforce the role capability set independently of the prompt."""
    if normalize_agent_role(role) != "scout":
        return True
    if not isinstance(plan, str):
        return False
    normalized = plan.strip().lower()
    return normalized.startswith((
        "go to ",
        "explore ",
        "send a message:",
        "wait",
        "[wait]",
    ))


def action_allowed_for_role(role, action):
    """Reject manipulation primitives for a non-manipulating scout.

    Type 8 is the Box adapter's explicit wait/no-op primitive.  It is opt-in:
    the original Replicant environment never receives it unless Scout mode is
    enabled.
    """
    if normalize_agent_role(role) != "scout":
        return True
    if not isinstance(action, dict) or "type" not in action:
        return False
    return action["type"] in {0, 1, 2, 6, 8, "ongoing"}


def _env_flag_enabled(name):
    """Return True only when an environment flag is explicitly enabled."""
    return os.getenv(name, "").strip().lower() in _TRUE_ENV_VALUES


def _normalize_communication_message(message, allow_unquoted=False):
    """Convert a generator response into an optional message candidate.

    With ``allow_unquoted=False`` this intentionally reproduces upstream
    CoELA: an unquoted response without an embedded quoted span becomes None.
    The opt-in path accepts a non-empty stripped response, which is required
    for OpenAI-compatible models such as Qwen that follow the content request
    but don't surround the message with ASCII double quotes.
    """
    if not allow_unquoted:
        if len(message) > 0 and message[0] != '"':
            message = re.search(r'"([^"]+)"', message)
            if message:
                message = '"' + message.group(1) + '"'
        return message

    if message is None:
        return None
    message = message.strip()
    if not message:
        return None
    if message[0] == '"':
        return message
    quoted_message = re.search(r'"([^"]+)"', message)
    if quoted_message:
        return '"' + quoted_message.group(1) + '"'
    return message


def _v4_coordination_event_reference(message, valid_event_ids):
    """Return one exact V4 event reference, never model-authored prose.

    The model still decides whether to communicate and, after making that
    decision, which public coordination event to send.  The environment
    adapter later renders the referenced structured event.  Rejecting every
    other output keeps raw observations and hidden reasoning out of the
    communication channel without trying to recognize benchmark-specific
    geometry in free text.
    """
    if message is None:
        return None
    text = str(message).strip().strip('"\'` ')
    if ":" not in text:
        return None
    prefix, value = text.split(":", 1)
    if prefix.strip().lower() != "coordination_event":
        return None
    try:
        event_id = int(value.strip())
    except (TypeError, ValueError):
        return None
    valid_ids = {int(item) for item in valid_event_ids}
    if event_id not in valid_ids:
        return None
    return f"coordination_event:{event_id}"


class LLM:
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
                 lm_id,
                 prompt_template_path,
                 communication,
                 cot,
                 sampling_parameters,
                 agent_id,
                 agent_role="human",
                 opponent_role="human",
                 ):
        self.rooms_explored = None
        self.goal_desc = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 0 else "Bob"
        self.oppo_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 1 else "he"
        self.agent_role = normalize_agent_role(agent_role)
        self.opponent_role = normalize_agent_role(opponent_role)
        self.debug = sampling_parameters.debug
        self.rooms = []
        self.prompt_template_path = prompt_template_path
        self.single = 'single' in self.prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        role_replacements = {
            "$AGENT_NAME$": self.agent_name,
            "$OPPO_NAME$": self.oppo_name,
            "$AGENT_ROLE$": ("Box Scout (Dog)" if self.agent_role == "scout"
                              else "Human manipulator"),
            "$OPPO_ROLE$": ("Box Scout (Dog)" if self.opponent_role == "scout"
                             else "Human manipulator"),
        }
        self.prompt_template = df['prompt'][0]
        for placeholder, value in role_replacements.items():
            self.prompt_template = self.prompt_template.replace(placeholder, value)
        if communication:
            self.generator_prompt_template = df['prompt'][1]
            for placeholder, value in role_replacements.items():
                self.generator_prompt_template = self.generator_prompt_template.replace(placeholder, value)
        else:
            self.generator_prompt_template = None

        self.communication = communication
        # PeerConsult V3.1 can close only the outbound message slot while
        # retaining the communication prompt and received dialogue context.
        self.allow_message_this_turn = True
        # Disabled by default for exact upstream-baseline behavior.
        self.fix_lm_communication = _env_flag_enabled(
            "TDW_MAT_FIX_LM_COMMUNICATION")
        # Qwen3.5 enables its native hidden-thinking mode by default. CoELA
        # already performs an explicit two-stage chain-of-thought call with a
        # tight output budget, so native thinking can consume the entire
        # response before the parseable action is emitted. Keep this opt-in so
        # upstream and earlier-model behavior remains unchanged.
        self.disable_model_thinking = _env_flag_enabled(
            "TDW_MAT_DISABLE_MODEL_THINKING")
        self.cot = cot
        self.source = source
        self.model = None
        self.tokenizer = None
        self.lm_id = lm_id
        # OpenAI-compatible servers such as vLLM expose chat completions even
        # when the served model name doesn't contain "chat".
        self.chat = (self.source == "openai" or 'gpt-3.5-turbo' in lm_id or
                     'gpt-4' in lm_id or 'chat' in lm_id.lower())
        self.OPENAI_KEY = None
        self.total_cost = 0

        if self.source == "openai":
            openai_base_url = os.getenv("OPENAI_BASE_URL")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            request_timeout = float(os.getenv("TDW_MAT_OPENAI_TIMEOUT", "90"))
            if openai_base_url:
                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                    base_url=openai_base_url,
                    timeout=request_timeout,
                    max_retries=0,
                )
            elif azure_endpoint:
                client = AzureOpenAI(
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
                    azure_endpoint=azure_endpoint,
                    timeout=request_timeout,
                    max_retries=0,
                )
            else:
                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                    timeout=request_timeout,
                    max_retries=0,
                )
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
                if self.disable_model_thinking:
                    self.sampling_params["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif self.source == 'hf':
            # Keep the heavy local-HF stack optional. The recommended setup in
            # this workspace uses a remote vLLM OpenAI-compatible endpoint.
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.lm_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.lm_id, device_map='auto', load_in_4bit=True)
            self.sampling_params = {
                "max_new_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "num_return_sequences": sampling_parameters.n,
                'use_cache': True,
                # 'output_scores': True,
                'return_dict_in_generate': True,
                'do_sample': True,
                # 'early_stopping': True,
            }
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id):

            # A broken SSH tunnel must fail the current episode instead of
            # retrying forever and leaving a multi-day evaluation hung.
            # Long LM Studio/Funnel runs can also see brief TLS EOF windows;
            # allow tagged recovery runs to opt into a longer bounded retry
            # window without changing the default evaluation behavior.
            retry_max_tries = int(os.getenv("TDW_MAT_OPENAI_RETRY_MAX_TRIES", "4"))
            retry_max_time = int(os.getenv("TDW_MAT_OPENAI_RETRY_MAX_TIME", "300"))
            @backoff.on_exception(backoff.expo, OpenAIError,
                                  max_tries=retry_max_tries,
                                  max_time=retry_max_time)
            def openai_generate(prompt, sampling_params):
                usage = 0
                try:
                    if self.chat:
                        response = client.chat.completions.create(model=self.lm_id, messages=prompt, **sampling_params)
                        if self.debug:
                            debug_dir = os.getenv("TDW_MAT_LOG_DIR", "LLM")
                            os.makedirs(debug_dir, exist_ok=True)
                            with open(os.path.join(debug_dir, "chat_raw.json"), 'a') as f:
                                f.write(json.dumps(response.choices[0].message.content, indent=4))
                                f.write('\n')
                        generated_samples = [response.choices[i].message.content for i in
                                             range(sampling_params['n'])]
                        if 'gpt-4' in self.lm_id or 'gpt4' in self.lm_id:
                            usage = response.usage.prompt_tokens * 0.03 / 1000 + response.usage.completion_tokens * 0.06 / 1000
                        elif 'gpt-3.5' in self.lm_id:
                            usage = response.usage.total_tokens * 0.002 / 1000
                    # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                    #                   range(sampling_params['n'])]
                    elif "text-" in lm_id:
                        response = client.completions.create(model=lm_id, prompt=prompt, **sampling_params)
                        # print(json.dumps(response, indent=4))
                        if self.debug:
                            debug_dir = os.getenv("TDW_MAT_LOG_DIR", "LLM")
                            os.makedirs(debug_dir, exist_ok=True)
                            with open(os.path.join(debug_dir, "raw.json"), 'a') as f:
                                f.write(json.dumps(response, indent=4))
                                f.write('\n')
                        generated_samples = [response.choices[i].text for i in range(sampling_params['n'])]
                    # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                    #               range(sampling_params['n'])]
                    else:
                        raise ValueError(f"{lm_id} not available!")
                except OpenAIError as e:
                    print(e)
                    raise e
                return generated_samples, usage

            def tokenize_dialog(dialog):
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                prompt_tokens = []
                # print(dialog)
                if dialog[0]["role"] == "system":
                    dialog = [
                                 {
                                     "role": dialog[1]["role"],
                                     "content": B_SYS
                                                + dialog[0]["content"]
                                                + E_SYS
                                                + dialog[1]["content"],
                                 }
                             ] + dialog[2:]
                assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                    [msg["role"] == "assistant" for msg in dialog[1::2]]
                ), (
                    "model only supports 'system', 'user' and 'assistant' roles, "
                    "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
                )
                dialog_tokens: List[int] = sum(
                    [
                        [self.tokenizer.bos_token_id] +
                        self.tokenizer.encode(
                            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                            add_special_tokens=False
                        )
                        + [self.tokenizer.eos_token_id]
                        for prompt, answer in zip(dialog[::2], dialog[1::2], )
                    ],
                    [],
                )
                assert (
                        dialog[-1]["role"] == "user"
                ), f"Last message must be from user, got {dialog[-1]['role']}"
                dialog_tokens += [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}", add_special_tokens=False
                )
                prompt_tokens.append(dialog_tokens)
                return self.torch.tensor(prompt_tokens).to('cuda')

            def hf_generate(prompt, sampling_params):
                with self.torch.inference_mode():
                    if self.chat:
                        input_ids = tokenize_dialog(prompt)
                    else:
                        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
                    prompt_len = input_ids.shape[-1]
                    output_dict = self.model.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id, # max_length=prompt_len + sampling_params['max_new_tokens'],
                                                 **sampling_params)
                generated_samples = self.tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
                generated_samples = [s.strip() for s in generated_samples]
                generated_samples = [s[:-4] if '</s>' in s[-4:] else s for s in generated_samples]
                if self.debug:
                    print(generated_samples)
                return generated_samples, 0

            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    return openai_generate(prompt, sampling_params)
                elif self.source == 'hf':
                    return hf_generate(prompt, sampling_params)
                else:
                    raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.source, self.lm_id)

        self.current_room = None
        self.object_list = None
        self.holding_objects = None
        self.obj_per_room = None


    def reset(self, rooms_name, goal_objects):
        self.rooms = rooms_name
        self.goal_desc = self.goal2description(goal_objects)


    def goal2description(self, goals):  # {predicate: count}
        s = "Transport "
        r = None
        for object_name, count in goals.items():
            s += f"{count} {object_name}{'s' if count > 1 else ''}, "

        s = s[:-2] + f" to the bed."
        return s


    def parse_answer(self, available_actions, text):
        if (getattr(self, "peer_consult_protocol", None) == "PeerConsultV4" and
                getattr(self, "peer_consult_policy", "legacy") == "llm"):
            matches = [action for action in available_actions
                       if action.lower() in str(text).lower()]
            if len(matches) == 1:
                return matches[0], "AC"
            labels = re.findall(r"\b(?:option|action)\s+([A-Z])\b|^\s*([A-Z])[.)]?\s*$",
                                str(text), flags=re.IGNORECASE)
            indices = {ord((left or right).upper()) - ord("A")
                       for left, right in labels}
            if len(indices) == 1:
                index = indices.pop()
                if index in range(len(available_actions)):
                    return available_actions[index], "AC"
            return None, "invalid_or_ambiguous_model_selection"
        flags = 'AC'
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action.startswith("send a message:"):
                action = "send a message"
            if action.lower() in text.lower():
                return available_actions[i], flags
        sents = text.split('\n')  # Split by space
        words = []
        for sent in sents:
            words.extend(sent.split(' '))
        words = list(filter(None, words))  # Remove empty strings from the result

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in words or f"{option}," in words or f"{option}\n" in text.split(" ") or f"Option {option}" in text or f"({option})" in words or f"action {option}" in text or (len(text) <= 2 and option in text):
                return action, flags
        print("WARNING! Fuzzy match!")
        flags = "Fuzzy match"
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action.startswith("send a message"):
                continue
            act = "None"
            name = "None"
            id = "None"
            if action.startswith('go to'):
                # act = 'go to'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('explore'):
                act = 'explore'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('go grasp'):
                act = 'grasp'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('put'):
                act = 'put'
            elif action.startswith('transport'):
                act = 'transport'
            option = chr(ord('A') + i)
            if name in text and id in text:
                return action, flags

        for i in range(len(available_actions)):
            action = available_actions[i]
            if action.startswith("send a message"):
                continue
            act = "None"
            name = "None"
            id = "None"
            if action.startswith('go to'):
                # act = 'go to'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('explore'):
                act = 'explore'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('go grasp'):
                act = 'grasp'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('put'):
                act = 'put'
            elif action.startswith('transport'):
                act = 'transport'
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action, flags
        if len(text) == 1:
            i = ord(text) - ord('A')
            if i in range(len(available_actions)):
                return available_actions[i]
        print("WARNING! No available action parsed!!! Random choose one")
        flags = "failed to parse"
        return random.choice(available_actions), flags


    def progress2text(self, current_step, satisfied, opponent_grabbed_objects, opponent_last_room,):
        s = f"I've taken {current_step}/3000 steps. "

        sss = {}
        for room, obj_list in self.obj_per_room.items():
            sr = ""
            s_obj = ""
            s_con = ""
            s_bed = ""
            objs = obj_list[0]
            cons = obj_list[1]
            if len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"a target object <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in objs])
                    s_obj += f"target objects " + ss

            if len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"a container <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in cons])
                    s_con = f"containers " + ss
            if len(obj_list[2]) > 0:
                s_bed = 'the goal position bed'
            if s_obj == "" and s_con == "" and s_bed == "":
                sr += 'nothing'
            elif s_obj != "" and s_con != "" and s_bed == "":
                sr += s_obj + ', and ' + s_con
            elif s_obj != "" and s_con == "" and s_bed != "":
                sr += s_obj + ', and ' + s_bed
            elif s_obj == "" and s_con != "" and s_bed != "":
                sr += s_con + ', and ' + s_bed
            elif s_obj != "" and s_con != "" and s_bed != "":
                sr += s_obj + ', ' + s_con + ', and ' + s_bed
            else:
                sr += s_obj + s_con + s_bed
            sss[room] = sr

        if len(satisfied) == 0:
            if len(self.object_list[2]) == 0:
                s += "I haven't found the goal position bed. "
            else:
                s += ""
        else:
            s += f"{'I' if self.single else 'We'}'ve already transported "
            unique_satisfied = []
            for x in satisfied:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            if len([x for x in unique_satisfied if x['type'] == 0]) == 0:
                s += 'nothing'
            s += ', '.join([f"<{x['name']}> ({x['id']})" for x in unique_satisfied if x['type'] == 0])
            s += ' to the bed. '

        s_hold = ["", ""]
        for i, obj in enumerate(self.holding_objects):
            if obj['type'] == 0:
                s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
            elif obj['type'] == 1:
                ss = ""
                cnt = 0
                for j, o in enumerate(obj['contained']):
                    if o is None:
                        break
                    cnt += 1
                    ss += f"<{obj['contained_name'][j]}> ({o}), "
                if cnt == 0:
                    ss = 'nothing'
                else:
                    ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "

        if self.holding_objects[0]["type"] == 0 and self.holding_objects[1]['type'] == 0:
            s += f"I'm holding two target objects <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) and <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}). "
        elif s_hold[0] == "" and s_hold[1] == "":
            s += "I'm holding nothing. "
        elif s_hold[0] != "" and s_hold[1] != "":
            s += f"I'm holding {s_hold[0][:-2]}, and {s_hold[1]}"
        else:
            s += f"I'm holding {s_hold[0]}{s_hold[1]}"

        # print(self.current_room, self.obj_per_room)
        if self.current_room not in self.rooms_explored: pred_room = 'none'
        else: pred_room = self.rooms_explored[self.current_room]
        if pred_room != 'all' and sss[self.current_room] == 'nothing':
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it. "
        else:
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it and found {sss[self.current_room]}. "
        ### opponent modeling
        if not self.single:
            s_hold = ["", ""]
            for i, obj in enumerate(opponent_grabbed_objects):
                if obj['type'] == 0:
                    s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
                elif obj['type'] == 1:
                    ss = ""
                    cnt = 0
                    for j, o in enumerate(obj['contained']):
                        if o is None:
                            break
                        cnt += 1
                        ss += f"<{obj['contained_name'][j]}> ({o}), "
                    if cnt == 0:
                        ss = 'nothing'
                    else:
                        ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                    s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "
            if opponent_grabbed_objects[0]["type"] == 0 and opponent_grabbed_objects[1]['type'] == 0:
                ss = f"two target objects <{opponent_grabbed_objects[0]['name']}> ({opponent_grabbed_objects[0]['id']}) and <{opponent_grabbed_objects[1]['name']}> ({opponent_grabbed_objects[1]['id']}). "
            if s_hold[0] == "" and s_hold[1] == "":
                ss = "nothing. "
            elif s_hold[0] != "" and s_hold[1] != "":
                ss = f"{s_hold[0][:-2]}, and {s_hold[1]}"
            else:
                ss = f"{s_hold[0]}{s_hold[1]}"

            if opponent_last_room is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room == self.current_room:
                s += f"I also see {self.oppo_name} here in the {self.current_room}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == self.current_room:
                continue
            #s += f"I've explored {self.rooms_explored[room] if room in self.rooms_explored else 'None'} of the {room}, and I found {sss[room]} there. "
            if room not in self.rooms_explored: pred_room = 'none'
            else: pred_room = self.rooms_explored[room]
            if pred_room != 'all' and sss[room] == 'nothing':
                s += f"I've explored {pred_room} of the {room}. "
            else:
                s += f"I've explored {pred_room} of the {room}, and I found {sss[room]} there. "

        return s


    def get_available_plans(self, message=None, include_message=False):
        """
        go to room {}
        explore current room {}
        go grasp target object / container {}
        holding both container and object: put obj into the container
        holding any goal objects: transport holding objects to the bed
        send a message: ""
        """
        # Some legacy utilities/tests construct this class with ``__new__``
        # and populate only the original fields.  Treat those objects as the
        # historical all-capable human role.
        agent_role = getattr(self, "agent_role", "human")
        available_plans = []
        if self.communication and include_message:
            # V3.3 asks the planner whether communication is worth an action
            # before spending a separate model call on wording it.
            available_plans.append("send a message")
        elif self.communication and message is not None:
            # Preserve the legacy API used by older protocols and tests.
            available_plans.append(f"send a message: {message}")
        task_card = getattr(self, "peer_decision_card", None)
        task_rows = ((task_card or {}).get("task_queue") or [])
        is_v4 = (getattr(self, "peer_consult_protocol", None) ==
                 "PeerConsultV4")
        common_v4 = is_v4 and getattr(self, "peer_consult_policy", "legacy") == "llm"
        # A target whose lifecycle has already advanced beyond acquisition
        # must never re-enter the grasp menu.  In particular, ``carried``
        # objects are still present in the V3.3 task queue and can remain in a
        # stale private object map; admitting them here caused the repeated
        # cannot_grasp loops seen in E14.
        non_acquirable_statuses = {
            "blocked", "carried", "completed", "surplus", "delivered",
            "in_use",
        }
        payload_ids = set()
        for owner_key in ("self", "peer"):
            for item in (((task_card or {}).get(owner_key) or {})
                         .get("payload") or []):
                if item.get("id") is not None:
                    payload_ids.add(int(item["id"]))
        allowed_target_ids = {
            int(row["object_id"]) for row in task_rows
            if (row.get("kind") == "deliver_goal_object" and
                str(row.get("status") or "").lower() not in
                non_acquirable_statuses and
                row.get("object_id") is not None and
                int(row["object_id"]) not in payload_ids)
        }
        allowed_container_ids = {
            int(item["id"]) for item in
            ((task_card or {}).get("actionable_containers") or [])
            if item.get("id") is not None
        }
        # V3.x's card is a shared-board whitelist.  V4 deliberately keeps
        # unpublished local perception private, so absence from the shared
        # task queue cannot make a locally observed, physically available
        # object disappear from this agent's candidate set.  Atomic claim and
        # physical-truth validation happen after proposal collection.
        local_goal_work = bool(task_card and any(
            int(obj["id"]) in allowed_target_ids
            for obj in self.object_list[0]
            if obj.get("id") is not None))
        holding_payload = any(
            obj.get('type') is not None for obj in self.holding_objects)
        local_bed_known = bool(self.object_list[2])
        shared_bed = self.shared_delivery_target()
        shared_bed_position = ((shared_bed or {}).get("position")
                               if shared_bed else None)
        shared_bed_room = ((shared_bed or {}).get("room")
                           if shared_bed else None)
        try:
            shared_bed_navigable = (
                shared_bed_position is not None and
                len(shared_bed_position) >= 3)
        except TypeError:
            shared_bed_navigable = False
        if agent_role != "scout":
            if self.holding_objects[0]['type'] is None or self.holding_objects[1]['type'] is None:
                for obj in self.object_list[0]:
                    if (task_card and not is_v4 and
                            int(obj['id']) not in allowed_target_ids):
                        continue
                    available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']})")
                if not (self.holding_objects[0]['type'] == 1 or self.holding_objects[1]['type'] == 1):
                    for obj in self.object_list[1]:
                        if (task_card and not is_v4 and
                                int(obj['id']) not in allowed_container_ids):
                            continue
                        available_plans.append(f"go grasp container <{obj['name']}> ({obj['id']})")
            else:
                if self.holding_objects[0]['type'] == 1 and self.holding_objects[0]['contained'][-1] is None and self.holding_objects[1]['type'] == 0:
                    available_plans.append(f"put <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}) into the container <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']})")
                elif self.holding_objects[1]['type'] == 1 and self.holding_objects[1]['contained'][-1] is None and self.holding_objects[0]['type'] == 0:
                    available_plans.append(f"put <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) into the container <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']})")
            if (holding_payload and
                    (local_bed_known or shared_bed_navigable)):
                available_plans.append(f"transport objects I'm holding to the bed")
        # V3.3-V3.5 intentionally suppress blind room search once useful local
        # goal work is known.  PeerConsult V4 restores upstream CoELA's
        # candidate completeness: a legal exploration action remains visible
        # even when a local target is also available.  This is protocol-gated
        # so archived V3.x runs and ablations retain their exact action space.
        #
        # The LLM, rather than this compiler, decides whether the information
        # value of exploration is worth giving up target-task continuity.
        # No distance, confidence, container-utility, or waypoint policy is
        # introduced here.
        # Liveness invariant: a full carrier that hasn't personally observed
        # the bed must still be able to act.  A team-shared bed position enables
        # transport directly; otherwise room navigation/exploration remains
        # available even while a known goal task exists.  This avoids the
        # ``known goal -> suppress search`` + ``unknown local bed -> suppress
        # transport`` empty-action intersection.
        locating_bed = bool(
            agent_role != "scout" and holding_payload and
            not local_bed_known and not shared_bed_navigable)
        if is_v4 or not local_goal_work or locating_bed:
            rooms = list(self.rooms)
            if shared_bed_room in rooms:
                rooms.remove(shared_bed_room)
                rooms.insert(0, shared_bed_room)
            for room in rooms:
                if room == self.current_room or room is None or room == 'None':
                    continue
                available_plans.append(f"go to {room}")
            if (common_v4 or self.current_room not in self.rooms_explored or
                    self.rooms_explored[self.current_room] != 'all' or
                    (locating_bed and shared_bed_room == self.current_room)):
                available_plans.append(f"explore current room {self.current_room}")
            # Even a one-room scene marked fully explored must not collapse to
            # an empty plan while a payload is stranded without a delivery
            # position. Re-scan is preferable to thousands of forced waits.
            if (locating_bed and not any(
                    not plan.startswith("send a message")
                    for plan in available_plans)):
                available_plans.append(
                    f"explore current room {self.current_room}")
        if agent_role == "scout":
            available_plans.append("wait")
        elif common_v4:
            available_plans.append("wait")
        if common_v4:
            available_plans.append("release current task")

        # Defense in depth: even future plan generators cannot accidentally
        # expose a manipulation plan to the Box Scout.
        available_plans = [plan for plan in available_plans
                           if plan_allowed_for_role(agent_role, plan)]

        if is_v4 and not common_v4:
            # Persistence is not insistence.  Only a genuinely active task
            # receives the stable continuity position.  Suspended/blocked
            # tasks and a task referenced by the one-boundary planning-loop
            # guard remain legal candidates, but are not forced to the front.
            # The post-proposal V4 validator, not this candidate compiler,
            # consumes and enforces the guard.
            continuity_task = self._v4_continuity_task(task_card)
            available_plans = self._prefer_task_plan(
                available_plans, continuity_task)

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

    @staticmethod
    def _v4_continuity_task(task_card):
        """Return the V4 task eligible for non-binding plan continuity.

        The task is deliberately *not* removed when it is suspended or when a
        planning-loop guard is pending.  Returning ``None`` only removes its
        ordering bonus, keeping the complete legal action set available to the
        local planner.
        """
        card = task_card or {}
        active_task = card.get("active_task")
        if not isinstance(active_task, dict) or not active_task:
            return None
        status = str(active_task.get("status") or "").lower()
        if status in {
                "suspended", "blocked", "completed", "surplus", "carried",
                "delivery_pending", "parked_at_bed", "in_use"}:
            return None

        guard = card.get("planning_loop_guard") or {}
        if (isinstance(guard, dict) and
                (guard.get("applies_to_next_boundary") or
                 guard.get("applies_to_next_planning_boundary")) and
                guard.get("task_id") == active_task.get("task_id")):
            return None
        return active_task

    @staticmethod
    def _prefer_task_plan(available_plans, task):
        """Stable-partition plans so an active V4 task keeps continuity.

        Communication candidates retain their existing leading slot.  This is
        an ordering hint only: no candidate is added, removed, or assigned a
        benchmark-specific utility score.
        """
        if not isinstance(task, dict):
            return available_plans

        task_kind = str(task.get("kind") or "")
        task_id = task.get("task_id")
        object_id = task.get("object_id")
        room = task.get("room")

        def matches(plan):
            if task_kind == "deliver_payload":
                return plan.startswith("transport objects I'm holding")
            if task_kind == "deliver_goal_object" and object_id is not None:
                return (plan.startswith("go grasp target object ") and
                        plan.endswith(f"({int(object_id)})"))
            if task_kind == "container_resource" and object_id is not None:
                return (plan.startswith("go grasp container ") and
                        plan.endswith(f"({int(object_id)})"))
            if task_kind == "explore_room" and room:
                return plan in {
                    f"go to {room}",
                    f"explore current room {room}",
                }
            # Stable task IDs keep the helper tolerant of a compact V4 card
            # that omits the redundant ``kind`` field.
            if (isinstance(task_id, str) and
                    task_id.startswith("entity:") and
                    object_id is not None):
                return plan.endswith(f"({int(object_id)})")
            return False

        communication = [
            plan for plan in available_plans
            if plan == "send a message" or
            plan.startswith("send a message:")]
        physical = [plan for plan in available_plans
                    if plan not in communication]
        preferred = [plan for plan in physical if matches(plan)]
        if not preferred:
            return available_plans
        remainder = [plan for plan in physical if not matches(plan)]
        return communication + preferred + remainder

    def shared_delivery_target(self):
        """Return a compact team-shared bed record when the card provides it.

        V3.4 publishes ``shared_bed``.  The aliases keep this planner tolerant
        of diagnostic cards and make no change to V3.3/legacy runs, whose
        cards don't contain any of these keys.
        """
        card = getattr(self, "peer_decision_card", None) or {}
        advisory = card.get("delivery_advisory") or {}
        for source in (card, advisory):
            for key in ("shared_bed", "delivery_target", "known_bed",
                        "bed"):
                value = source.get(key)
                if isinstance(value, list):
                    value = value[0] if value else None
                if isinstance(value, dict):
                    return value
        # V3.4's delivery advisory also exposes compact flat fields so the
        # coordinator needn't duplicate the bed entity in the card.
        if advisory.get("bed_position") is not None:
            return {
                "id": advisory.get("bed_object_id"),
                "name": "bed",
                "type": 2,
                "position": advisory.get("bed_position"),
                "room": advisory.get("bed_room"),
                "knowledge_source": advisory.get(
                    "bed_knowledge_source"),
            }
        return None


    def run(self, current_step, current_room, rooms_explored, holding_objects, satisfied, object_list, obj_per_room, action_history, dialogue_history, opponent_grabbed_objects = None, opponent_last_room = None):
        info = {}
        print("current_step", current_step)
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        progress_desc = self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        # PeerConsult injects one compact, current decision card. Preserve a
        # wider tail of agent-authored dialogue around that card so requests
        # aren't discarded after only two later messages. Legacy CoELA keeps
        # its exact three-entry behavior when no Memory Board is present.
        board_items = [item for item in dialogue_history
                       if item.startswith("[Memory Board] ")]
        if board_items:
            natural_dialogue = [item for item in dialogue_history
                                if not item.startswith("[Memory Board] ")]
            selected_dialogue = natural_dialogue[-6:] + [board_items[-1]]
        else:
            selected_dialogue = (dialogue_history[-3:]
                                 if len(dialogue_history) > 3
                                 else dialogue_history)
        dialogue_history_desc = '\n'.join(selected_dialogue)
        prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
        prompt = prompt.replace('$PROGRESS$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)
        if self.communication:
            prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
        elif '$DIALOGUE_HISTORY$' in prompt:
            prompt = prompt.replace('$DIALOGUE_HISTORY$', '')

        protocol = getattr(self, "peer_consult_protocol", None)
        is_v33 = protocol in (
            "PeerConsultV3.3", "PeerConsultV3.4", "PeerConsultV3.5")
        is_v34 = protocol in ("PeerConsultV3.4", "PeerConsultV3.5")
        is_v4 = protocol == "PeerConsultV4"
        common_v4 = is_v4 and getattr(self, "peer_consult_policy", "legacy") == "llm"
        decision_first_communication = is_v33 or is_v4
        if is_v33:
            prompt += (
                f"\n{protocol} decision policy: continue or resume an "
                "unfinished goal-object task before blind exploration. "
                "Do not switch merely because new evidence appeared; switch "
                "only after completion, a physical impossibility, a conflict, "
                "or a blocking review. A communication slot is optional: "
                "choose 'send a message' only when the listed new event would "
                "change the peer's next action, and never repeat a fact."
            )
        elif common_v4:
            prompt += (
                "\nPeerConsultV4 common policy: choose your own task, recovery, retry, "
                "exploration and waiting strategy. Previous failures and coverage are "
                "evidence, never a ban on another attempt. Wait preserves your task; "
                "release current task explicitly withdraws your commitment. A scored "
                "object stays scored if moved again. Consult common_core for execution "
                "results, peer commitments, and cooperation proposals. A proposal or "
                "ready statement is not an observed world fact."
            )
        elif is_v4:
            prompt += (
                "\nPeerConsultV4 decision policy: every listed action is a "
                "legal option. The active task is a continuity hint, not a "
                "mandate; persistence must not become repeated insistence. "
                "If the Memory Board contains a planning_loop_guard for this "
                "boundary, choose a different listed action from the guarded "
                "one. Decide the alternative locally; the coordination layer "
                "does not prescribe a room, target, route, or container. A "
                "communication slot is optional and should be used only for "
                "a concise task, claim, failure, or release update that can "
                "change the peer's next action."
            )
        pending_events = list(
            getattr(self, "pending_perception_events", None) or [])
        if is_v34 and pending_events:
            prompt += (
                "\nNew perception events retained until this safe planning "
                "boundary: " + json.dumps(
                    pending_events, ensure_ascii=False,
                    separators=(",", ":")) +
                ". These are evidence, not mandatory task switches. Prefer "
                "a newly visible undelivered goal target over blind search; "
                "if a bed became known while carrying payload, consider "
                "delivery. Otherwise resume the persistent active task."
            )
        legacy_message = None
        if (self.communication and not decision_first_communication and
                self.allow_message_this_turn and
                not action_history[-1].startswith('send a message')):
            # Preserve the published/V3.x ablation behavior. Only V3.3 moves
            # content generation behind the planner's send decision.
            gen_prompt = self.generator_prompt_template.replace(
                '$GOAL$', self.goal_desc)
            gen_prompt = gen_prompt.replace('$PROGRESS$', progress_desc)
            gen_prompt = gen_prompt.replace(
                '$ACTION_HISTORY$', action_history_desc)
            gen_prompt = gen_prompt.replace(
                '$DIALOGUE_HISTORY$', dialogue_history_desc)
            gen_prompt += f"\n{self.agent_name}:"
            chat_prompt = [{"role": "user", "content": gen_prompt}]
            legacy_outputs, legacy_usage = self.generator(
                chat_prompt if self.chat else gen_prompt,
                self.sampling_params)
            self.total_cost += legacy_usage
            legacy_message = _normalize_communication_message(
                legacy_outputs[0],
                allow_unquoted=self.fix_lm_communication)
            legacy_message, communication_guard = (
                guard_communication_message(
                    legacy_message, self.agent_role, self.opponent_role))
            info['prompt_comm'] = gen_prompt
            info['output_comm'] = legacy_outputs
            info['usage_comm'] = legacy_usage
            info['communication_guard'] = communication_guard
        message_slot = bool(
            decision_first_communication and self.communication and
            self.allow_message_this_turn and
            not action_history[-1].startswith('send a message'))
        available_plans, num, available_plans_list = self.get_available_plans(
            message=legacy_message, include_message=message_slot)
        if num == 0 or (legacy_message is not None and num == 1):
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                     "plan": None})
            return plan, info

        decision_prompt_template = prompt
        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

        if self.cot:
            prompt = prompt + " Let's think step by step."
            if self.debug:
                print(f"cot_prompt:\n{prompt}")
            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
            output = outputs[0]
            ## truncate the unfinished cot
            last_index = output.rfind('.')
            if last_index != -1:
                output = output[:last_index + 1]
            else:
                output += '.'
            self.total_cost += usage
            # info['outputs_cot'] = outputs
            # info['usage_plan_stage_1'] = usage
            if self.debug:
                print(f"output_plan_stage_1:\n{output}")
            chat_prompt = [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": output},
                           {"role": "user", "content": "Answer with only one best next action. So the answer is option"}]
            normal_prompt = prompt + ' ' + output + ' Answer with only one best next action. So the answer is option'
            outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
            output = outputs[0]
            self.total_cost += usage
            # info['usage_plan_stage_2'] = usage
            if self.debug:
                print(f"output_plan_stage_1:\n{output}")
                print(f"total cost: {self.total_cost}")
        else:
            normal_prompt = prompt
            chat_prompt = [{"role": "user", "content": prompt}]
            if self.debug:
                print(f"base_prompt:\n{prompt}")
            outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
            output = outputs[0]
            # info['usage_step_1'] = usage
            if self.debug:
                print(f"output_plan_stage_1:\n{output}")
        plan, flags = self.parse_answer(available_plans_list, output)
        if plan == "send a message":
            # Communication content is generated only after the planner has
            # decided that a message is worth consuming an environment turn.
            gen_prompt = self.generator_prompt_template.replace(
                '$GOAL$', self.goal_desc)
            gen_prompt = gen_prompt.replace('$PROGRESS$', progress_desc)
            gen_prompt = gen_prompt.replace(
                '$ACTION_HISTORY$', action_history_desc)
            gen_prompt = gen_prompt.replace(
                '$DIALOGUE_HISTORY$', dialogue_history_desc)
            v4_event_ids = []
            if is_v33:
                events = ((getattr(self, "peer_decision_card", None) or {})
                          .get("delivery_advisory", {})
                          .get("communication_events", []))
                gen_prompt += (
                    "\nWrite one concise, non-redundant coordination sentence "
                    "about this newly opened event only: " +
                    (", ".join(events) if events else "none") +
                    ". Do not restate your destination, room search, or "
                    "container intent if the peer already knows it."
                )
            elif common_v4:
                card = getattr(self, "peer_decision_card", None) or {}
                gen_prompt += (
                    "\nChoose a public event with coordination_event:<event_id>, or "
                    "write coordination_intent: followed by one JSON object. To propose "
                    "cooperation use {\"kind\":\"propose\",\"participants\":[0,1],"
                    "\"description\":\"your request\",\"conditions\":[]}; optional keys "
                    "task_id and location express your proposal. To respond use "
                    "{\"kind\":\"accept\",\"intent_id\":\"existing ID\"}; kinds decline, "
                    "ready and cancel also reference an existing intent_id. To withdraw "
                    "your work use {\"kind\":\"release_work\"}; it waits for running "
                    "execution to end. Speak only "
                    "for yourself. Proposals and ready statements do not establish facts. "
                    "Public events: " + json.dumps(card.get("coordination_events", [])) +
                    " Cooperation state: " + json.dumps(card.get("common_core", {}).get("coordination", []))
                )
                v4_event_ids = [event["event_id"] for event in card.get("coordination_events", [])
                                if event.get("event_id") is not None]
            elif is_v4:
                events = ((getattr(self, "peer_decision_card", None) or {})
                          .get("coordination_events", []))
                allowed_event_types = {
                    "task_commitment", "task_suspended", "task_released",
                    "task_failure", "task_progress", "task_completed",
                    "claim_acquired", "claim_released",
                }
                selectable_events = []
                for event in events:
                    if (not isinstance(event, dict) or
                            event.get("event") not in allowed_event_types or
                            event.get("event_id") is None):
                        continue
                    event_id = int(event["event_id"])
                    v4_event_ids.append(event_id)
                    selectable_events.append({
                        key: event.get(key)
                        for key in (
                            "event_id", "event", "agent", "task_id",
                            "outcome", "reason", "progress_version")
                        if event.get(key) is not None
                    })
                gen_prompt += (
                    "\nChoose exactly one public structural event to send. "
                    "Respond only with coordination_event:<event_id>; do not "
                    "write a sentence or add observations. Selectable events: "
                    + json.dumps(selectable_events, ensure_ascii=False,
                                 separators=(",", ":"))
                )
            gen_prompt += f"\n{self.agent_name}:"
            chat_prompt = [{"role": "user", "content": gen_prompt}]
            comm_outputs, comm_usage = self.generator(
                chat_prompt if self.chat else gen_prompt,
                self.sampling_params)
            self.total_cost += comm_usage
            if is_v4:
                message = _v4_coordination_event_reference(
                    comm_outputs[0], v4_event_ids)
                if common_v4 and message is None:
                    raw_message = str(comm_outputs[0]).strip()
                    if raw_message.startswith("coordination_intent:"):
                        try:
                            payload = json.loads(raw_message.split(":", 1)[1])
                            if isinstance(payload, dict):
                                message = "coordination_intent:" + json.dumps(payload, ensure_ascii=False)
                        except (ValueError, TypeError):
                            pass
                communication_guard = {
                    "accepted": message is not None,
                    "reason": ("accepted_structured_intent"
                               if message is not None and message.startswith("coordination_intent:")
                               else "accepted_structured_event"
                               if message is not None else
                               "invalid_structured_event_reference"),
                    "sender_role": self.agent_role,
                    "receiver_role": self.opponent_role,
                }
            else:
                message = _normalize_communication_message(
                    comm_outputs[0],
                    allow_unquoted=self.fix_lm_communication)
                message, communication_guard = guard_communication_message(
                    message, self.agent_role, self.opponent_role)
            info['prompt_comm'] = gen_prompt
            info['output_comm'] = comm_outputs
            info['usage_comm'] = comm_usage
            info['communication_guard'] = communication_guard
            if message is None:
                # Replan once from physical actions on a rejected/empty
                # candidate without repeating communication generation.
                self.allow_message_this_turn = False
                fallback_text, fallback_num, fallback_actions = (
                    self.get_available_plans(include_message=False))
                if fallback_num:
                    fallback_prompt = decision_prompt_template.replace(
                        '$AVAILABLE_ACTIONS$', fallback_text)
                    fallback_prompt += (
                        "\nThe proposed message was empty or invalid. Choose "
                        "one physical next action. Answer with only the option.")
                    fallback_chat = [{"role": "user",
                                      "content": fallback_prompt}]
                    fallback_outputs, fallback_usage = self.generator(
                        fallback_chat if self.chat else fallback_prompt,
                        self.sampling_params)
                    self.total_cost += fallback_usage
                    plan, fallback_flags = self.parse_answer(
                        fallback_actions, fallback_outputs[0])
                    info['prompt_plan_after_rejected_message'] = (
                        fallback_prompt)
                    info['output_plan_after_rejected_message'] = (
                        fallback_outputs[0])
                    info['usage_plan_after_rejected_message'] = (
                        fallback_usage)
                    flags = f"{flags}; message fallback: {fallback_flags}"
                else:
                    plan = None
            else:
                plan = f"send a message: {message}"
            if self.debug:
                print(f"prompt_comm:\n{gen_prompt}")
                print(f"output_comm:\n{message}")
                if not communication_guard['accepted']:
                    print(
                        "COMMUNICATION_GUARD rejected candidate: "
                        f"{communication_guard['reason']}")
        if self.debug:
            print(f"plan: {plan}\n")
        info.update({"num_available_actions": num,
                     "prompt_plan_stage_2": normal_prompt,
                     "output_plan_stage_2": output,
                     "parse_exception": flags,
                     "plan": plan,
                     "total_cost": self.total_cost})
        return plan, info
