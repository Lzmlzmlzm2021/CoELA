import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tdw-gym')]
from LLM.react import MESSAGE_PATTERN, POLICY_VERSION, REASON_PATTERN, ReActLLM


def empty_hand():
    return dict(id=None, name=None, type=None, contained=[None]*3)


class ReActTests(unittest.TestCase):
    def make(self, role='human', communication=False):
        with patch('LLM.react.OpenAI') as constructor:
            planner = ReActLLM('openai', 'test-model', 'unused', communication,
                              False, SimpleNamespace(), 0, agent_role=role)
        planner.reset(['kitchen', 'bedroom'], {'apple': 1})
        planner.client = constructor.return_value
        planner.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"reason":"explore","action_id":"a0","message":"-"}'))],
            usage=SimpleNamespace(model_dump=lambda: dict(prompt_tokens=20, completion_tokens=10, total_tokens=30)))
        return planner

    def run_plan(self, planner, **overrides):
        values = dict(current_step=10, current_room='kitchen', rooms_explored={},
                      holding_objects=[empty_hand(),empty_hand()], satisfied=[],
                      object_list={0:[],1:[],2:[]}, obj_per_room={},
                      action_history=[], dialogue_history=[], feedback={'status':1,'valid':True})
        values.update(overrides)
        return planner.run(**values)

    def test_single_call_native_navigation_and_bounded_feedback(self):
        planner = self.make()
        for i in range(12):
            plan, _ = self.run_plan(planner, current_step=i)
        self.assertEqual(plan, 'go to bedroom')
        self.assertEqual(planner.client.chat.completions.create.call_count,12)
        self.assertEqual(len(planner.history),8)
        self.assertEqual(planner.history[-1]['feedback']['status'],1)
        planner.reset(['other'], {'banana':1})
        self.assertEqual(planner.history,[])
        self.assertIsNone(planner.previous)
        self.assertEqual(planner.rooms,['other'])
        self.assertEqual(planner.metrics,{})

    def test_native_role_filter_and_no_shared_target(self):
        planner = self.make('scout')
        self.run_plan(planner, object_list={0:[dict(id=1,name='apple')],1:[],2:[dict(id=2,name='bed')]})
        context = json.loads(planner.client.chat.completions.create.call_args.kwargs['messages'][1]['content'])
        actions = list(context['available_actions'].values())
        self.assertIn('wait',actions)
        self.assertFalse(any('grasp' in x or 'transport' in x for x in actions))
        self.assertIsNone(planner.shared_delivery_target())

    def test_container_and_transport_candidates_use_native_preconditions(self):
        planner = self.make()
        held = [dict(id=2,name='basket',type=1,contained=[None]*3),
                dict(id=1,name='apple',type=0,contained=[None]*3)]
        self.run_plan(planner,holding_objects=held,object_list={0:[],1:[],2:[dict(id=3,name='bed')]})
        context = json.loads(planner.client.chat.completions.create.call_args.kwargs['messages'][1]['content'])
        actions = list(context['available_actions'].values())
        self.assertTrue(any(x.startswith('put <apple>') for x in actions))
        self.assertIn("transport objects I'm holding to the bed",actions)
        self.assertFalse(any(x.startswith('go grasp') for x in actions))

    def test_message_is_one_call_and_exact_native_action(self):
        planner = self.make(communication=True)
        planner.client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
            dict(reason='share',action_id='a0',message='Apple in kitchen'))
        plan, _ = self.run_plan(planner)
        self.assertEqual(plan,'send a message: Apple in kitchen')
        self.assertEqual(planner.client.chat.completions.create.call_count,1)

    def test_invalid_output_is_logged_wait_not_random_action(self):
        planner = self.make()
        planner.client.chat.completions.create.return_value.choices[0].message.content = '{"reason":"x","action_id":"a999","message":"-"}'
        with tempfile.TemporaryDirectory() as folder:
            planner.set_output_dir(folder)
            plan, info = self.run_plan(planner)
            self.assertEqual(plan,'wait')
            self.assertEqual(info['metrics']['invalid_decisions'],1)
            row = json.loads(planner.audit_path.read_text(encoding='utf-8'))
            self.assertIn('current candidate',row['error'])
        with self.assertRaises(ValueError):
            ReActLLM.parse('{"reason":"share","action_id":"a0","message":""}', ['send a message'])

    def test_exhausted_api_errors_do_not_fabricate_action(self):
        planner = self.make()
        planner.max_retries = 0
        planner.client.chat.completions.create.side_effect = TimeoutError('test')
        with self.assertRaises(RuntimeError):
            self.run_plan(planner)
        self.assertIsNone(planner.previous)

    def test_agent_factory_and_executor_protocol_isolation(self):
        from lm_agent import lm_agent
        args = SimpleNamespace(planner='react', source='openai', lm_id='test',
                               prompt_template_path='unused', communication=True, cot=False)
        with patch('LLM.react.OpenAI'):
            agent = lm_agent(0, Mock(), 3000, args)
        self.assertIsInstance(agent.LLM,ReActLLM)
        self.assertTrue(agent._uses_episode_isolation_protocol())
        self.assertFalse(agent._uses_v4_protocol())
        self.assertFalse(agent._uses_v34_protocol())
        self.assertFalse(agent.stable_task_protocol)
        self.assertFalse(agent.coverage_exploration)
        agent.gotoroom = Mock(return_value={'type':0})
        agent.gotoroom_v34 = Mock(side_effect=AssertionError('V4 executor'))
        self.assertEqual(agent._execute_room_navigation_plan(),{'type':0})


    def test_ongoing_skips_model_and_invalid_selection_executes_native_wait(self):
        import numpy as np
        from lm_agent import lm_agent
        args = SimpleNamespace(planner='react', source='openai', lm_id='test',
                               prompt_template_path='unused', communication=False, cot=False)
        with patch('LLM.react.OpenAI'):
            agent = lm_agent(0, Mock(), 3000, args)
        agent.LLM = self.make()
        obs = dict(agent=np.array([0.,0.,0.,0.,0.,1.]),rgb=np.zeros((3,4,4)),
                   current_frames=10, status=0, valid=True, messages=[None,None],
                   held_objects=[empty_hand(),empty_hand()],
                   oppo_held_objects=[empty_hand(),empty_hand()],visible_objects=[])
        with tempfile.TemporaryDirectory() as folder, patch('lm_agent.AgentMemory',return_value=Mock()):
            agent.reset(obs=obs,goal_objects={'apple':1},output_dir=folder,
                        env_api={'belongs_to_which_room':lambda _: 'kitchen',
                                 'get_room_distance':lambda _: 0},
                        rooms_name=['kitchen','bedroom'],save_img=False)
            self.assertEqual(agent.act(obs),{'type':'ongoing'})
            agent.LLM.client.chat.completions.create.assert_not_called()
            agent.get_new_object_list = Mock()
            agent.get_object_list = Mock()
            agent.LLM.client.chat.completions.create.return_value.choices[0].message.content = '{}'
            obs['status'] = 1
            self.assertEqual(agent.act(obs),{'type':8,'delay':1})
            agent.LLM.client.chat.completions.create.assert_called_once()


    def test_missing_message_cannot_spin_forever(self):
        planner = self.make(communication=True)
        planner.client.chat.completions.create.return_value.choices[0].message.content = '{"reason":"share","action_id":"a0"}'
        for _ in range(2):
            plan, info = self.run_plan(planner)
            self.assertEqual(plan, 'wait')
        with self.assertRaisesRegex(RuntimeError, 'Three consecutive'):
            self.run_plan(planner)
        planner.reset(['kitchen', 'bedroom'], {'apple': 1})
        planner.client.chat.completions.create.return_value.choices[0].message.content = '{"reason":"share","action_id":"a0","message":"Apple in kitchen"}'
        self.assertTrue(self.run_plan(planner)[0].startswith('send a message:'))
        call = planner.client.chat.completions.create.call_args.kwargs
        schema = call['response_format']['json_schema']['schema']
        self.assertIn('message', schema['required'])
        self.assertEqual(schema['properties']['action_id']['enum'][0], 'a0')

    def test_message_cooldown_forces_model_to_choose_native_action_then_restores(self):
        planner = self.make(communication=True)
        result = planner.client.chat.completions.create.return_value
        result.choices[0].message.content = json.dumps(
            dict(reason='share', action_id='a0', message='Apple in kitchen'))
        with tempfile.TemporaryDirectory() as folder:
            planner.set_output_dir(folder)
            self.assertTrue(self.run_plan(planner)[0].startswith('send a message:'))
            plan, info = self.run_plan(planner, current_step=11,
                feedback={'last_low_level_action': {'type': 6}, 'status': 1, 'valid': True})
            self.assertEqual(plan, 'go to bedroom')
            context = json.loads(planner.client.chat.completions.create.call_args.kwargs['messages'][1]['content'])
            self.assertNotIn('send a message', context['available_actions'].values())
            self.assertTrue(context['communication_cooldown']['message_suppressed'])
            self.assertTrue(info['communication_cooldown']['active_after_decision'])
            # A failed physical action still counts as an attempt; no success oracle.
            plan, info = self.run_plan(planner, current_step=12,
                feedback={'last_low_level_action': {'type': 0}, 'status': 2, 'valid': False})
            self.assertTrue(plan.startswith('send a message:'))
            self.assertFalse(info['communication_cooldown']['active'])
            rows = [json.loads(line) for line in planner.audit_path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[1]['settings']['policy_version'], POLICY_VERSION)
            self.assertTrue(rows[1]['communication_cooldown']['message_suppressed'])

    def test_same_frame_replan_does_not_bypass_message_cooldown(self):
        planner = self.make(communication=True)
        self.run_plan(planner)
        self.run_plan(planner, current_step=11,
            feedback={'last_low_level_action': {'type': 6}})
        plan, info = self.run_plan(planner, current_step=11,
            feedback={'last_low_level_action': {'type': 0}})
        self.assertEqual(plan, 'go to bedroom')
        self.assertTrue(info['communication_cooldown']['message_suppressed'])

    def test_invalid_wait_does_not_release_message_cooldown(self):
        planner = self.make(communication=True)
        self.run_plan(planner)
        result = planner.client.chat.completions.create.return_value
        result.choices[0].message.content = '{}'
        self.assertEqual(self.run_plan(planner, current_step=11)[0], 'wait')
        result.choices[0].message.content = '{"reason":"explore","action_id":"a0","message":"-"}'
        plan, info = self.run_plan(planner, current_step=12,
            feedback={'last_low_level_action': {'type': 8}})
        self.assertEqual(plan, 'go to bedroom')
        self.assertTrue(info['communication_cooldown']['message_suppressed'])

    def test_reset_clears_message_cooldown_and_pending_execution(self):
        planner = self.make(communication=True)
        self.run_plan(planner)
        self.run_plan(planner, current_step=11)
        self.assertTrue(planner.communication_cooldown)
        self.assertEqual(planner.pending_non_message_frame, 11)
        planner.reset(['kitchen', 'bedroom'], {'apple': 1})
        self.assertFalse(planner.communication_cooldown)
        self.assertIsNone(planner.pending_non_message_frame)
        self.assertTrue(self.run_plan(planner)[0].startswith('send a message:'))

    def test_message_only_candidates_remain_available_during_cooldown(self):
        planner = self.make(communication=True)
        planner.reset(['kitchen'], {'apple': 1})
        for frame in (10, 11, 12):
            plan, info = self.run_plan(planner, current_step=frame,
                rooms_explored={'kitchen': 'all'},
                feedback={'last_low_level_action': {'type': 6}})
            self.assertTrue(plan.startswith('send a message:'))
            self.assertEqual(info['num_available_actions'], 1)
            self.assertFalse(info['communication_cooldown']['message_suppressed'])
        self.assertEqual(planner.metrics['invalid_decisions'], 0)

    def test_schema_and_parser_enforce_ascii_length_bounds(self):
        schema = ReActLLM.response_format(['send a message'])['json_schema']['schema']
        self.assertEqual(schema['properties']['reason']['pattern'], REASON_PATTERN)
        self.assertEqual(schema['properties']['message']['pattern'], MESSAGE_PATTERN)
        decision = dict(reason='r' * 120, action_id='a0', message='m' * 240)
        self.assertEqual(ReActLLM.parse(json.dumps(decision), ['send a message'])[1], decision)
        for field, value in [('reason', 'r' * 121), ('message', 'm' * 241),
                             ('reason', '\u4f60\u597d'), ('message', 'hello\nworld'),
                             ('reason', ''), ('message', '  ')]:
            with self.subTest(field=field, value=value):
                invalid = dict(decision, **{field: value})
                with self.assertRaisesRegex(ValueError, 'printable ASCII'):
                    ReActLLM.parse(json.dumps(invalid), ['send a message'])
        # The same contract applies to physical actions when a backend ignores schema.
        with self.assertRaisesRegex(ValueError, 'exactly'):
            ReActLLM.parse('{"reason":"go","action_id":"a0"}', ['go to bedroom'])


if __name__ == '__main__':
    unittest.main()
