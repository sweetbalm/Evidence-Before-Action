"""Server-mode scheduler registered with the checked-out ms-swift API."""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import traceback

from swift.infer_engine.protocol import RolloutOutput
from swift.rollout import MultiTurnScheduler

from osworld_opd.core import Trace, evidence_regions, load_config, load_task, parse_action, validate_turn
from osworld_opd.environment import OSWorldEnvironment


async def blocking_call(function, *args):
    """Keep one VM operation in flight, including on coroutine cancellation.

    Python cannot safely cancel a running provider thread. Drain that operation
    before the finally block closes its VM. Provider HTTP timeouts still apply.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class OSWorldScheduler(MultiTurnScheduler):
    def __init__(self, *args, config=None, environment_factory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or load_config()
        self.environment_factory = environment_factory or OSWorldEnvironment
        self.slots = asyncio.Semaphore(self.config['max_concurrent_envs'])

    async def run(self, infer_request, request_config, **kwargs):
        if self.infer_engine is None:
            raise RuntimeError('osworld_opd requires swift rollout server mode')
        if request_config.n != 1 or request_config.stream:
            raise ValueError('OSWorld requires n=1 and stream=false')
        async with self.slots:
            return await self._episode(infer_request, request_config, **kwargs)

    async def _episode(self, infer_request, request_config, **kwargs):
        request = copy.deepcopy(infer_request)
        task_path = request.data_dict.get('osworld_task')
        if not task_path:
            raise ValueError('Dataset osworld_task is missing; set vllm_server_pass_dataset=true')
        task = load_task(self.config['task_root'], task_path)
        if len(request.messages) != 2 or [m['role'] for m in request.messages] != ['system', 'user']:
            raise ValueError('Dataset rows must contain only system and user messages')
        if request.messages[-1]['content'] != task['instruction']:
            raise ValueError('Dataset instruction differs from task JSON; rebuild the manifest')
        if request.images or request.videos or request.audios:
            raise ValueError('Online dataset rows must not contain offline observations')
        trace = Trace(self.config['trace_root'])
        environment = self.environment_factory(self.config, trace)
        trace.snapshot('task.json', task)
        trace.snapshot('config.json', self.config)
        trace.event('start', request_id=request.uuid, task_path=task_path, task_id=task['id'],
                    request_config=dataclasses.asdict(request_config))
        all_ids, all_masks, all_lps = [], [], []
        image_files = []
        result = None
        try:
            observation = await blocking_call(environment.reset, task)
            image = trace.observation(observation, 0)
            request.images = [image]
            image_files.append('observation-000.png')
            request.messages[-1]['content'] = task['instruction'] + '\n<image>'
            limit = min(self.config['max_steps'], self.max_turns or self.config['max_steps'])
            reason = 'max_steps'
            for step in range(1, limit + 1):
                rc = copy.deepcopy(request_config)
                remaining = self.config['max_total_response_tokens'] - sum(map(len, all_lps))
                rc.max_tokens = min(rc.max_tokens or remaining, remaining)
                rc.logprobs = True
                rc.return_details = True
                # Freeze all historical assistant contents as IDs. Template text
                # normalization must not rewrite a previously sampled response.
                response = await self.infer_engine.infer_async(copy.deepcopy(request), rc, **kwargs)
                if len(response.choices) != 1:
                    raise ValueError('Expected exactly one rollout choice')
                choice = response.choices[0]
                ids, mask = self.get_response_token_data(request, choice)
                lps = [item['logprob'] for item in (choice.logprobs or {}).get('content', [])]
                validate_turn(choice.token_ids, ids, mask, lps)
                all_ids.append(list(ids))
                all_masks.append(list(mask))
                all_lps.append(lps)
                request.messages.append({'role': 'assistant', 'content': list(ids)})
                text = choice.message.content
                trace.event('generation', step=step, text=text,
                            sampled_ids=choice.token_ids, response_ids=ids, loss_mask=mask,
                            rollout_logprobs=lps, finish_reason=choice.finish_reason,
                            usage=dataclasses.asdict(response.usage),
                            prompt_token_ids=getattr(response, 'prompt_token_ids', None))
                if choice.finish_reason == 'length':
                    reason = 'generation_truncated'
                    break  # Never execute an incomplete model action.
                try:
                    action = parse_action(text)
                except (ValueError, SyntaxError) as error:
                    trace.event('parse_error', step=step, error=str(error))
                    reason = 'parse_error'
                    break
                await blocking_call(environment.capture_evidence, evidence_regions(text), step)
                trace.event('action', step=step, action=action)
                observation, reward, done, info = await blocking_call(environment.step, action)
                trace.event('transition', step=step, reward=float(reward), done=bool(done), info=info)
                image = trace.observation(observation, step)
                if done:
                    reason = 'done' if action == 'DONE' else 'fail' if action == 'FAIL' else 'environment_done'
                    break
                if sum(map(len, all_lps)) >= self.config['max_total_response_tokens']:
                    reason = 'response_budget'
                    break
                if step < limit:
                    request.messages.append({'role': 'user', 'content': 'Observation after your previous action:\n<image>'})
                    request.images.append(image)
                    image_files.append(f'observation-{step:03d}.png')
            score = None
            if self.config['evaluate_at_end']:
                # Evaluation may run postconfig. Do it exactly once, after rollout.
                score = await blocking_call(environment.evaluate)
            trace.event('end', reason=reason, num_turns=len(all_ids), score=score)
            trace.snapshot('rollout.json', {
                'trace_id': trace.id, 'request_id': request.uuid,
                'messages': request.messages, 'images': image_files,
                'chat_template_kwargs': request.chat_template_kwargs,
                'response_token_ids': all_ids, 'response_loss_mask': all_masks,
                'rollout_logprobs': all_lps, 'termination_reason': reason,
                'osworld_score': score,
            })
            result = RolloutOutput(
                response=response, messages=request.messages,
                response_token_ids=all_ids, response_loss_mask=all_masks,
                rollout_logprobs=all_lps,
                rollout_infos={'trace_id': trace.id, 'task_id': task['id'], 'num_turns': len(all_ids),
                               'termination_reason': reason, 'osworld_score': score,
                               'images': request.images})
        except BaseException as error:
            trace.event('error', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            try:
                await blocking_call(environment.close)
                trace.event('closed')
            except BaseException as error:
                trace.event('close_error', error=str(error))
                # Do not silently return a successful trajectory with leaked resources.
                raise
        return result
