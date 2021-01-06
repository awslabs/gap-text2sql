import asyncio
import collections
import copy
import functools
import itertools

import attr
import torch

from . import barrier


@attr.s
class ResultHandle(object):
    coro = attr.ib()
    node = attr.ib()
    all_results = attr.ib()
    accessor = attr.ib(default=lambda x: x)

    def __await__(self):
        result = self.all_results.get(self.node)
        if result is None:
            yield from self.coro().__await__()
            result = self.all_results[self.node]
        return self.accessor(result)

    def with_shape(self, *shape):
        copied = copy.copy(self)
        copied.shape = shape
        return copied

    def split(self, num_splits):
        result = []
        for i in range(num_splits):
            copied = copy.copy(self)
            # Without default argument, captured `i` is always num_splits - 1.
            copied.accessor = lambda x, i=i: self.accessor(x)[i]
            result.append(copied)
        return tuple(result)


@attr.s(frozen=True, cmp=False, hash=False)
class BatchKey(object):
    callable = attr.ib()
    args = attr.ib()
    kwargs = attr.ib()

    def __attrs_post_init__(self):
        if isinstance(self.callable, functools.partial):
            callable_exp = (self.callable.func, self.callable.args,
                            tuple((k, v) for k, v in sorted(self.callable.keywords.items())))
        else:
            callable_exp = (self.callable, (), ())
        self.__dict__['_callable_exp'] = callable_exp
        self.__dict__['_hash'] = hash((callable_exp, self.args, self.kwargs))

    def __eq__(self, other):
        if not isinstance(other, BatchKey):
            return False

        return (self._callable_exp == other._callable_exp and
                self.args == other.args and
                self.kwargs == other.kwargs)

    def __hash__(self):
        return self._hash


@attr.s(cmp=False)
class Node(object):
    args = attr.ib()
    kwargs = attr.ib()

    batch_key = attr.ib()
    depth = attr.ib(default=0)

    outgoing = attr.ib(default=attr.Factory(list))
    num_incoming = attr.ib(default=0)



class StreamingMean(object):
    def __init__(self):
        self.value = None
        self.count = 0.0

    def add(self, value):
        if not self.count:
            self.value = value
        else:
            self.value *= self.count / (self.count + 1)
            self.value += value / (self.count + 1)
        self.count += 1


class TorchBatcher(object):
    def __init__(self):
        self.barrier = None
        self._reset()

    def _reset(self):
        self.enqueued_nodes = []
        self.results = {}
        self.mean_depth_by_key = collections.defaultdict(StreamingMean)

    def __call__(self, callable, *args, **kwargs):
        batch_key = self._batch_key(callable, *args, **kwargs)
        node = Node(args, kwargs, batch_key)
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, ResultHandle):
                node.num_incoming += 1
                node.depth = max(node.depth, arg.node.depth + 1)
                arg.node.outgoing.append(node)
        self.enqueued_nodes.append(node)
        self.mean_depth_by_key[batch_key].add(node.depth)
        coro = ResultHandle(self.barrier.wait, node, self.results)
        return coro

    async def _wrap_deregister(self, coroutine):
        result = await coroutine
        await self.barrier.deregister()
        return result

    def run(self, coroutines):
        # Set up barrier depending on how many coroutines exist
        self.barrier = barrier.Barrier(len(coroutines), self._compute)

        # Run everything
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(asyncio.gather(*(
            self._wrap_deregister(c) for c in coroutines)))

    def _compute(self):
        agenda = collections.defaultdict(list)
        while self.enqueued_nodes or agenda:
            # Update agenda
            remaining_nodes = []
            while self.enqueued_nodes:
                node = self.enqueued_nodes.pop()
                if node.num_incoming == 0:
                    agenda[node.batch_key].append(node)
                else:
                    remaining_nodes.append(node)
            self.enqueued_nodes = remaining_nodes

            # Run computation on best group of nodes from agenda
            batch_key = min(agenda, key=lambda k: self.mean_depth_by_key[k].value)
            nodes = agenda[batch_key]
            args = [self._stack([self._to_value(node.args[i]) for node in nodes])
                    for i in range(len(batch_key.args))]
            kwargs = {
                k: self._stack([self._to_value(node.kwargs[k]) for node in nodes])
                for k, shape in batch_key.kwargs}

            results = self._unstack(batch_key.callable(*args, **kwargs))
            for node, result in zip(nodes, results):
                self.results[node] = result
                for next_node in node.outgoing:
                    next_node.num_incoming -= 1
            del agenda[batch_key]

        self._reset()

    def _batch_key(self, callable, *args, **kwargs):
        return BatchKey(
            callable,
            tuple(self._batch_key_single(arg) for arg in args),
            tuple((k, self._batch_key_single(v))
                  for k, v in sorted(kwargs.items())))

    def _to_value(self, handle_or_value):
        if isinstance(handle_or_value, ResultHandle):
            # Avoid warnings about "was never awaited"
            # by starting and immediately stopping the coroutine
            return handle_or_value.accessor(self.results[handle_or_value.node])
        return handle_or_value

    def _stack(self, items):
        return torch.stack(items)

    def _unstack(self, stacked):
        if isinstance(stacked, tuple):
            return zip(*(
                [piece.squeeze(0) for piece in stacked_elem.split(1)]
                for stacked_elem in stacked))
        return [piece.squeeze(0) for piece in stacked.split(1)]

    def _batch_key_single(self, arg):
        return arg.shape


class TorchNoOpBatcher(TorchBatcher):
    async def __call__(self, callable, *args, **kwargs):
        args = [self._noop_stack(arg) for arg in args]
        kwargs = {k: self._noop_stack(arg) for k, arg in kwargs.items()}
        return self._noop_unstack(callable(*args, **kwargs))

    def _noop_stack(self, item):
        return torch.unsqueeze(item, 0)

    def _noop_unstack(self, stacked):
        return torch.squeeze(stacked, 0)
