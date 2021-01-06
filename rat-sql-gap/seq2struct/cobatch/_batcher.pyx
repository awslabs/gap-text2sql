import asyncio
import collections
import functools
import itertools

import torch

from . import barrier


cdef class ResultHandle:
    cdef:
        coro
        Node node
        dict all_results
        accessor
        readonly shape

    def __init__(
            self, 
            coro,
            Node node,
            dict all_results,
            accessor=lambda x: x):
        self.coro = coro
        self.node = node
        self.all_results = all_results
        self.accessor = accessor
        self.shape = None

    def __await__(self):
        result = self.all_results.get(self.node)
        if result is None:
            yield from self.coro().__await__()
            result = self.all_results[self.node]
        return self.accessor(result)

    def with_shape(self, *shape):
        self.shape = shape
        return self

    def split(self, num_splits):
        cdef:
            list result
            int i
            ResultHandle copied

        result = []
        for i in range(num_splits):
            copied = ResultHandle(
                    self.coro, self.node, self.all_results, self.accessor)
            # Without default argument, captured `i` is always num_splits - 1.
            copied.accessor = lambda x, i=i: self.accessor(x)[i]
            result.append(copied)
        return tuple(result)


cdef class BatchKey:
    cdef:
        callable
        tuple arg_shapes
        tuple kwarg_shapes
        tuple _callable_exp
        long _hash
    
    def __init__(self,
            callable,
            tuple arg_shapes,
            tuple kwarg_shapes):
        self.callable = callable
        self.arg_shapes = arg_shapes
        self.kwarg_shapes = kwarg_shapes

        if isinstance(callable, functools.partial):
            callable_exp = (
                callable.func, callable.args,
                tuple(
                    (k, v) for k, v in sorted(callable.keywords.items())))
        else:
            callable_exp = (callable, (), ())
        self._callable_exp = callable_exp
        self._hash = hash((callable_exp, self.arg_shapes, self.kwarg_shapes))

    def __eq__(self, other):
        if not isinstance(other, BatchKey):
            return False
        return self._same_type_eq(other)

    cdef _same_type_eq(self, BatchKey other):
        return (self._callable_exp == other._callable_exp and
                self.arg_shapes == other.arg_shapes and
                self.kwarg_shapes == other.kwarg_shapes)

    def __hash__(self):
        return self._hash


cdef class Node:
    cdef:
        tuple args
        dict kwargs
        BatchKey batch_key
        int depth
        list outgoing
        int num_incoming

    def __init__(
            self, 
            tuple args, 
            dict kwargs, 
            BatchKey batch_key, 
            int depth=0, 
            list outgoing=None,
            int num_incoming=0):
        self.args = args
        self.kwargs = kwargs

        self.batch_key = batch_key
        self.depth = depth

        if outgoing is  None:
            self.outgoing = []
        else:
            self.outgoing = outgoing
        self.num_incoming = num_incoming

    cdef decrement_incoming(self):
        self.num_incoming -= 1


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


cdef class TorchBatcher:
    cdef:
        barrier
        list enqueued_nodes
        dict results
        mean_depth_by_key

    def __init__(self):
        self.barrier = None
        self._reset()

    cdef _reset(self):
        self.enqueued_nodes = []
        self.results = {}
        self.mean_depth_by_key = collections.defaultdict(StreamingMean)

    def __call__(self, callable, *args, **kwargs):
        cdef:
            BatchKey batch_key
            Node node

        batch_key = self._batch_key(callable, args, kwargs)
        node = Node(args, kwargs, batch_key)
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, ResultHandle):
                self._process_result_handle_arg(arg, node)
        self.enqueued_nodes.append(node)
        self.mean_depth_by_key[batch_key].add(node.depth)
        coro = ResultHandle(self.barrier.wait, node, self.results)
        return coro
    
    cdef _process_result_handle_arg(self, ResultHandle arg, Node node):
        node.num_incoming += 1
        node.depth = max(node.depth, arg.node.depth + 1)
        arg.node.outgoing.append(node)

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
        cdef:
            Node node
            list remaining_nodes
            BatchKey batch_key
            list nodes
            list args
            dict kwargs
            Node next_node

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
            batch_key = min(
                    agenda, key=lambda k: self.mean_depth_by_key[k].value)
            nodes = agenda[batch_key]
            args = [
                    self._stack(
                        [self._to_value(node.args[i]) for node in nodes])
                    for i in range(len(batch_key.arg_shapes))]
            kwargs = {
                k: self._stack(
                    [self._to_value(node.kwargs[k]) for node in nodes])
                for k, _ in batch_key.kwarg_shapes}

            results = self._unstack(batch_key.callable(*args, **kwargs))
            for node, result in zip(nodes, results):
                self.results[node] = result
                for next_node in node.outgoing:
                    next_node.decrement_incoming()
            del agenda[batch_key]

        self._reset()

    cdef BatchKey _batch_key(self, callable, tuple args, dict kwargs):
        return BatchKey(
            callable,
            tuple(self._batch_key_single(arg) for arg in args),
            tuple((k, self._batch_key_single(v))
                  for k, v in sorted(kwargs.items())))

    cdef _to_value(self, handle_or_value):
        if isinstance(handle_or_value, ResultHandle):
            return self._to_value_result_handle(handle_or_value)
        return handle_or_value

    cdef _to_value_result_handle(self, ResultHandle handle):
        return handle.accessor(self.results[handle.node])

    cdef _stack(self, list items):
        return torch.stack(items)

    cdef _unstack(self, stacked):
        if isinstance(stacked, tuple):
            return zip(*(
                [piece.squeeze(0) for piece in stacked_elem.split(1)]
                for stacked_elem in stacked))
        return [piece.squeeze(0) for piece in stacked.split(1)]

    cdef _batch_key_single(self, arg):
        if arg.shape is None:
            raise ValueError('shape is undefined')
        return arg.shape


cdef class NoOpResultHandle:
    cdef:
        result
        accessor
        shape

    def __init__(
            self, 
            result,
            accessor=lambda x: x):
        self.result = result
        self.accessor = accessor
        self.shape = None

    def __await__(self):
        if False:
            yield
        return self.accessor(self.result)

    def with_shape(self, *shape):
        self.shape = shape
        return self

    def split(self, num_splits):
        cdef:
            list result
            int i
            NoOpResultHandle copied

        result = []
        for i in range(num_splits):
            # Without default argument, captured `i` is always num_splits - 1.
            copied = NoOpResultHandle(self.result, 
                    lambda x, i=i: self.accessor(x)[i])
            result.append(copied)
        return tuple(result)


cdef class TorchNoOpBatcher(TorchBatcher):
    def __call__(self, callable, *args, **kwargs):
        args = [self._noop_stack(arg) for arg in args]
        kwargs = {k: self._noop_stack(arg) for k, arg in kwargs.items()}
        return NoOpResultHandle(self._noop_unstack(callable(*args, **kwargs)))

    def _noop_stack(self, item):
        if isinstance(item, NoOpResultHandle):
            item = self._get_result(item)
        return torch.unsqueeze(item, 0)

    def _noop_unstack(self, stacked):
        if isinstance(stacked, tuple):
            return tuple(torch.squeeze(piece, 0) for piece in stacked)
        return torch.squeeze(stacked, 0)

    def _get_result(self, NoOpResultHandle result):
        return result.accessor(result.result)
