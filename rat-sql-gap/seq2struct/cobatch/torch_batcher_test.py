import unittest.mock
import random

import pytest
import torch

from . import torch_batcher

def test_simple_linear():
    batch_size = 32
    linear = unittest.mock.Mock(wraps=torch.nn.Linear(8, 16))
    inp = torch.autograd.Variable(torch.rand(batch_size, 8))

    batcher = torch_batcher.TorchBatcher()
    async def process(item):
        result = await batcher(linear, item)
        return result

    results = batcher.run([process(inp[i]) for i in range(batch_size)])
    assert linear.call_count == 1

    for i in range(batch_size):
        single_result = linear(inp[i:i+1]).squeeze(0)
        assert results[i].data.numpy() == pytest.approx(
            single_result.data.numpy(), abs=1e-6)


def test_simple_linear_defer():
    batch_size = 4
    linear = unittest.mock.Mock(wraps=torch.nn.Linear(8, 16))
    inp = torch.autograd.Variable(torch.rand(batch_size, 8))

    batcher = torch_batcher.TorchBatcher()
    async def process(item1, item2):
        result1 = batcher(linear, item1)
        result2 = batcher(linear, item2)
        return await result1, await result2

    results = batcher.run([process(inp[i], inp[i + 1]) for i in range(0, batch_size, 2)])
    assert linear.call_count == 1
    results = [r for pair in results for r in pair]

    for i in range(batch_size):
        single_result = linear(inp[i:i+1]).squeeze(0)
        assert results[i].data.numpy() == pytest.approx(
            single_result.data.numpy(), abs=1e-6)


def test_multi_stage():
    batch_size = 32
    linear = unittest.mock.Mock(wraps=torch.nn.Linear(8, 8))
    inp = torch.autograd.Variable(torch.rand(batch_size, 8))

    batcher = torch_batcher.TorchBatcher()
    async def process(item, iters):
        for iter in range(iters):
            item = await batcher(linear, item)
        return item

    results = batcher.run(
        [process(inp[i], (i + 1) // 4) for i in range(batch_size)])
    assert linear.call_count == 32 // 4

    for i in range(batch_size):
        row = inp[i:i+1]
        for iter in range((i + 1) // 4):
            row = linear(row)
        single_result = row.squeeze(0)
        assert results[i].data.numpy() == pytest.approx(
                single_result.data.numpy(), abs=1e-6)


def test_multi_stage_deferred():
    batch_size = 32
    linear = unittest.mock.Mock(wraps=torch.nn.Linear(8, 8))
    inp = torch.autograd.Variable(torch.rand(batch_size, 8))

    batcher = torch_batcher.TorchBatcher()
    async def process(item, iters):
        for iter in range(iters):
            item = batcher(linear, item).with_shape(8)
        if iters:
            return await item
        return item

    results = batcher.run(
        [process(inp[i], (i + 1) // 4) for i in range(batch_size)])
    assert linear.call_count == 32 // 4

    for i in range(batch_size):
        row = inp[i:i+1]
        for iter in range((i + 1) // 4):
            row = linear(row)
        single_result = row.squeeze(0)
        assert results[i].data.numpy() == pytest.approx(
            single_result.data.numpy(), abs=1e-6)

def test_multi_stage_and_modules():
    batch_size = 32
    linears = [
        unittest.mock.Mock(wraps=torch.nn.Linear(8, 8))
        for _ in range(5)]
    inp = torch.autograd.Variable(torch.rand(batch_size, 8))

    batcher = torch_batcher.TorchBatcher()
    async def process(i, item, iters):
        for iter in range(iters):
            item = await batcher(linears[(i + iter) % len(linears)], item)
        return item

    results = batcher.run(
        [process(i, inp[i], (i + 1) // 4) for i in range(batch_size)])
    for i in range(len(linears)):
        # TODO: Figure out exact call count
        #assert linears[i].call_count == max((j + 1) // 4 for j in range(
        #    batch_size) if j % len(linears) == i)
        assert linears[i].call_count <= 8

    for i in range(batch_size):
        row = inp[i:i+1]
        for iter in range((i + 1) // 4):
            row = linears[(i + iter) % len(linears)](row)
        single_result = row.squeeze(0)
        assert results[i].data.numpy() == pytest.approx(single_result.data.numpy(), abs=1e-6)

def test_multi_args():
    batch_size = 32
    add = unittest.mock.Mock(wraps=torch.nn.Bilinear(1, 1, 1))
    inp = torch.autograd.Variable(torch.arange(batch_size).view(-1, 1))

    batcher = torch_batcher.TorchBatcher()
    async def process(item, iters):
        for iter in range(iters):
            item = await batcher(add, item, input2=item)
        return item

    results = batcher.run(
        [process(inp[i], (i + 1) // 8) for i in range(batch_size)])
    assert add.call_count == 4

    for i in range(batch_size):
        row = inp[i:i+1]
        for iter in range((i + 1) // 8):
            row = add(row, row)
        single_result = row.squeeze(0)
        assert results[i].data.numpy() == pytest.approx(single_result.data.numpy(), abs=1e-6)

def test_multi_args_deferred():
    batch_size = 32
    add = unittest.mock.Mock(wraps=torch.nn.Bilinear(1, 1, 1))
    inp = torch.autograd.Variable(torch.arange(batch_size).view(-1, 1))

    batcher = torch_batcher.TorchBatcher()
    async def process(item, iters):
        for iter in range(iters):
            item = batcher(add, item, input2=item).with_shape(1)
        if iters:
            return await item
        else:
            return item

    results = batcher.run(
        [process(inp[i], (i + 1) // 8) for i in range(batch_size)])
    assert add.call_count == 4

    for i in range(batch_size):
        row = inp[i:i+1]
        for iter in range((i + 1) // 8):
            row = add(row, row)
        single_result = row.squeeze(0)
        assert results[i].data.numpy() == pytest.approx(single_result.data.numpy(), abs=1e-6)


def test_multi_args_mixed_deferred():
    batch_size = 6
    add = unittest.mock.Mock(wraps=torch.nn.Bilinear(1, 1, 1))
    inp = torch.autograd.Variable(torch.arange(batch_size).view(-1, 1))
    double = lambda x: x * 2
    # args: multiple N x 1 tensors
    sum = lambda *args: torch.sum(torch.cat(args, dim=1), dim=1)

    batcher = torch_batcher.TorchBatcher()
    async def process(item, iters):
        to_sum = []
        for iter in range(iters):
            item = batcher(add, item, input2=item).with_shape(1)
            to_sum.append(batcher(double, item).with_shape(1))
            item = await item

        return item, await batcher(sum, *to_sum)

    num_iters = [1, 1, 2, 2, 3, 3]
    results = batcher.run(
        [process(inp[i], num_iters[i]) for i in range(batch_size)])
    assert add.call_count == 3

    for i in range(batch_size):
        row = inp[i:i+1]
        to_sum = []
        for iter in range(num_iters[i]):
            row = add(row, row)
            to_sum.append(double(row))
        single_result1 = row.squeeze(0)
        single_result2 = sum(*to_sum)
        assert results[i][0].data.numpy() == pytest.approx(single_result1.data.numpy(), abs=1e-6)
        assert results[i][1].data.numpy() == pytest.approx(single_result2.data.numpy(), abs=1e-6)


def test_multi_shape():
    sizes = [i // 4 + 1 for i in range(32)]
    random.seed(32)
    random.shuffle(sizes)
    inps = []
    for size in sizes:
        inps.append(torch.rand(size))

    with unittest.mock.patch('torch.exp', wraps=torch.exp) as mock:
        batcher = torch_batcher.TorchBatcher()
        async def process(item):
            return await batcher(torch.exp, item)
        results = batcher.run([process(inp) for inp in inps])
        assert mock.call_count == 8

    for inp, result in zip(inps, results):
        assert torch.exp(inp).numpy() == pytest.approx(result.numpy())

def test_partial_softmax():
    import functools
    batch_size = 32
    inp = torch.autograd.Variable(torch.rand(batch_size, 8))
    torch_softmax = functools.partial(
        unittest.mock.Mock(wraps=torch.nn.functional.softmax), dim=-1)

    batcher = torch_batcher.TorchBatcher()
    async def process(item):
        return await batcher(torch_softmax, item)

    results = batcher.run([process(inp[i]) for i in range(batch_size)])
    assert torch_softmax.func.call_count == 1

    for i in range(batch_size):
        single_result = torch_softmax(inp[i:i+1]).squeeze(0)
        assert results[i].data.numpy() == pytest.approx(
            single_result.data.numpy(), abs=1e-6)

def test_partial_max():
    import functools
    batch_size = 3
    inp = torch.autograd.Variable(torch.rand(batch_size, 8))
    torch_max = functools.partial(
        unittest.mock.Mock(wraps=torch.max), dim=-1)
    torch_get = lambda x, i: x[range(x.shape[0]), i.view(-1)]
    double = lambda x: x * 2

    batcher = torch_batcher.TorchBatcher()
    async def process(item):
        max_value, max_idx = batcher(torch_max, item).split(2)
        max_idx = max_idx.with_shape(1)

        doubled_idx = batcher(double, max_idx)
        max_value2 = batcher(torch_get, item, max_idx)
        max_value = await max_value
        max_idx = await max_idx
        doubled_idx = await doubled_idx
        max_value2 = await max_value2
        assert max_value.data[0] == max_value2.data[0]

        max_value3, _ = batcher(torch_max, item).split(2)
        max_value3 = await max_value3
        assert max_value.data[0] == max_value3.data[0]

        return max_value, max_idx, doubled_idx

    results = batcher.run([process(inp[i]) for i in range(batch_size)])
    assert torch_max.func.call_count == 2

    for i in range(batch_size):
        max_value, max_idx = torch_max(inp[i])
        doubled_idx = double(max_idx)

        assert results[i][0].data.numpy() == pytest.approx(
            max_value.data.numpy(), abs=1e-6)
        assert results[i][1].data.numpy() == pytest.approx(
            max_idx.data.numpy(), abs=1e-6)
        assert results[i][2].data.numpy() == pytest.approx(
            doubled_idx.data.numpy(), abs=1e-6)
