import asyncio

class Barrier(object):
    def __init__(self, parties, action=lambda: None):
        self._parties = parties
        self._action = action

        self._cond = asyncio.Condition()
        self._count = 0

    async def wait(self):
        self._count += 1
        with (await self._cond):
            if self._maybe_release():
                return
            await self._cond.wait()

    async def deregister(self):
        self._parties -= 1
        with (await self._cond):
            self._maybe_release()

    @property
    def empty(self):
        return self._parties == 0

    @property
    def n_waiting(self):
        return self._count

    @property
    def parties(self):
        return self._parties

    def _maybe_release(self):
        if self._count == self._parties:
            # Release everyone
            self._cond.notify_all()
            self._count = 0
            self._action()
            return True
        return False

