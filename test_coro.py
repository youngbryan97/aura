import asyncio
import inspect

class A:
    async def foo(self): pass
    def bar(self): pass

a = A()
print("asyncio foo:", asyncio.iscoroutinefunction(a.foo))
print("inspect foo:", inspect.iscoroutinefunction(a.foo))
print("asyncio bar:", asyncio.iscoroutinefunction(a.bar))
print("inspect bar:", inspect.iscoroutinefunction(a.bar))
