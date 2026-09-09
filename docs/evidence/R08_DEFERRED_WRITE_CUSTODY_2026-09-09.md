# R08: A retry must not duplicate its pending write

The resident neural stream reported episodic queue overflow at 02:22:22,
then 60 shed entries by 02:32. The store retried an episode through its normal
write method. When admission deferred it again, that method enqueued a new
copy; the drain then restored the original. Repeated unsuccessful attempts
therefore grew the queue without new experience.

Two regression tests failed on the original code: one pending write became
two after a retry, and a retry that also introduced one new write retained
three instead of two.

The queue now retains identity custody across pending and in-flight work.
An episode supplies its existing idempotency key, so reconstructed retry
payloads identify the same obligation. A repeated hold does not allocate a
second slot. Capacity includes the in-flight item, preventing a concurrent
hold followed by restoration from silently discarding another entry.

Twenty-five deferral and custody tests passed in 4.65 seconds. They include
repeated failure, reconstructed payloads, genuinely new work during retry,
capacity pressure, exception handling, and nested replay without a lock held
over the write. Admission policy remains unchanged.

Smoke, lint, compile, and governance lint passed. Layering initially found
the newly shared percept reader absent from the world-model dependency list;
the specific `core.state.percepts` edge is now declared with its purpose.
Layering then passed with 37 grandfathered entries.

This repair does not establish that the governor eventually admits every
deferred write. Live queue drain and the continuing admission policy require
separate observation; R08 remains unchecked.
