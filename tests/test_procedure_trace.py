"""Execution dataflow joins the shared event graph without correctness labels."""

from core.cognition.cognitive_event import Epistemic, EventGraph, Phase
from core.cognition.procedure import Backend, Effect, Precondition, ProcedureRegistry, Signature, compose
from core.cognition.procedure_execution import BackendResult, execute_procedure


def _step(registry, name, reads=(), writes=(Effect("answer", "integer"),)):
    return registry.register(name, Backend.TOOL, Signature(tuple(reads), tuple(writes)))


def test_data_dependencies_link_actual_reads_and_exclude_unused_state():
    registry, graph = ProcedureRegistry(), EventGraph()
    source = graph.record(Phase.RETRIEVE, "knowledge", "source")
    first = _step(registry, "first", (Precondition("a", "integer"),), (Effect("x", "integer"),))
    unrelated = _step(registry, "unrelated", (), (Effect("y", "integer"),))
    final = _step(registry, "final", (Precondition("x", "integer"),))
    chain = compose(registry, (first, unrelated, final))

    def backend(p, s, context):
        if p.procedure_id == first.procedure_id:
            return BackendResult({"x": s["a"] * 2})
        if p.procedure_id == unrelated.procedure_id:
            return BackendResult({"y": 19})
        return BackendResult({"answer": s["x"] + context["offset"]})

    result = execute_procedure(registry, chain.procedure_id, {"a": 4, "unused": "secret"},
        backends={Backend.TOOL: backend}, context={"offset": 3, "unused": "private"},
        event_graph=graph, parent_events=(source.seq,))
    assert result.completed and result.resulting_state["answer"] == 11
    a, b, c = (graph.get(item.event_id) for item in result.steps)
    assert c.parents == (source.seq, a.seq)
    assert b.seq not in {event.seq for event in graph.ancestors(c.seq)}
    assert {item.key for item in c.reads} == {"procedure:state:x", "procedure:context:offset"}
    assert c.produced == ("procedure:state:answer",)
    assert c.outcome == "executed" and c.detail["correctness_measured"] is False
    bundle = graph.bundle(c.seq)
    assert "secret" not in repr(bundle) and "private" not in repr(bundle)
    assert "procedure:state:unused" not in {d["key"] for d in bundle["minimal_support"]}


def test_overwritten_input_depends_on_latest_writer():
    registry, graph = ProcedureRegistry(), EventGraph()
    write = _step(registry, "write", (), (Effect("x", "integer"),))
    read = _step(registry, "read", (Precondition("x", "integer"),))
    chain = compose(registry, (write, write, read))
    result = execute_procedure(registry, chain.procedure_id, {}, event_graph=graph, backends={
        Backend.TOOL: lambda p, s, c: BackendResult(
            {"x": 3} if p.procedure_id == write.procedure_id else {"answer": s["x"]},
        ),
    })
    assert graph.get(result.steps[-1].event_id).parents == (result.steps[1].event_id,)


def test_missing_binding_and_false_value_have_distinct_observation_status():
    registry, graph = ProcedureRegistry(), EventGraph()
    procedure = _step(registry, "probe")
    def backend(p, state, context):
        assert state.get("missing") is None
        assert state["flag"] is False
        return BackendResult({"answer": 0})
    result = execute_procedure(registry, procedure.procedure_id, {"flag": False},
        backends={Backend.TOOL: backend}, event_graph=graph)
    deps = {d.key: d for d in graph.get(result.steps[0].event_id).reads}
    assert deps["procedure:state:missing"].status is Epistemic.OBSERVED_ABSENT
    assert deps["procedure:state:flag"].status is Epistemic.OBSERVED


def test_failed_backend_records_reads_but_no_successful_writes():
    registry, graph = ProcedureRegistry(), EventGraph()
    procedure = _step(registry, "fail")
    def fail(p, state, context):
        state["nested"].append(9)
        raise RuntimeError("private failure detail")
    initial = {"nested": [1]}
    result = execute_procedure(registry, procedure.procedure_id, initial,
        backends={Backend.TOOL: fail}, event_graph=graph)
    event = graph.get(result.steps[0].event_id)
    assert not result.completed and result.resulting_state == initial == {"nested": [1]}
    assert event.outcome == "failed" and not event.produced
    assert event.detail["error_type"] == "RuntimeError"
    assert "private failure detail" not in repr(event.to_dict())


def test_copying_mapping_is_conservative_without_structure_key_collision():
    registry, graph = ProcedureRegistry(), EventGraph()
    procedure = _step(registry, "copy")
    def backend(p, state, context):
        data = dict(state)
        return BackendResult({"answer": data["$keys"]})
    result = execute_procedure(registry, procedure.procedure_id, {"$keys": 3, "extra": 4},
        backends={Backend.TOOL: backend}, event_graph=graph)
    keys = {d.key for d in graph.get(result.steps[0].event_id).reads}
    assert {"procedure:state:$keys", "procedure:state:extra", "procedure:keys:procedure:state:"} <= keys


def test_trace_writer_state_does_not_cross_independent_executions():
    registry, graph = ProcedureRegistry(), EventGraph()
    procedure = _step(registry, "increment", (Precondition("answer", "integer"),))
    def backend(p, s, c):
        return BackendResult({"answer": s["answer"] + 1})
    first = execute_procedure(registry, procedure.procedure_id, {"answer": 1},
        backends={Backend.TOOL: backend}, event_graph=graph)
    second = execute_procedure(registry, procedure.procedure_id, {"answer": 8},
        backends={Backend.TOOL: backend}, event_graph=graph)
    assert first.steps[0].event_id != second.steps[0].event_id
    assert not graph.get(second.steps[0].event_id).parents


def test_failed_digest_does_not_stop_a_backend_read(monkeypatch):
    import core.cognition.procedure_trace as tracing

    registry, graph = ProcedureRegistry(), EventGraph()
    procedure = _step(registry, "read")
    def broken_digest(*args, **kwargs):
        raise ValueError("private serialization error")
    monkeypatch.setattr(tracing, "reads", broken_digest)
    result = execute_procedure(registry, procedure.procedure_id, {"x": 4},
        backends={Backend.TOOL: lambda p, s, c: BackendResult({"answer": s["x"] * 2})},
        event_graph=graph)
    assert result.completed and result.resulting_state["answer"] == 8
    assert not result.trace_complete
    assert result.steps[0].trace_errors == ("read_digest:ValueError",)
    event = graph.get(result.steps[0].event_id)
    assert event.reads[0].status is Epistemic.INACCESSIBLE
    assert "private serialization" not in repr(event.to_dict())


def test_record_failure_preserves_answer_and_removes_stale_writer(monkeypatch):
    registry, graph = ProcedureRegistry(), EventGraph()
    write = _step(registry, "write", (), (Effect("x", "integer"),))
    read = _step(registry, "read", (Precondition("x", "integer"),))
    chain = compose(registry, (write, write, read))
    record, count = graph.record, 0
    def unreliable_record(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("private storage detail")
        return record(*args, **kwargs)
    monkeypatch.setattr(graph, "record", unreliable_record)
    result = execute_procedure(registry, chain.procedure_id, {}, event_graph=graph, backends={
        Backend.TOOL: lambda p, s, c: BackendResult(
            {"x": 3} if p.procedure_id == write.procedure_id else {"answer": s["x"]},
        ),
    })
    assert result.completed and result.resulting_state["answer"] == 3
    assert not result.trace_complete
    assert result.steps[1].event_id == 0
    assert result.steps[1].trace_errors == ("event_record:OSError",)
    last = graph.get(result.steps[2].event_id)
    assert not last.parents
    assert last.detail["unrecorded_input_writers"] == ("x",)
    assert result.steps[2].trace_errors == ("upstream_event_unavailable",)


def test_backend_failure_stays_backend_failure_when_trace_also_fails(monkeypatch):
    registry, graph = ProcedureRegistry(), EventGraph()
    procedure = _step(registry, "fail")
    def fail_backend(*args):
        raise ValueError("backend failed")
    def fail_trace(*args, **kwargs):
        raise OSError("trace failed")
    monkeypatch.setattr(graph, "record", fail_trace)
    result = execute_procedure(registry, procedure.procedure_id, {}, event_graph=graph,
        backends={Backend.TOOL: fail_backend})
    assert not result.completed and not result.trace_complete
    assert result.steps[0].error == "ValueError: backend failed"
    assert result.steps[0].trace_errors == ("event_record:OSError",)


def test_missing_external_parent_is_not_complete_evidence():
    registry, graph = ProcedureRegistry(), EventGraph()
    procedure = _step(registry, "answer")
    result = execute_procedure(registry, procedure.procedure_id, {}, event_graph=graph,
        parent_events=(999,), backends={Backend.TOOL: lambda p, s, c: BackendResult({"answer": 3})})
    assert result.completed and not result.trace_complete
    assert result.steps[0].trace_errors == ("parent_event_not_retained",)


def test_broken_degradation_sink_does_not_become_an_answer_gate(monkeypatch, caplog):
    import core.cognition.procedure_trace as tracing

    registry, graph = ProcedureRegistry(), EventGraph()
    procedure = _step(registry, "answer")
    def fail(*args, **kwargs):
        raise OSError("private sink error")
    monkeypatch.setattr(graph, "record", fail)
    monkeypatch.setattr(tracing, "record_degradation", fail)
    result = execute_procedure(registry, procedure.procedure_id, {}, event_graph=graph,
        backends={Backend.TOOL: lambda p, s, c: BackendResult({"answer": 3})})
    assert result.completed and result.resulting_state["answer"] == 3
    assert not result.trace_complete
    assert "Procedure trace reporting failed: OSError" in caplog.text
    assert "private sink" not in caplog.text


def test_false_binding_invariant():
    from core.cognition.procedure_trace import _false_binding_is_observed

    assert _false_binding_is_observed() == ()
