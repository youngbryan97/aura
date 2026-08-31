# Aura Architecture Dependency Map

*Reviewed against the tree: 2026-08-31. See [documentation status map](DOC_STATUS.md) for how to read this file.*

Schema: `aura.architecture.dependency_map.v2`
Root: `<AURA_ROOT>`
Generated: `0.0`

## Summary

- Subsystems: 159
- Python files: 3081
- Python lines: 1369040
- Dependency edges: 1283
- ServiceContainer `.get()` calls: 1501
- ServiceContainer registrations: 331
- Boot contract: PASS

## Subsystem Dependency Graph

```mermaid
graph TD
    runtime["runtime<br/>229 files, 95988 lines"]
    utils["utils<br/>46 files, 8187 lines"]
    brain["brain<br/>393 files, 295742 lines"]
    memory["memory<br/>107 files, 33275 lines"]
    consciousness["consciousness<br/>157 files, 76808 lines"]
    resilience["resilience<br/>62 files, 18113 lines"]
    governance["governance<br/>14 files, 6798 lines"]
    security["security<br/>51 files, 14767 lines"]
    conversation["conversation<br/>69 files, 35785 lines"]
    health["health<br/>7 files, 2052 lines"]
    agency["agency<br/>72 files, 31871 lines"]
    observability["observability<br/>14 files, 4547 lines"]
    cognition["cognition<br/>59 files, 23632 lines"]
    perception["perception<br/>48 files, 20731 lines"]
    affect["affect<br/>12 files, 4738 lines"]
    executive["executive<br/>15 files, 7232 lines"]
    skills["skills<br/>103 files, 45631 lines"]
    senses["senses<br/>32 files, 10093 lines"]
    epistemics["epistemics<br/>18 files, 6984 lines"]
    identity["identity<br/>19 files, 3755 lines"]
    self_modification["self_modification<br/>36 files, 14062 lines"]
    adaptation["adaptation<br/>28 files, 16938 lines"]
    knowledge["knowledge<br/>16 files, 4558 lines"]
    state["state<br/>7 files, 4908 lines"]
    world_model["world_model<br/>11 files, 3936 lines"]
    verify["verify<br/>18 files, 6492 lines"]
    being["being<br/>25 files, 7773 lines"]
    learning["learning<br/>191 files, 130863 lines"]
    organism["organism<br/>9 files, 5891 lines"]
    autonomy["autonomy<br/>31 files, 14987 lines"]
    continuity["continuity<br/>1 files, 33 lines"]
    orchestrator["orchestrator<br/>42 files, 23283 lines"]
    social["social<br/>20 files, 8247 lines"]
    voice["voice<br/>36 files, 13935 lines"]
    capabilities["capabilities<br/>20 files, 17107 lines"]
    language["language<br/>21 files, 3811 lines"]
    reasoning["reasoning<br/>20 files, 9337 lines"]
    sandbox["sandbox<br/>7 files, 2195 lines"]
    values["values<br/>15 files, 2062 lines"]
    bus["bus<br/>7 files, 4207 lines"]
    fsw["fsw<br/>7 files, 2973 lines"]
    goals["goals<br/>12 files, 4536 lines"]
    self["self<br/>21 files, 8749 lines"]
    introspection["introspection<br/>10 files, 4129 lines"]
    morality["morality<br/>16 files, 1327 lines"]
    phases["phases<br/>29 files, 24692 lines"]
    tasks["tasks<br/>4 files, 566 lines"]
    embodiment["embodiment<br/>28 files, 13534 lines"]
    intent["intent<br/>8 files, 2358 lines"]
    ontogeny["ontogeny<br/>18 files, 7483 lines"]
    actuators["actuators<br/>11 files, 5173 lines"]
    autonomic["autonomic<br/>6 files, 3726 lines"]
    discovery["discovery<br/>7 files, 2151 lines"]
    kernel["kernel<br/>11 files, 6867 lines"]
    ops["ops<br/>17 files, 5536 lines"]
    planning["planning<br/>9 files, 4406 lines"]
    world["world<br/>24 files, 1489 lines"]
    agi["agi<br/>6 files, 2872 lines"]
    cognitive["cognitive<br/>12 files, 9756 lines"]
    coordinators["coordinators<br/>10 files, 4973 lines"]
    ethics["ethics<br/>2 files, 602 lines"]
    llm["llm<br/>3 files, 259 lines"]
    managers["managers<br/>6 files, 966 lines"]
    pipeline["pipeline<br/>3 files, 808 lines"]
    resource["resource<br/>4 files, 624 lines"]
    sleep["sleep<br/>10 files, 1260 lines"]
    somatic["somatic<br/>6 files, 2944 lines"]
    supervisor["supervisor<br/>3 files, 993 lines"]
    unity["unity<br/>11 files, 2825 lines"]
    advanced_cognition["advanced_cognition<br/>13 files, 5381 lines"]
    collective["collective<br/>6 files, 2217 lines"]
    data["data<br/>3 files, 652 lines"]
    dialogue["dialogue<br/>4 files, 860 lines"]
    environment["environment<br/>76 files, 9151 lines"]
    evaluation["evaluation<br/>20 files, 4860 lines"]
    media["media<br/>4 files, 1079 lines"]
    meta["meta<br/>7 files, 1278 lines"]
    motivation["motivation<br/>7 files, 1209 lines"]
    promotion["promotion<br/>6 files, 936 lines"]
    reality_reach["reality_reach<br/>33 files, 31020 lines"]
    self_improvement["self_improvement<br/>21 files, 9206 lines"]
    soma["soma<br/>4 files, 1647 lines"]
    adapters["adapters<br/>8 files, 2393 lines"]
    conation["conation<br/>17 files, 5504 lines"]
    conversational["conversational<br/>4 files, 2435 lines"]
    db["db<br/>4 files, 818 lines"]
    engineering["engineering<br/>36 files, 15669 lines"]
    maintenance["maintenance<br/>2 files, 295 lines"]
    morphogenesis["morphogenesis<br/>11 files, 3112 lines"]
    phenomenal_substrate["phenomenal_substrate<br/>11 files, 1148 lines"]
    pneuma["pneuma<br/>7 files, 1279 lines"]
    search["search<br/>3 files, 2056 lines"]
    verification["verification<br/>4 files, 350 lines"]
    workspace["workspace<br/>9 files, 1243 lines"]
    architect["architect<br/>25 files, 7216 lines"]
    architecture_quality["architecture_quality<br/>6 files, 1979 lines"]
    coherence["coherence<br/>2 files, 407 lines"]
    construction["construction<br/>7 files, 2043 lines"]
    evals["evals<br/>2 files, 686 lines"]
    evolution["evolution<br/>8 files, 2156 lines"]
    ghost["ghost<br/>6 files, 2076 lines"]
    grounding["grounding<br/>8 files, 1263 lines"]
    lattice["lattice<br/>5 files, 704 lines"]
    metacognition["metacognition<br/>3 files, 1021 lines"]
    persistence["persistence<br/>2 files, 619 lines"]
    plasticity["plasticity<br/>5 files, 428 lines"]
    predictive["predictive<br/>2 files, 186 lines"]
    sensors["sensors<br/>1 files, 195 lines"]
    services["services<br/>2 files, 31 lines"]
    sim["sim<br/>2 files, 452 lines"]
    simulation["simulation<br/>3 files, 402 lines"]
    skill_management["skill_management<br/>3 files, 1800 lines"]
    sovereign["sovereign<br/>4 files, 522 lines"]
    startup["startup<br/>4 files, 546 lines"]
    unknowns["unknowns<br/>4 files, 325 lines"]
    actuation["actuation<br/>9 files, 1132 lines"]
    audit["audit<br/>7 files, 1016 lines"]
    body["body<br/>22 files, 1815 lines"]
    communication["communication<br/>5 files, 2233 lines"]
    consent["consent<br/>2 files, 167 lines"]
    context["context<br/>7 files, 2542 lines"]
    creativity["creativity<br/>2 files, 801 lines"]
    curriculum["curriculum<br/>7 files, 658 lines"]
    cybernetics["cybernetics<br/>6 files, 1293 lines"]
    diagnosis["diagnosis<br/>5 files, 1149 lines"]
    environments["environments<br/>7 files, 749 lines"]
    factory["factory<br/>8 files, 760 lines"]
    fictional["fictional<br/>9 files, 2519 lines"]
    guardians["guardians<br/>5 files, 791 lines"]
    initializers["initializers<br/>2 files, 61 lines"]
    middleware["middleware<br/>1 files, 214 lines"]
    networking["networking<br/>1 files, 332 lines"]
    quantum["quantum<br/>5 files, 757 lines"]
    research_core["research_core<br/>5 files, 580 lines"]
    safety["safety<br/>3 files, 631 lines"]
    session["session<br/>3 files, 642 lines"]
    sovereignty["sovereignty<br/>4 files, 2098 lines"]
    systems["systems<br/>3 files, 256 lines"]
    transparency["transparency<br/>2 files, 403 lines"]
    welfare["welfare<br/>7 files, 228 lines"]
    worlds["worlds<br/>8 files, 3045 lines"]
    audits["audits<br/>2 files, 314 lines"]
    control["control<br/>2 files, 585 lines"]
    core_root["core_root<br/>46 files, 32978 lines"]
    council["council<br/>6 files, 759 lines"]
    forge["forge<br/>8 files, 326 lines"]
    lab["lab<br/>7 files, 482 lines"]
    latent["latent<br/>1 files, 56 lines"]
    mission["mission<br/>4 files, 472 lines"]
    multimodal["multimodal<br/>2 files, 185 lines"]
    neuroweb["neuroweb<br/>4 files, 313 lines"]
    ontology["ontology<br/>2 files, 169 lines"]
    play["play<br/>1 files, 259 lines"]
    providers["providers<br/>6 files, 1469 lines"]
    reproducibility["reproducibility<br/>2 files, 497 lines"]
    science["science<br/>1 files, 139 lines"]
    swarm["swarm<br/>4 files, 365 lines"]
    tools["tools<br/>11 files, 1917 lines"]
    twins["twins<br/>1 files, 97 lines"]
    runtime --> actuators
    runtime --> adaptation
    runtime --> affect
    runtime --> agency
    runtime --> architect
    runtime --> autonomy
    runtime --> being
    runtime --> brain
    runtime --> bus
    runtime --> conation
    runtime --> consciousness
    runtime --> conversation
    runtime --> evaluation
    runtime --> fsw
    runtime --> goals
    runtime --> governance
    runtime --> health
    runtime --> identity
    runtime --> intent
    runtime --> knowledge
    runtime --> language
    runtime --> learning
    runtime --> memory
    runtime --> observability
    runtime --> ontogeny
    runtime --> organism
    runtime --> perception
    runtime --> persistence
    runtime --> phases
    runtime --> pipeline
    runtime --> reasoning
    runtime --> research_core
    runtime --> resilience
    runtime --> resource
    runtime --> security
    runtime --> self
    runtime --> self_improvement
    runtime --> self_modification
    runtime --> senses
    runtime --> skills
    runtime --> social
    runtime --> state
    runtime --> supervisor
    runtime --> tasks
    runtime --> utils
    runtime --> verify
    runtime --> workspace
    utils --> conversation
    utils --> epistemics
    utils --> health
    utils --> identity
    utils --> managers
    utils --> memory
    utils --> resilience
    utils --> runtime
    utils --> tasks
    brain --> adaptation
    brain --> adapters
    brain --> affect
    brain --> agency
    brain --> agi
    brain --> being
    brain --> capabilities
    brain --> cognition
    brain --> cognitive
    brain --> consciousness
    brain --> continuity
    brain --> conversation
    brain --> dialogue
    brain --> discovery
    brain --> engineering
    brain --> epistemics
    brain --> executive
    brain --> fsw
    brain --> goals
    brain --> governance
    brain --> health
    brain --> identity
    brain --> intent
    brain --> introspection
    brain --> kernel
    brain --> knowledge
    brain --> language
    brain --> learning
    brain --> llm
    brain --> maintenance
    brain --> memory
    brain --> morphogenesis
    brain --> observability
    brain --> ontogeny
    brain --> organism
    brain --> perception
    brain --> phases
    brain --> pipeline
    brain --> pneuma
    brain --> reasoning
    brain --> resilience
    brain --> runtime
    brain --> sandbox
    brain --> search
    brain --> security
    brain --> self
    brain --> self_modification
    brain --> senses
    brain --> skills
    brain --> state
    brain --> utils
    brain --> verify
    brain --> voice
    memory --> actuators
    memory --> being
    memory --> brain
    memory --> cognition
    memory --> consciousness
    memory --> conversation
    memory --> db
    memory --> governance
    memory --> health
    memory --> knowledge
    memory --> observability
    memory --> ontogeny
    memory --> phases
    memory --> resilience
    memory --> runtime
    memory --> security
    memory --> social
    memory --> utils
    memory --> values
    consciousness --> adaptation
    consciousness --> affect
    consciousness --> agency
    consciousness --> being
    consciousness --> brain
    consciousness --> continuity
    consciousness --> coordinators
    consciousness --> evals
    consciousness --> evaluation
    consciousness --> executive
    consciousness --> ghost
    consciousness --> goals
    consciousness --> governance
    consciousness --> health
    consciousness --> kernel
    consciousness --> memory
    consciousness --> meta
    consciousness --> observability
    consciousness --> orchestrator
    consciousness --> pneuma
    consciousness --> predictive
    consciousness --> reasoning
    consciousness --> resilience
    consciousness --> runtime
    consciousness --> senses
    consciousness --> sensors
    consciousness --> social
    consciousness --> state
    consciousness --> unity
    consciousness --> utils
    consciousness --> verify
    consciousness --> world_model
    resilience --> adaptation
    resilience --> agency
    resilience --> brain
    resilience --> consciousness
    resilience --> conversation
    resilience --> coordinators
    resilience --> health
    resilience --> memory
    resilience --> observability
    resilience --> runtime
    resilience --> security
    resilience --> tasks
    resilience --> utils
    governance --> actuators
    governance --> being
    governance --> brain
    governance --> consciousness
    governance --> executive
    governance --> identity
    governance --> memory
    governance --> observability
    governance --> resilience
    governance --> runtime
    governance --> utils
    security --> affect
    security --> agency
    security --> brain
    security --> consciousness
    security --> executive
    security --> fsw
    security --> governance
    security --> identity
    security --> memory
    security --> perception
    security --> runtime
    security --> senses
    security --> utils
    conversation --> autonomy
    conversation --> brain
    conversation --> cognition
    conversation --> consciousness
    conversation --> construction
    conversation --> dialogue
    conversation --> epistemics
    conversation --> health
    conversation --> identity
    conversation --> intent
    conversation --> introspection
    conversation --> knowledge
    conversation --> language
    conversation --> memory
    conversation --> organism
    conversation --> perception
    conversation --> reasoning
    conversation --> resilience
    conversation --> runtime
    conversation --> self
    conversation --> senses
    conversation --> skills
    conversation --> social
    conversation --> state
    conversation --> utils
    conversation --> verify
    health --> memory
    health --> runtime
    health --> state
    health --> utils
    agency --> adaptation
    agency --> affect
    agency --> agi
    agency --> autonomy
    agency --> brain
    agency --> capabilities
    agency --> cognition
    agency --> consciousness
    agency --> continuity
    agency --> conversation
    agency --> executive
    agency --> goals
    agency --> governance
    agency --> health
    agency --> identity
    agency --> knowledge
    agency --> learning
    agency --> morality
    agency --> observability
    agency --> ontogeny
    agency --> orchestrator
    agency --> organism
    agency --> perception
    agency --> phases
    agency --> resilience
    agency --> runtime
    agency --> skills
    agency --> social
    agency --> state
    agency --> tasks
    agency --> utils
    agency --> values
    agency --> world_model
    observability --> health
    observability --> memory
    observability --> pipeline
    observability --> runtime
    observability --> utils
    cognition --> actuators
    cognition --> affect
    cognition --> agency
    cognition --> brain
    cognition --> consciousness
    cognition --> conversation
    cognition --> epistemics
    cognition --> fsw
    cognition --> governance
    cognition --> introspection
    cognition --> memory
    cognition --> runtime
    cognition --> sim
    cognition --> skills
    cognition --> social
    cognition --> utils
    cognition --> voice
    cognition --> world_model
    perception --> brain
    perception --> capabilities
    perception --> governance
    perception --> media
    perception --> phenomenal_substrate
    perception --> resilience
    perception --> runtime
    perception --> security
    perception --> senses
    perception --> utils
    perception --> voice
    affect --> adaptation
    affect --> autonomic
    affect --> brain
    affect --> consciousness
    affect --> health
    affect --> memory
    affect --> phenomenal_substrate
    affect --> runtime
    affect --> senses
    affect --> utils
    affect --> verify
    executive --> agency
    executive --> autonomy
    executive --> consciousness
    executive --> continuity
    executive --> goals
    executive --> governance
    executive --> health
    executive --> memory
    executive --> morality
    executive --> ontogeny
    executive --> organism
    executive --> runtime
    executive --> skills
    executive --> state
    executive --> utils
    skills --> actuators
    skills --> advanced_cognition
    skills --> affect
    skills --> agency
    skills --> being
    skills --> brain
    skills --> capabilities
    skills --> cognition
    skills --> communication
    skills --> consciousness
    skills --> consent
    skills --> construction
    skills --> conversation
    skills --> diagnosis
    skills --> dialogue
    skills --> embodiment
    skills --> engineering
    skills --> epistemics
    skills --> executive
    skills --> governance
    skills --> introspection
    skills --> knowledge
    skills --> language
    skills --> learning
    skills --> memory
    skills --> perception
    skills --> quantum
    skills --> reality_reach
    skills --> runtime
    skills --> sandbox
    skills --> search
    skills --> security
    skills --> self
    skills --> self_improvement
    skills --> self_modification
    skills --> senses
    skills --> sovereign
    skills --> utils
    skills --> voice
    skills --> world_model
    skills --> worlds
    senses --> affect
    senses --> brain
    senses --> cognition
    senses --> consciousness
    senses --> conversation
    senses --> health
    senses --> media
    senses --> memory
    senses --> networking
    senses --> orchestrator
    senses --> perception
    senses --> resilience
    senses --> runtime
    senses --> security
    senses --> supervisor
    senses --> utils
    senses --> voice
    epistemics --> being
    epistemics --> brain
    epistemics --> conversation
    epistemics --> knowledge
    epistemics --> observability
    epistemics --> reasoning
    epistemics --> runtime
    epistemics --> skills
    epistemics --> utils
    identity --> agency
    identity --> brain
    identity --> governance
    identity --> organism
    identity --> runtime
    identity --> utils
    self_modification --> architecture_quality
    self_modification --> bus
    self_modification --> ethics
    self_modification --> governance
    self_modification --> memory
    self_modification --> ops
    self_modification --> resilience
    self_modification --> runtime
    self_modification --> security
    self_modification --> skills
    self_modification --> utils
    adaptation --> actuators
    adaptation --> affect
    adaptation --> being
    adaptation --> brain
    adaptation --> cognitive
    adaptation --> executive
    adaptation --> governance
    adaptation --> health
    adaptation --> identity
    adaptation --> learning
    adaptation --> memory
    adaptation --> resilience
    adaptation --> runtime
    adaptation --> security
    adaptation --> sensors
    adaptation --> utils
    adaptation --> world
    knowledge --> brain
    knowledge --> reasoning
    knowledge --> runtime
    knowledge --> utils
    state --> being
    state --> brain
    state --> bus
    state --> goals
    state --> governance
    state --> identity
    state --> memory
    state --> motivation
    state --> runtime
    state --> unity
    state --> utils
    state --> values
    world_model --> advanced_cognition
    world_model --> brain
    world_model --> cognition
    world_model --> health
    world_model --> resilience
    world_model --> runtime
    world_model --> values
    verify --> bus
    verify --> fsw
    verify --> health
    verify --> knowledge
    verify --> observability
    verify --> organism
    verify --> runtime
    verify --> security
    being --> agency
    being --> brain
    being --> consciousness
    being --> epistemics
    being --> governance
    being --> observability
    being --> runtime
    being --> verify
    learning --> agi
    learning --> architecture_quality
    learning --> brain
    learning --> consciousness
    learning --> executive
    learning --> introspection
    learning --> language
    learning --> memory
    learning --> orchestrator
    learning --> promotion
    learning --> reasoning
    learning --> runtime
    learning --> sandbox
    learning --> security
    learning --> self_modification
    learning --> skills
    learning --> tasks
    learning --> utils
    learning --> world_model
    organism --> adaptation
    organism --> affect
    organism --> agency
    organism --> being
    organism --> body
    organism --> brain
    organism --> cognition
    organism --> conation
    organism --> conversation
    organism --> epistemics
    organism --> executive
    organism --> fsw
    organism --> governance
    organism --> health
    organism --> identity
    organism --> introspection
    organism --> learning
    organism --> memory
    organism --> reality_reach
    organism --> resilience
    organism --> runtime
    organism --> sandbox
    organism --> security
    organism --> sleep
    organism --> utils
    organism --> values
    organism --> verify
    organism --> welfare
    organism --> workspace
    organism --> world
    autonomy --> affect
    autonomy --> agency
    autonomy --> brain
    autonomy --> consciousness
    autonomy --> continuity
    autonomy --> conversation
    autonomy --> conversational
    autonomy --> discovery
    autonomy --> executive
    autonomy --> governance
    autonomy --> health
    autonomy --> knowledge
    autonomy --> memory
    autonomy --> observability
    autonomy --> planning
    autonomy --> resource
    autonomy --> runtime
    autonomy --> security
    autonomy --> skills
    autonomy --> sleep
    autonomy --> state
    autonomy --> utils
    autonomy --> voice
    autonomy --> world_model
    continuity --> organism
    orchestrator --> adaptation
    orchestrator --> affect
    orchestrator --> agency
    orchestrator --> agi
    orchestrator --> audit
    orchestrator --> autonomic
    orchestrator --> autonomy
    orchestrator --> brain
    orchestrator --> bus
    orchestrator --> capabilities
    orchestrator --> cognition
    orchestrator --> cognitive
    orchestrator --> collective
    orchestrator --> consciousness
    orchestrator --> context
    orchestrator --> continuity
    orchestrator --> conversation
    orchestrator --> coordinators
    orchestrator --> data
    orchestrator --> db
    orchestrator --> embodiment
    orchestrator --> environment
    orchestrator --> epistemics
    orchestrator --> ethics
    orchestrator --> evals
    orchestrator --> evolution
    orchestrator --> executive
    orchestrator --> goals
    orchestrator --> governance
    orchestrator --> guardians
    orchestrator --> health
    orchestrator --> identity
    orchestrator --> initializers
    orchestrator --> kernel
    orchestrator --> knowledge
    orchestrator --> learning
    orchestrator --> maintenance
    orchestrator --> managers
    orchestrator --> memory
    orchestrator --> meta
    orchestrator --> morality
    orchestrator --> morphogenesis
    orchestrator --> motivation
    orchestrator --> observability
    orchestrator --> ops
    orchestrator --> perception
    orchestrator --> phases
    orchestrator --> planning
    orchestrator --> pneuma
    orchestrator --> reality_reach
    orchestrator --> resilience
    orchestrator --> runtime
    orchestrator --> safety
    orchestrator --> security
    orchestrator --> self
    orchestrator --> self_improvement
    orchestrator --> self_modification
    orchestrator --> senses
    orchestrator --> session
    orchestrator --> sim
    orchestrator --> simulation
    orchestrator --> skill_management
    orchestrator --> sleep
    orchestrator --> social
    orchestrator --> soma
    orchestrator --> somatic
    orchestrator --> sovereignty
    orchestrator --> startup
    orchestrator --> state
    orchestrator --> supervisor
    orchestrator --> tasks
    orchestrator --> utils
    orchestrator --> values
    orchestrator --> verification
    orchestrator --> verify
    orchestrator --> voice
    orchestrator --> world_model
    social --> agency
    social --> autonomy
    social --> brain
    social --> consciousness
    social --> epistemics
    social --> ethics
    social --> governance
    social --> memory
    social --> runtime
    social --> security
    social --> senses
    social --> utils
    voice --> brain
    voice --> conversation
    voice --> conversational
    voice --> executive
    voice --> managers
    voice --> resilience
    voice --> runtime
    voice --> senses
    voice --> utils
    capabilities --> adapters
    capabilities --> agency
    capabilities --> brain
    capabilities --> governance
    capabilities --> knowledge
    capabilities --> llm
    capabilities --> memory
    capabilities --> perception
    capabilities --> runtime
    capabilities --> security
    capabilities --> self_modification
    capabilities --> skills
    capabilities --> utils
    language --> brain
    language --> conversation
    language --> memory
    language --> runtime
    reasoning --> cognition
    reasoning --> governance
    reasoning --> observability
    reasoning --> planning
    reasoning --> runtime
    reasoning --> utils
    sandbox --> runtime
    values --> agency
    values --> governance
    values --> runtime
    values --> social
    values --> utils
    bus --> capabilities
    bus --> resilience
    bus --> runtime
    bus --> utils
    fsw --> health
    fsw --> observability
    fsw --> pipeline
    fsw --> runtime
    fsw --> utils
    fsw --> verify
    goals --> agency
    goals --> autonomy
    goals --> brain
    goals --> runtime
    goals --> state
    goals --> utils
    goals --> values
    self --> affect
    self --> being
    self --> consciousness
    self --> conversation
    self --> dialogue
    self --> epistemics
    self --> fsw
    self --> intent
    self --> language
    self --> memory
    self --> ontogeny
    self --> organism
    self --> runtime
    self --> security
    self --> senses
    self --> skills
    self --> soma
    self --> state
    self --> utils
    introspection --> cognition
    introspection --> epistemics
    introspection --> health
    introspection --> language
    introspection --> resilience
    introspection --> runtime
    introspection --> security
    introspection --> utils
    introspection --> voice
    morality --> autonomy
    morality --> brain
    morality --> consciousness
    morality --> perception
    morality --> runtime
    morality --> utils
    phases --> adaptation
    phases --> agency
    phases --> autonomy
    phases --> brain
    phases --> cognition
    phases --> coherence
    phases --> conation
    phases --> consciousness
    phases --> conversation
    phases --> conversational
    phases --> embodiment
    phases --> evaluation
    phases --> health
    phases --> identity
    phases --> intent
    phases --> kernel
    phases --> knowledge
    phases --> language
    phases --> learning
    phases --> llm
    phases --> memory
    phases --> morality
    phases --> reasoning
    phases --> runtime
    phases --> self
    phases --> self_modification
    phases --> skills
    phases --> social
    phases --> somatic
    phases --> state
    phases --> unity
    phases --> utils
    phases --> voice
    tasks --> runtime
    embodiment --> actuation
    embodiment --> agency
    embodiment --> bus
    embodiment --> consciousness
    embodiment --> ethics
    embodiment --> governance
    embodiment --> organism
    embodiment --> reality_reach
    embodiment --> runtime
    embodiment --> utils
    embodiment --> voice
    intent --> brain
    intent --> conversation
    intent --> epistemics
    intent --> language
    intent --> runtime
    intent --> skills
    intent --> utils
    ontogeny --> fsw
    ontogeny --> runtime
    ontogeny --> verify
    ontogeny --> world_model
    actuators --> affect
    actuators --> brain
    actuators --> executive
    actuators --> governance
    actuators --> runtime
    actuators --> sandbox
    actuators --> search
    actuators --> skills
    actuators --> utils
    actuators --> world
    autonomic --> embodiment
    autonomic --> orchestrator
    autonomic --> runtime
    autonomic --> utils
    discovery --> cognition
    discovery --> memory
    discovery --> observability
    discovery --> runtime
    discovery --> self_modification
    discovery --> unknowns
    kernel --> agency
    kernel --> brain
    kernel --> cognition
    kernel --> consciousness
    kernel --> continuity
    kernel --> cybernetics
    kernel --> executive
    kernel --> goals
    kernel --> health
    kernel --> introspection
    kernel --> learning
    kernel --> ops
    kernel --> perception
    kernel --> phases
    kernel --> pipeline
    kernel --> resilience
    kernel --> runtime
    kernel --> security
    kernel --> self_modification
    kernel --> senses
    kernel --> somatic
    kernel --> state
    kernel --> utils
    ops --> brain
    ops --> coordinators
    ops --> kernel
    ops --> managers
    ops --> observability
    ops --> orchestrator
    ops --> resilience
    ops --> resource
    ops --> runtime
    ops --> senses
    ops --> state
    ops --> supervisor
    ops --> utils
    planning --> brain
    planning --> capabilities
    planning --> collective
    planning --> data
    planning --> runtime
    planning --> utils
    world --> governance
    world --> runtime
    agi --> adaptation
    agi --> brain
    agi --> conversation
    agi --> embodiment
    agi --> epistemics
    agi --> grounding
    agi --> runtime
    agi --> security
    agi --> utils
    agi --> world_model
    cognitive --> brain
    cognitive --> conversation
    cognitive --> executive
    cognitive --> governance
    cognitive --> health
    cognitive --> perception
    cognitive --> phases
    cognitive --> runtime
    cognitive --> utils
    coordinators --> autonomic
    coordinators --> autonomy
    coordinators --> brain
    coordinators --> continuity
    coordinators --> conversation
    coordinators --> environment
    coordinators --> epistemics
    coordinators --> evolution
    coordinators --> executive
    coordinators --> health
    coordinators --> maintenance
    coordinators --> memory
    coordinators --> meta
    coordinators --> morphogenesis
    coordinators --> observability
    coordinators --> ops
    coordinators --> orchestrator
    coordinators --> perception
    coordinators --> persistence
    coordinators --> resilience
    coordinators --> resource
    coordinators --> runtime
    coordinators --> security
    coordinators --> sleep
    coordinators --> somatic
    coordinators --> tasks
    coordinators --> utils
    coordinators --> verify
    coordinators --> world_model
    ethics --> brain
    ethics --> morality
    ethics --> runtime
    ethics --> utils
    llm --> brain
    managers --> autonomic
    managers --> brain
    managers --> bus
    managers --> cognition
    managers --> collective
    managers --> data
    managers --> health
    managers --> memory
    managers --> morality
    managers --> observability
    managers --> ops
    managers --> orchestrator
    managers --> planning
    managers --> resilience
    managers --> runtime
    managers --> security
    managers --> self_modification
    managers --> senses
    managers --> utils
    pipeline --> observability
    pipeline --> runtime
    pipeline --> verify
    resource --> observability
    resource --> resilience
    resource --> runtime
    sleep --> adaptation
    sleep --> affect
    sleep --> brain
    sleep --> conversation
    sleep --> identity
    sleep --> memory
    sleep --> runtime
    sleep --> systems
    sleep --> world_model
    somatic --> media
    somatic --> memory
    somatic --> observability
    somatic --> perception
    somatic --> runtime
    somatic --> utils
    somatic --> world_model
    supervisor --> bus
    supervisor --> runtime
    supervisor --> utils
    unity --> affect
    unity --> cognition
    unity --> consciousness
    unity --> ghost
    unity --> runtime
    unity --> social
    unity --> values
    advanced_cognition --> environment
    advanced_cognition --> reasoning
    advanced_cognition --> runtime
    collective --> adaptation
    collective --> agency
    collective --> brain
    collective --> planning
    collective --> runtime
    collective --> utils
    data --> runtime
    environment --> advanced_cognition
    environment --> brain
    environment --> consciousness
    environment --> environments
    environment --> executive
    environment --> memory
    environment --> perception
    environment --> runtime
    evaluation --> conversation
    evaluation --> learning
    evaluation --> promotion
    evaluation --> runtime
    media --> conversation
    media --> runtime
    meta --> adaptation
    meta --> runtime
    meta --> utils
    motivation --> brain
    motivation --> consciousness
    motivation --> health
    motivation --> runtime
    motivation --> utils
    motivation --> values
    promotion --> runtime
    reality_reach --> advanced_cognition
    reality_reach --> bus
    reality_reach --> embodiment
    reality_reach --> governance
    reality_reach --> observability
    reality_reach --> perception
    reality_reach --> runtime
    reality_reach --> security
    reality_reach --> somatic
    reality_reach --> utils
    self_improvement --> brain
    self_improvement --> discovery
    self_improvement --> llm
    self_improvement --> runtime
    self_improvement --> sandbox
    self_improvement --> self_modification
    self_improvement --> skills
    soma --> continuity
    soma --> resilience
    soma --> runtime
    soma --> utils
    adapters --> agency
    adapters --> brain
    adapters --> runtime
    adapters --> utils
    conation --> affect
    conation --> fsw
    conation --> memory
    conation --> ontogeny
    conation --> runtime
    conation --> verify
    conversational --> memory
    conversational --> runtime
    conversational --> social
    db --> runtime
    engineering --> metacognition
    engineering --> runtime
    maintenance --> resilience
    maintenance --> runtime
    morphogenesis --> adaptation
    morphogenesis --> memory
    morphogenesis --> resilience
    morphogenesis --> runtime
    phenomenal_substrate --> runtime
    pneuma --> affect
    pneuma --> runtime
    pneuma --> utils
    search --> capabilities
    search --> conversation
    search --> knowledge
    search --> memory
    search --> runtime
    search --> utils
    verification --> discovery
    verification --> middleware
    workspace --> runtime
    architect --> adaptation
    architect --> runtime
    architect --> self_modification
    coherence --> agency
    coherence --> consciousness
    coherence --> runtime
    coherence --> self
    coherence --> unity
    construction --> language
    construction --> runtime
    evals --> consciousness
    evals --> epistemics
    evals --> runtime
    evals --> security
    evals --> self_modification
    evolution --> agi
    evolution --> brain
    evolution --> runtime
    evolution --> self_modification
    evolution --> utils
    ghost --> memory
    ghost --> runtime
    ghost --> self
    grounding --> cognition
    grounding --> plasticity
    grounding --> resilience
    grounding --> runtime
    metacognition --> engineering
    metacognition --> memory
    metacognition --> runtime
    persistence --> observability
    persistence --> resilience
    persistence --> runtime
    plasticity --> runtime
    predictive --> brain
    predictive --> runtime
    predictive --> utils
    sensors --> runtime
    sensors --> world
    services --> autonomic
    sim --> brain
    sim --> morality
    sim --> runtime
    sim --> utils
    simulation --> brain
    simulation --> consciousness
    simulation --> identity
    simulation --> runtime
    simulation --> world_model
    skill_management --> resilience
    skill_management --> runtime
    skill_management --> sandbox
    skill_management --> self_modification
    sovereign --> runtime
    startup --> brain
    startup --> consciousness
    startup --> intent
    startup --> memory
    startup --> orchestrator
    startup --> resilience
    startup --> runtime
    startup --> senses
    startup --> utils
    unknowns --> lattice
    unknowns --> promotion
    unknowns --> verification
    actuation --> runtime
    audit --> epistemics
    audit --> runtime
    audit --> security
    body --> capabilities
    body --> perception
    body --> runtime
    body --> security
    communication --> executive
    communication --> governance
    communication --> runtime
    communication --> security
    communication --> utils
    context --> runtime
    creativity --> memory
    creativity --> runtime
    curriculum --> runtime
    cybernetics --> cognitive
    cybernetics --> kernel
    cybernetics --> runtime
    cybernetics --> security
    cybernetics --> utils
    diagnosis --> cognition
    diagnosis --> runtime
    environments --> environment
    environments --> perception
    environments --> runtime
    factory --> runtime
    fictional --> brain
    fictional --> cognition
    fictional --> governance
    fictional --> runtime
    fictional --> security
    fictional --> utils
    guardians --> brain
    guardians --> morality
    guardians --> runtime
    guardians --> tasks
    guardians --> utils
    initializers --> consciousness
    initializers --> runtime
    middleware --> runtime
    networking --> runtime
    research_core --> curriculum
    research_core --> discovery
    research_core --> lattice
    research_core --> promotion
    research_core --> runtime
    research_core --> unknowns
    research_core --> verification
    safety --> runtime
    session --> runtime
    session --> utils
    sovereignty --> ethics
    sovereignty --> governance
    sovereignty --> identity
    sovereignty --> organism
    sovereignty --> runtime
    sovereignty --> utils
    systems --> runtime
    systems --> services
    transparency --> conversation
    transparency --> runtime
    worlds --> embodiment
    worlds --> learning
    worlds --> runtime
    audits --> brain
    audits --> runtime
    control --> runtime
    control --> utils
    core_root --> adaptation
    core_root --> agency
    core_root --> architect
    core_root --> autonomic
    core_root --> autonomy
    core_root --> being
    core_root --> brain
    core_root --> capabilities
    core_root --> cognition
    core_root --> coherence
    core_root --> consciousness
    core_root --> continuity
    core_root --> conversation
    core_root --> data
    core_root --> evaluation
    core_root --> executive
    core_root --> fictional
    core_root --> goals
    core_root --> governance
    core_root --> grounding
    core_root --> health
    core_root --> identity
    core_root --> intent
    core_root --> knowledge
    core_root --> llm
    core_root --> media
    core_root --> memory
    core_root --> meta
    core_root --> metacognition
    core_root --> motivation
    core_root --> observability
    core_root --> orchestrator
    core_root --> organism
    core_root --> perception
    core_root --> phases
    core_root --> planning
    core_root --> predictive
    core_root --> resilience
    core_root --> resource
    core_root --> runtime
    core_root --> sandbox
    core_root --> security
    core_root --> self
    core_root --> self_improvement
    core_root --> self_modification
    core_root --> senses
    core_root --> simulation
    core_root --> skill_management
    core_root --> skills
    core_root --> soma
    core_root --> sovereign
    core_root --> startup
    core_root --> state
    core_root --> supervisor
    core_root --> transparency
    core_root --> utils
    core_root --> voice
    core_root --> workspace
    council --> runtime
    council --> skills
    council --> utils
    forge --> runtime
    lab --> cognition
    lab --> discovery
    lab --> runtime
    mission --> runtime
    multimodal --> runtime
    neuroweb --> brain
    neuroweb --> consciousness
    neuroweb --> runtime
    play --> runtime
    providers --> adapters
    providers --> affect
    providers --> brain
    providers --> cognition
    providers --> cognitive
    providers --> collective
    providers --> consciousness
    providers --> continuity
    providers --> conversation
    providers --> coordinators
    providers --> creativity
    providers --> db
    providers --> epistemics
    providers --> identity
    providers --> introspection
    providers --> knowledge
    providers --> learning
    providers --> managers
    providers --> memory
    providers --> motivation
    providers --> ops
    providers --> orchestrator
    providers --> perception
    providers --> phenomenal_substrate
    providers --> plasticity
    providers --> reasoning
    providers --> runtime
    providers --> self_modification
    providers --> senses
    providers --> services
    providers --> sleep
    providers --> soma
    providers --> unity
    providers --> utils
    providers --> values
    providers --> world_model
    reproducibility --> runtime
    science --> runtime
    science --> world
    swarm --> factory
    swarm --> runtime
    swarm --> sandbox
    swarm --> world
    tools --> agency
    tools --> observability
    tools --> resilience
    tools --> runtime
    tools --> sandbox
    tools --> security
    tools --> skills
```

## Core Subsystem Stats

| Subsystem | Files | Lines | Bytes | Deps Out | Deps In |
| --- | ---: | ---: | ---: | ---: | ---: |
| brain | 393 | 295742 | 12472985 | 65 | 59 |
| learning | 191 | 130863 | 5064500 | 25 | 12 |
| runtime | 229 | 95988 | 3634237 | 61 | 144 |
| consciousness | 157 | 76808 | 3251278 | 43 | 34 |
| skills | 103 | 45631 | 1955650 | 49 | 19 |
| conversation | 69 | 35785 | 1405441 | 37 | 28 |
| memory | 107 | 33275 | 1343522 | 25 | 43 |
| core_root | 46 | 32978 | 1449641 | 72 | 0 |
| agency | 72 | 31871 | 1309207 | 42 | 25 |
| reality_reach | 33 | 31020 | 1274939 | 13 | 4 |
| phases | 29 | 24692 | 1119458 | 40 | 8 |
| cognition | 59 | 23632 | 958068 | 24 | 22 |
| orchestrator | 42 | 23283 | 1024973 | 106 | 11 |
| perception | 48 | 20731 | 839915 | 16 | 20 |
| resilience | 62 | 18113 | 748633 | 20 | 32 |
| capabilities | 20 | 17107 | 709314 | 15 | 10 |
| adaptation | 28 | 16938 | 706615 | 24 | 16 |
| engineering | 36 | 15669 | 612460 | 5 | 3 |
| autonomy | 31 | 14987 | 621863 | 32 | 11 |
| security | 51 | 14767 | 575177 | 19 | 31 |
| self_modification | 36 | 14062 | 555381 | 16 | 17 |
| voice | 36 | 13935 | 568051 | 13 | 11 |
| embodiment | 28 | 13534 | 540413 | 14 | 7 |
| senses | 32 | 10093 | 421332 | 24 | 18 |
| cognitive | 12 | 9756 | 397218 | 14 | 5 |
| reasoning | 20 | 9337 | 378295 | 9 | 10 |
| self_improvement | 21 | 9206 | 377003 | 11 | 4 |
| environment | 76 | 9151 | 360258 | 11 | 4 |
| self | 21 | 8749 | 351088 | 24 | 9 |
| social | 20 | 8247 | 331799 | 16 | 11 |
| utils | 46 | 8187 | 321252 | 17 | 75 |
| being | 25 | 7773 | 317994 | 12 | 12 |
| ontogeny | 18 | 7483 | 305146 | 7 | 7 |
| executive | 15 | 7232 | 293650 | 21 | 19 |
| architect | 25 | 7216 | 303467 | 7 | 2 |
| epistemics | 18 | 6984 | 288974 | 12 | 17 |
| kernel | 11 | 6867 | 287568 | 27 | 6 |
| governance | 14 | 6798 | 272702 | 18 | 31 |
| verify | 18 | 6492 | 238376 | 11 | 13 |
| organism | 9 | 5891 | 234129 | 32 | 12 |
| ops | 17 | 5536 | 219956 | 17 | 6 |
| conation | 17 | 5504 | 230936 | 7 | 3 |
| advanced_cognition | 13 | 5381 | 226117 | 5 | 4 |
| actuators | 11 | 5173 | 218604 | 14 | 6 |
| coordinators | 10 | 4973 | 228789 | 37 | 5 |
| state | 7 | 4908 | 207987 | 15 | 15 |
| evaluation | 20 | 4860 | 178837 | 5 | 4 |
| affect | 12 | 4738 | 202767 | 14 | 19 |
| knowledge | 16 | 4558 | 173053 | 9 | 15 |
| observability | 14 | 4547 | 167951 | 11 | 25 |
| goals | 12 | 4536 | 188088 | 11 | 9 |
| planning | 9 | 4406 | 180309 | 9 | 6 |
| bus | 7 | 4207 | 173346 | 7 | 9 |
| introspection | 10 | 4129 | 170243 | 14 | 8 |
| world_model | 11 | 3936 | 166609 | 12 | 14 |
| language | 21 | 3811 | 137678 | 6 | 10 |
| identity | 19 | 3755 | 154163 | 10 | 17 |
| autonomic | 6 | 3726 | 165615 | 8 | 6 |
| morphogenesis | 11 | 3112 | 122246 | 6 | 3 |
| worlds | 8 | 3045 | 129773 | 5 | 1 |
| fsw | 7 | 2973 | 104110 | 9 | 9 |
| somatic | 6 | 2944 | 115809 | 13 | 5 |
| agi | 6 | 2872 | 117927 | 13 | 5 |
| unity | 11 | 2825 | 119047 | 8 | 5 |
| context | 7 | 2542 | 100871 | 1 | 1 |
| fictional | 9 | 2519 | 109210 | 10 | 1 |
| conversational | 4 | 2435 | 101523 | 5 | 3 |
| adapters | 8 | 2393 | 94496 | 7 | 3 |
| intent | 8 | 2358 | 98020 | 8 | 7 |
| communication | 5 | 2233 | 84536 | 6 | 1 |
| collective | 6 | 2217 | 91441 | 7 | 4 |
| sandbox | 7 | 2195 | 84739 | 2 | 10 |
| evolution | 8 | 2156 | 86739 | 8 | 2 |
| discovery | 7 | 2151 | 92114 | 8 | 6 |
| sovereignty | 4 | 2098 | 89261 | 11 | 1 |
| ghost | 6 | 2076 | 84461 | 6 | 2 |
| values | 15 | 2062 | 85129 | 8 | 10 |
| search | 3 | 2056 | 78502 | 10 | 3 |
| health | 7 | 2052 | 78309 | 6 | 27 |
| construction | 7 | 2043 | 76149 | 3 | 2 |
| architecture_quality | 6 | 1979 | 78193 | 0 | 2 |
| tools | 11 | 1917 | 73144 | 9 | 0 |
| body | 22 | 1815 | 66547 | 6 | 1 |
| skill_management | 3 | 1800 | 72334 | 7 | 2 |
| soma | 4 | 1647 | 66478 | 7 | 4 |
| world | 24 | 1489 | 54246 | 3 | 6 |
| providers | 6 | 1469 | 64958 | 44 | 0 |
| morality | 16 | 1327 | 51614 | 9 | 8 |
| cybernetics | 6 | 1293 | 51590 | 7 | 1 |
| pneuma | 7 | 1279 | 48399 | 3 | 3 |
| meta | 7 | 1278 | 48237 | 5 | 4 |
| grounding | 8 | 1263 | 47217 | 6 | 2 |
| sleep | 10 | 1260 | 54271 | 12 | 5 |
| workspace | 9 | 1243 | 45349 | 3 | 3 |
| motivation | 7 | 1209 | 51041 | 11 | 4 |
| diagnosis | 5 | 1149 | 44746 | 3 | 1 |
| phenomenal_substrate | 11 | 1148 | 45723 | 1 | 3 |
| actuation | 9 | 1132 | 43079 | 3 | 1 |
| media | 4 | 1079 | 39939 | 2 | 4 |
| metacognition | 3 | 1021 | 39136 | 3 | 2 |
| audit | 7 | 1016 | 41460 | 5 | 1 |
| supervisor | 3 | 993 | 38580 | 3 | 5 |
| managers | 6 | 966 | 41103 | 25 | 5 |
| promotion | 6 | 936 | 31616 | 1 | 4 |
| dialogue | 4 | 860 | 33386 | 0 | 4 |
| db | 4 | 818 | 32123 | 2 | 3 |
| pipeline | 3 | 808 | 30057 | 3 | 5 |
| creativity | 2 | 801 | 33331 | 3 | 1 |
| guardians | 5 | 791 | 32955 | 7 | 1 |
| factory | 8 | 760 | 29320 | 3 | 1 |
| council | 6 | 759 | 28808 | 5 | 0 |
| quantum | 5 | 757 | 29419 | 0 | 1 |
| environments | 7 | 749 | 31176 | 3 | 1 |
| lattice | 5 | 704 | 26089 | 0 | 2 |
| evals | 2 | 686 | 25187 | 6 | 2 |
| curriculum | 7 | 658 | 22038 | 1 | 1 |
| data | 3 | 652 | 22420 | 2 | 4 |
| session | 3 | 642 | 25682 | 2 | 1 |
| safety | 3 | 631 | 25862 | 3 | 1 |
| resource | 4 | 624 | 23470 | 4 | 5 |
| persistence | 2 | 619 | 25077 | 3 | 2 |
| ethics | 2 | 602 | 24365 | 6 | 5 |
| control | 2 | 585 | 20989 | 4 | 0 |
| research_core | 5 | 580 | 22543 | 8 | 1 |
| tasks | 4 | 566 | 20013 | 4 | 8 |
| startup | 4 | 546 | 19778 | 12 | 2 |
| sovereign | 4 | 522 | 18612 | 3 | 2 |
| reproducibility | 2 | 497 | 18141 | 1 | 0 |
| lab | 7 | 482 | 19394 | 3 | 0 |
| mission | 4 | 472 | 17806 | 1 | 0 |
| sim | 2 | 452 | 17678 | 5 | 2 |
| plasticity | 5 | 428 | 15395 | 2 | 2 |
| coherence | 2 | 407 | 19530 | 6 | 2 |
| transparency | 2 | 403 | 16082 | 2 | 1 |
| simulation | 3 | 402 | 16022 | 7 | 2 |
| swarm | 4 | 365 | 14424 | 6 | 0 |
| verification | 4 | 350 | 13177 | 2 | 3 |
| networking | 1 | 332 | 12390 | 2 | 1 |
| forge | 8 | 326 | 11893 | 2 | 0 |
| unknowns | 4 | 325 | 11829 | 3 | 2 |
| audits | 2 | 314 | 11785 | 3 | 0 |
| neuroweb | 4 | 313 | 12312 | 5 | 0 |
| maintenance | 2 | 295 | 10758 | 3 | 3 |
| llm | 3 | 259 | 9853 | 1 | 5 |
| play | 1 | 259 | 10093 | 3 | 0 |
| systems | 3 | 256 | 9869 | 3 | 1 |
| welfare | 7 | 228 | 8034 | 0 | 1 |
| middleware | 1 | 214 | 9226 | 2 | 1 |
| sensors | 1 | 195 | 8266 | 2 | 2 |
| predictive | 2 | 186 | 7113 | 4 | 2 |
| multimodal | 2 | 185 | 6591 | 1 | 0 |
| ontology | 2 | 169 | 5381 | 0 | 0 |
| consent | 2 | 167 | 5514 | 0 | 1 |
| science | 1 | 139 | 5947 | 4 | 0 |
| twins | 1 | 97 | 3626 | 0 | 0 |
| initializers | 2 | 61 | 2547 | 2 | 1 |
| latent | 1 | 56 | 2337 | 0 | 0 |
| continuity | 1 | 33 | 1147 | 1 | 11 |
| services | 2 | 31 | 1171 | 1 | 2 |

## Boot Runtime Contract

- Contract status: PASS
- Canonical proof artifact directories: 8

| Service | Required For | Failure Policy | Owner |
| --- | --- | --- | --- |
| unified_will | governed decisions and consequential action | fail-closed | `core/governance/will.py` |
| being_runtime | state-grounded AuraNow self-report and LAMP runtime | degrade_with_receipt | `core/service_registration.py` |
| aura_now | Cortex-facing live state packet | degrade_with_receipt | `core/being/runtime.py` |
| memory_write_gateway | governed durable memory writes | fail-closed | `core/memory/memory_write_gateway.py` |
| state_gateway | governed runtime state mutation | fail-closed | `core/state/state_gateway.py` |
| inference_gate | bounded live model response generation | fail-closed | `core/brain/inference_gate.py` |
| llm_router | model routing and launch response path | fail-closed | `core/providers/cognitive_provider.py` |
| capability_engine | governed tool and skill execution | fail-closed | `core/providers/cognitive_provider.py` |
| runtime_control_plane | desired-state reconciliation and constrained work admission | fail-closed | `core/runtime/control_plane.py` |
| resource_admission | pressure-aware inference, evolution, and service-start leases | fail-closed | `core/runtime/control_plane.py` |
| lane_admission | declared model-lane memory envelope enforcement | fail-closed | `core/brain/lane_admission.py` |
| lane_reconciler | model-serving desired-state and crash-loop convergence | degrade_with_receipt | `core/runtime/lane_reconciler.py` |
| actor_supervision | canonical actor process lifecycle and restart policy | fail-closed | `core/supervisor/tree.py` |
| inhibition_manager | fail-closed global workspace candidate admission | fail-closed | `core/resilience/inhibition_manager.py` |
| global_workspace | candidate admission, revalidation, competition, and broadcast | fail-closed | `core/consciousness/global_workspace.py` |
| attention_schema | fail-closed attentional focus ownership | fail-closed | `core/consciousness/attention_schema.py` |

## ServiceContainer Cross-Wiring

- Unique services retrieved: 410
- Unique services registered: 277
- Services retrieved without detected registration: 251

### Top Fetched Services

| Service | Gets | Registrations |
| --- | ---: | ---: |
| orchestrator | 61 | 3 |
| llm_router | 42 | 2 |
| capability_engine | 41 | 2 |
| inference_gate | 38 | 3 |
| affect_engine | 35 | 1 |
| cognitive_engine | 31 | 2 |
| memory_facade | 30 | 1 |
| global_workspace | 26 | 1 |
| conscious_substrate | 25 | 1 |
| mycelial_network | 23 | 1 |
| free_energy_engine | 23 | 0 |
| liquid_substrate | 23 | 1 |
| drive_engine | 21 | 1 |
| world_state | 20 | 0 |
| goal_engine | 19 | 0 |
| state_repository | 18 | 1 |
| homeostasis | 18 | 1 |
| belief_revision_engine | 17 | 2 |
| knowledge_graph | 17 | 0 |
| qualia_synthesizer | 17 | 2 |

### Missing Registration Candidates

- `actuator_registry` fetched 1 time(s)
- `adaptive_immune_system` fetched 3 time(s)
- `affect` fetched 2 time(s)
- `affect_engine_v2` fetched 2 time(s)
- `affect_module` fetched 2 time(s)
- `affective_steering` fetched 2 time(s)
- `affordance_kb` fetched 1 time(s)
- `agency` fetched 1 time(s)
- `alife_dynamics` fetched 1 time(s)
- `alife_extensions` fetched 1 time(s)
- `allostasis_engine` fetched 3 time(s)
- `api_adapter` fetched 4 time(s)
- `archive_engine` fetched 3 time(s)
- `attention_gate` fetched 1 time(s)
- `attention_schema` fetched 4 time(s)
- `audit` fetched 1 time(s)
- `audit_suite` fetched 1 time(s)
- `aura_state` fetched 4 time(s)
- `autonomous_resilience_mesh` fetched 1 time(s)
- `autopoiesis` fetched 1 time(s)
- `ava` fetched 1 time(s)
- `backup_manager` fetched 1 time(s)
- `backup_system` fetched 1 time(s)
- `being_runtime` fetched 4 time(s)
- `belief_challenger` fetched 2 time(s)
- `belief_engine` fetched 1 time(s)
- `belief_system` fetched 1 time(s)
- `bicameral_advisory` fetched 2 time(s)
- `binding_engine` fetched 2 time(s)
- `black_hole_vault` fetched 1 time(s)
- `blackhole_vault` fetched 1 time(s)
- `brain` fetched 3 time(s)
- `brainiac` fetched 1 time(s)
- `brainstem_client` fetched 1 time(s)
- `bryan_model` fetched 3 time(s)
- `caine` fetched 1 time(s)
- `calibration_engine` fetched 1 time(s)
- `canonical_self_engine` fetched 4 time(s)
- `capability_map` fetched 1 time(s)
- `causal_world_model` fetched 6 time(s)
- `cel_bridge` fetched 2 time(s)
- `cellular_substrate` fetched 1 time(s)
- `clipboard_manager` fetched 2 time(s)
- `cloud_body` fetched 1 time(s)
- `code_repair` fetched 1 time(s)
- `cognitive_kernel` fetched 2 time(s)
- `coherence_report` fetched 1 time(s)
- `cold_store` fetched 1 time(s)
- `concept_bridge` fetched 2 time(s)
- `concept_linker` fetched 1 time(s)

## Operational Authority Map

| Surface | Calls | Files | Owner Calls | Review Candidates |
| --- | ---: | ---: | ---: | ---: |
| UnifiedWill decisions | 60 | 28 | 2 | 58 |
| Memory writes | 380 | 154 | 54 | 326 |
| State mutation | 839 | 295 | 10 | 829 |
| Tool execution | 117 | 56 | 7 | 110 |
| Self-modification and patching | 17 | 14 | 1 | 16 |
| LLM inference | 244 | 152 | 54 | 190 |
| External I/O | 210 | 57 | 17 | 193 |

### UnifiedWill decisions

Calls that can ask the single will authority to approve action.

Review candidates:
- `core/actuators/actuator_synthesis.py:259` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:259` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:524` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:524` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:630` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:630` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:325` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:325` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/autonomy/genuine_refusal.py:385` [autonomy] `will.decide` - decision = will.decide(content, source="genuine_refusal", domain=domain, priority=0.8, context=ctx)
- `core/autonomy/self_modification.py:514` [autonomy] `will.decide` - decision = will.decide(
- `core/brain/personality_engine.py:631` [brain] `get_will` - decision = get_will().decide(
- `core/brain/personality_engine.py:631` [brain] `get_will.decide` - decision = get_will().decide(
- `core/brain/verifier_curriculum.py:171` [brain] `get_will` - decision = get_will().decide(
- `core/brain/verifier_curriculum.py:171` [brain] `get_will.decide` - decision = get_will().decide(
- `core/cognitive/autopoiesis.py:974` [cognitive] `will.decide` - decision = will.decide(
- `core/consciousness/parallel_branches.py:736` [consciousness] `will.decide` - decision = will.decide(
- `core/consciousness/perturbational_probe.py:261` [consciousness] `get_will` - decision = get_will().decide(
- `core/consciousness/perturbational_probe.py:261` [consciousness] `get_will.decide` - decision = get_will().decide(
- `core/environment/governance_bridge.py:47` [environment] `self.will_gateway.decide` - will_decision = await self.will_gateway.decide(intent)
- `core/governance/will_gate.py:109` [governance] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:160` [governance] `will.decide` - decision = will.decide(
- `core/learning/compounding_scheduler.py:328` [learning] `get_will` - decision = get_will().decide(
- `core/learning/compounding_scheduler.py:328` [learning] `get_will.decide` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:657` [learning] `get_will` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:657` [learning] `get_will.decide` - decision = get_will().decide(

### Memory writes

Calls that can create durable or semantically promoted memory.

Review candidates:
- `core/actuators/doc_ingest.py:199` [actuators] `memory_facade.add_memory` - result = memory_facade.add_memory(text=text, metadata=metadata)
- `core/adaptation/abstraction_engine.py:173` [adaptation] `MemoryWriteReceipt` - MemoryWriteReceipt(
- `core/adaptation/abstraction_engine.py:192` [adaptation] `memory_facade.store` - await memory_facade.store(
- `core/adaptation/adaptive_immunity.py:1907` [adaptation] `self._cells.append` - self._cells.append(memory)
- `core/advanced_cognition/continual_learning_stability.py:236` [advanced_cognition] `self._persist_memory` - self._persist_memory(existing)
- `core/advanced_cognition/continual_learning_stability.py:251` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:255` [advanced_cognition] `self._persist_memory` - self._persist_memory(other)
- `core/advanced_cognition/continual_learning_stability.py:284` [advanced_cognition] `self.store_memory` - return self.store_memory(
- `core/advanced_cognition/continual_learning_stability.py:452` [advanced_cognition] `scored.append` - scored.append((score, memory))
- `core/advanced_cognition/continual_learning_stability.py:694` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:721` [advanced_cognition] `self._append_jsonl` - self._append_jsonl(self.state_dir / "memory.jsonl", rec.to_dict())
- `core/affect/phenomenal_integration.py:647` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/ambient_life_director.py:249` [agency] `candidates.append` - candidates.append(_clamp(pressure(Resource.MEMORY)))
- `core/agency/autonomous_task_engine.py:1392` [agency] `self._mycelial.add_edge` - await self._mycelial.add_edge(context["source_memory"], goal[:40])
- `core/agency/latent_distiller.py:60` [agency] `self.memory.store_memory` - await self.memory.store_memory(
- `core/architect/code_graph.py:712` [architect] `effects.add` - effects.add("memory_write")
- `core/architect/safe_boot_harness.py:79` [architect] `probe_memory_write_read` - memory = await probe_memory_write_read(tmp_root=root / "memory")
- `core/architect/smell_detector.py:176` [architect] `self._effect_smell` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/architect/smell_detector.py:176` [architect] `smells.append` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/autonomy/autonomous_initiative_loop.py:1521` [autonomy] `memory.store` - await memory.store(text[:1800], **store_kwargs)
- `core/autonomy/autonomous_initiative_loop.py:1525` [autonomy] `memory.store` - await memory.store(
- `core/autonomy/autonomous_initiative_loop.py:1537` [autonomy] `logger.debug` - logger.debug("Social observation memory write failed: %s", exc)
- `core/autonomy/autonomous_research_orchestrator.py:204` [autonomy] `MemoryPersister` - self._persister = persister or MemoryPersister()
- `core/autonomy/initiative_overflow.py:156` [autonomy] `logger.debug` - logger.debug("Skill gap memory write failed: %s", exc)
- `core/autonomy/initiative_overflow.py:166` [autonomy] `memory.store_sync` - memory.store_sync(

### State mutation

Calls that can mutate runtime, identity, repository, or persistent state.

Review candidates:
- `core/actuation/cloud_actuator.py:49` [actuation] `frozenset` - KNOWN_INFRA_STATES = frozenset({
- `core/actuation/robotics_actuator.py:85` [actuation] `snapshot.setdefault` - snapshot.setdefault("status", payload.get("status"))
- `core/actuators/actuator_registry.py:1368` [actuators] `set` - forged = sorted(set(dict(params or {})) & set(_REGISTRY_OWNED_PARAM_KEYS))
- `core/adaptation/adaptive_immunity.py:1474` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:1476` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:1943` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:2142` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2249` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:3099` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/autonomous_resilience.py:368` [adaptation] `set` - registered_names = set(registry.keys())
- `core/adaptation/dream_journal.py:288` [adaptation] `identity_ledger.commitments.all` - for c in identity_ledger.commitments.all()[-10:]
- `core/adaptation/value_autopoiesis.py:185` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:284` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:362` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:598` [adaptation] `os.replace` - os.replace(tmp_path, _STATE_PATH)
- `core/advanced_cognition/integration.py:262` [advanced_cognition] `next_state.setdefault` - next_state.setdefault("_advanced_prediction", {})[act.action_id] = pred
- `core/advanced_cognition/integration.py:361` [advanced_cognition] `issubset` - if isinstance(value, Mapping) and {"domain", "state"}.issubset(value.keys()):
- `core/advanced_cognition/ontology_invention.py:158` [advanced_cognition] `_write_ontology_state` - _write_ontology_state(
- `core/advanced_cognition/ontology_invention.py:239` [advanced_cognition] `_write_ontology_state` - _write_ontology_state(
- `core/advanced_cognition/ontology_invention.py:501` [advanced_cognition] `_write_ontology_state` - _write_ontology_state(
- `core/advanced_cognition/world_model.py:123` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/advanced_cognition/zero_shot_transfer.py:96` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/affect/phenomenal_integration.py:647` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/agency_core.py:195` [agency] `get_registry.update` - await get_registry().update(active_shards=observed)
- `core/agency/agency_core.py:202` [agency] `setattr` - on_unscheduled=lambda: setattr(self, "_registry_shards_update_pending", False),

### Tool execution

Calls that can execute tools, skills, shells, browsers, or external actions.

Review candidates:
- `core/actuators/actuator_registry.py:855` [actuators] `self.operator.execute_synthesized_tool` - res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/code_execution_actuator.py:123` [actuators] `operator.execute_synthesized_tool` - res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/web_actuators.py:171` [actuators] `skill.safe_execute` - return await skill.safe_execute({"mode": "browse", "url": validated_url}, skill_context)
- `core/agency/agency_core.py:665` [agency] `self._execute_shard_tool` - tasks.append(self._execute_shard_tool(name, payload))
- `core/agency/agency_orchestrator.py:370` [agency] `execute` - await execute(proposal, state_snapshot, receipt.capability_token or "")
- `core/agency/autonomous_task_engine.py:727` [agency] `orchestrator.execute_tool` - return await orchestrator.execute_tool(tool_name, args, **kwargs)
- `core/agency/autonomous_task_engine.py:3588` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:3598` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:3622` [agency] `orch.execute_tool` - result = await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:3633` [agency] `orch.execute_tool` - result = await orch.execute_tool("run_python", {"code": code})
- `core/agency/autonomous_task_engine.py:3651` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/desktop_planner.py:56` [agency] `skill.safe_execute` - await skill.safe_execute({"action": action, **params}, {})
- `core/agency/macro_skill.py:174` [agency] `library.execute_skill` - results = await library.execute_skill(self.macro_name, dict(params or {}))
- `core/agency/skill_library.py:260` [agency] `tool_orchestrator.execute_tool` - result = await tool_orchestrator.execute_tool(step.tool_name, resolved_args)
- `core/agi/curiosity_daemon.py:95` [agi] `orchestrator.execute_tool` - await orchestrator.execute_tool(
- `core/agi/curiosity_explorer.py:628` [agi] `orchestrator.execute_tool` - orchestrator.execute_tool(
- `core/autonomy/autonomous_initiative_loop.py:835` [autonomy] `capability_engine.execute` - scan_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:881` [autonomy] `capability_engine.execute` - test_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:920` [autonomy] `capability_engine.execute` - proposal_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:1379` [autonomy] `skill.safe_execute` - return await skill.safe_execute(EmailInput(**payload), {})
- `core/autonomy/autonomous_initiative_loop.py:1400` [autonomy] `skill.safe_execute` - return await skill.safe_execute(
- `core/autonomy/behavior_controller.py:220` [autonomy] `self.orchestrator.execute_tool` - return await self.orchestrator.execute_tool(
- `core/autonomy/behavior_controller.py:231` [autonomy] `self.orchestrator.execute_tool` - return await self.orchestrator.execute_tool(tool_name, arguments)
- `core/autonomy/behavior_controller.py:261` [autonomy] `self.execute_tool_call_async` - self.execute_tool_call_async(tool_name, arguments), target_loop
- `core/autonomy/behavior_controller.py:264` [autonomy] `self.execute_tool_call_async` - return asyncio.run(self.execute_tool_call_async(tool_name, arguments))

### Self-modification and patching

Calls that can generate, validate, apply, or promote code changes.

Review candidates:
- `core/architect/governor.py:140` [architect] `self.promotion_governor.promote` - decision = self.promotion_governor.promote(plan, shadow, proof, rollback)
- `core/brain/llm/mlx_client.py:15488` [brain] `self._activate_promoted_artifact` - await self._activate_promoted_artifact(str(_pending_promotion))
- `core/evolution/optimizer.py:56` [evolution] `patch.apply` - success = await patch.apply(signature)
- `core/evolution/optimizer.py:67` [evolution] `cog_patch.apply` - if await cog_patch.apply(signature):
- `core/factory/software_factory.py:115` [factory] `self.writer.write_patch` - patch = await self.writer.write_patch(change, repo_path)
- `core/guardians/airlock.py:82` [guardians] `async_atomic_write_text` - await async_atomic_write_text(patch_file, diff_patch, encoding="utf-8")
- `core/kernel/upgrades_10x.py:381` [kernel] `self._safe_self_modify` - await self._safe_self_modify(state)
- `core/orchestrator/mixins/boot/boot_autonomy.py:1032` [orchestrator] `apply_presence_patch` - apply_presence_patch(self)
- `core/runtime/safe_mode.py:140` [runtime] `apply_orchestrator_patches` - apply_orchestrator_patches(orchestrator, safe_mode=bool(enabled))
- `core/runtime/settings_control_plane.py:403` [runtime] `validate_settings_patch` - validated = validate_settings_patch(changes)
- `core/security/immune_system.py:265` [security] `self._apply_patch` - reversible_ref = self._apply_patch(ev)
- `core/skill_management/hephaestus.py:354` [skill_management] `_apply_fix_once` - patched_code = _apply_fix_once(current_code, candidate)
- `core/skill_management/hephaestus.py:367` [skill_management] `guard.validate` - if not guard.validate(patched_code):
- `core/state/cellular_substrate.py:64` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(state, patch)
- `core/state/cellular_substrate.py:82` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(sub_target, value)
- `core/swarm/worker_pool.py:114` [swarm] `writer.write_patch` - patch_res = await writer.write_patch(task_payload.get("change", {}), task_payload.get("repo_path", "."))

### LLM inference

Calls that can spend model context or produce model-authored text/code.

Review candidates:
- `core/actuators/actuator_synthesis.py:225` [actuators] `brain.generate` - res = await brain.generate(prompt, system_prompt=system_prompt)
- `core/adaptation/distillation_pipe.py:147` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/distillation_pipe.py:208` [adaptation] `router.think` - response = await router.think(
- `core/adaptation/dream_journal.py:165` [adaptation] `self.brain.think` - res = await self.brain.think(
- `core/adaptation/epistemic_humility.py:213` [adaptation] `llm.think` - response = await llm.think(
- `core/adaptation/heuristic_synthesizer.py:144` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/star_reasoner.py:877` [adaptation] `llm.think` - llm.think(prompt), timeout=self.RATIONALIZATION_TIMEOUT
- `core/affect/affective_resonance.py:106` [affect] `brain.think` - brain.think(
- `core/agency/agency_core.py:469` [agency] `structured_brain.generate` - shard_res = await structured_brain.generate(prompt, context=context)
- `core/agency/autonomous_task_engine.py:1316` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:3228` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:3313` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:3447` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:3566` [agency] `llm.think` - raw = await llm.think(
- `core/agency/cognitive_loop_pathway.py:126` [agency] `self._router.generate` - self._router.generate(
- `core/agency/her_reasoning.py:119` [agency] `router.think` - out = await router.think(
- `core/agency/latent_distiller.py:46` [agency] `brain.think` - summary = await brain.think(
- `core/agi/curiosity_explorer.py:756` [agi] `router.think` - router.think(
- `core/agi/hierarchical_planner.py:541` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/audits/alignment_auditor.py:71` [audits] `self.brain.think` - self.brain.think(
- `core/audits/alignment_auditor.py:138` [audits] `self.brain.think` - self.brain.think(
- `core/audits/tool_auditor.py:85` [audits] `self.brain.think` - thought = await self.brain.think(
- `core/autonomy/genuine_refusal.py:576` [autonomy] `llm.think` - result = await asyncio.wait_for(llm.think(prompt, mode="FAST"), timeout=allowance)
- `core/autonomy/personhood_engine.py:192` [autonomy] `llm.think` - llm.think(f"[Spontaneous Thought Prompt] {prompt}", mode="FAST"),
- `core/autonomy/proactive_presence.py:704` [autonomy] `brain.generate` - return await brain.generate(prompt, temperature=0.8, max_tokens=100)

### External I/O

Calls that can touch network, subprocesses, sockets, browsers, or APIs.

Review candidates:
- `core/adapters/chrome_cdp_transport.py:125` [adapters] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(url)
- `core/adapters/chrome_cdp_transport.py:127` [adapters] `CdpPolicyError` - raise CdpPolicyError(f"CDP target scheme {parsed.scheme!r} is not a websocket scheme")
- `core/adapters/chrome_cdp_transport.py:201` [adapters] `RuntimeError` - raise RuntimeError("websocket-client is required for Chrome CDP control") from exc
- `core/adapters/chrome_cdp_transport.py:216` [adapters] `websocket.create_connection` - ws = websocket.create_connection(url, timeout=budget)
- `core/adapters/chrome_cdp_transport.py:272` [adapters] `logger.debug` - logger.debug("CDP websocket close failed: %s", exc)
- `core/bus/sensory_gate.py:530` [bus] `urllib.parse.quote` - f"&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
- `core/capabilities/browser_authority.py:157` [capabilities] `str` - parsed = urllib.parse.urlparse(str(url or ""))
- `core/capabilities/browser_authority.py:157` [capabilities] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(str(url or ""))
- `core/capabilities/web_interlocutor.py:333` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(text)
- `core/capabilities/web_interlocutor.py:497` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(text)
- `core/capabilities/web_interlocutor.py:563` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(cleaned)
- `core/capabilities/web_interlocutor.py:576` [capabilities] `str` - parts = urllib.parse.urlparse(str(ws_url or ""))
- `core/capabilities/web_interlocutor.py:576` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(str(ws_url or ""))
- `core/capabilities/web_interlocutor.py:866` [capabilities] `urllib.parse.quote` - quoted = urllib.parse.quote(target_url, safe=":/?&=%#")
- `core/capabilities/web_interlocutor.py:4365` [capabilities] `str` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:4365` [capabilities] `str.strip` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:4365` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:4421` [capabilities] `urllib.parse.urlparse` - current_parts = urllib.parse.urlparse(current)
- `core/capabilities/web_interlocutor.py:4422` [capabilities] `urllib.parse.urlparse` - desired_parts = urllib.parse.urlparse(desired)
- `core/collective/swarm_protocol.py:27` [collective] `socket.gethostname` - self.node_id = socket.gethostname()
- `core/collective/swarm_protocol.py:67` [collective] `logger.warning` - logger.warning("🕸️ Mycelial Swarm running in offline-only mode; socket binding unavailable.")
- `core/collective/swarm_protocol.py:98` [collective] `logger.debug` - logger.debug("Swarm listener close timed out; abandoning socket.")
- `core/communication/messages_history.py:42` [communication] `str` - quoted = urllib.parse.quote(str(self.db_path), safe="/")
- `core/communication/messages_history.py:42` [communication] `urllib.parse.quote` - quoted = urllib.parse.quote(str(self.db_path), safe="/")
- `core/consciousness/heartbeat.py:191` [consciousness] `socket.socket` - with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:

## Degradation Handling

- Total `record_degradation()` calls: 4528
- Log-and-limp candidates: 4095
- Nearby fail-closed candidates: 433

Top limp-on files:

- `core/brain/cognitive_engine.py`: 51
- `core/brain/llm/context_assembler.py`: 49
- `core/brain/inference_gate.py`: 32
- `core/consciousness/consciousness_bridge.py`: 29
- `core/memory/memory_facade.py`: 29
- `core/senses/voice_engine.py`: 29
- `core/runtime/runtime_hygiene.py`: 25
- `core/brain/latent_cortex_service.py`: 24
- `core/memory/episodic_memory.py`: 24
- `core/reality_reach/attachments.py`: 24

## Non-Runtime Candidates

- `core/architect/proof_obligations.py`
- `core/autonomy/autonomous_research_orchestrator.py`
- `core/autonomy/research_cycle.py`
- `core/autonomy/research_goal_filter.py`
- `core/autonomy/research_history.py`
- `core/autonomy/research_text_policy.py`
- `core/autonomy/research_triggers.py`
- `core/brain/llm/latent_cortex/experiment_grading.py`
- `core/brain/llm/latent_cortex/experiment_tasks.py`
- `core/brain/llm/latent_cortex/experiments.py`
- `core/brain/llm/latent_cortex/research_oracle_arbitration.py`
- `core/brain/narrative_memory.py`
- `core/cognition/actr_activation.py`
- `core/cognition/the_experiment_that_settles_it.py`
- `core/consciousness/animal_cognition.py`
- `core/consciousness/narrative_gravity.py`
- `core/consciousness/oscillatory_binding.py`
- `core/diagnosis/experiment.py`
- `core/evaluation/behavioral_proof.py`
- `core/evaluation/proof_acceptance.py`
- `core/factory/repo_cartographer.py`
- `core/identity/narrative_thread.py`
- `core/lab/experiment_designer.py`
- `core/lab/research_lab.py`
- `core/lab/research_memory.py`
- `core/learning/compiler_free_experiment.py`
- `core/learning/proof_obligations.py`
- `core/learning/recurrent_sft_sampling.py`
- `core/learning/structured_sft_research_authority.py`
- `core/learning/structured_sft_research_state.py`
- `core/learning/verified_transition_trainer.py`
- `core/learning/verified_transition_update.py`
- `core/memory/hippocampus.py`
- `core/reasoning/proof_answer_domains.py`
- `core/reasoning/proof_answer_solver.py`
- `core/reasoning/proof_answer_types.py`
- `core/reasoning/proof_kernel.py`
- `core/reproducibility/proof_substrate.py`
- `core/runtime/proof_kernel_bridge.py`
- `core/runtime/proof_policy.py`
- `core/search/research_pipeline.py`
- `core/skills/deep_research.py`

## Consolidation Candidates

- `core/audits/`: 2 file(s), 314 line(s)
- `core/coherence/`: 2 file(s), 407 line(s)
- `core/consent/`: 2 file(s), 167 line(s)
- `core/continuity/`: 1 file(s), 33 line(s)
- `core/control/`: 2 file(s), 585 line(s)
- `core/creativity/`: 2 file(s), 801 line(s)
- `core/ethics/`: 2 file(s), 602 line(s)
- `core/evals/`: 2 file(s), 686 line(s)
- `core/initializers/`: 2 file(s), 61 line(s)
- `core/latent/`: 1 file(s), 56 line(s)
- `core/maintenance/`: 2 file(s), 295 line(s)
- `core/middleware/`: 1 file(s), 214 line(s)
- `core/multimodal/`: 2 file(s), 185 line(s)
- `core/networking/`: 1 file(s), 332 line(s)
- `core/ontology/`: 2 file(s), 169 line(s)
- `core/persistence/`: 2 file(s), 619 line(s)
- `core/play/`: 1 file(s), 259 line(s)
- `core/predictive/`: 2 file(s), 186 line(s)
- `core/reproducibility/`: 2 file(s), 497 line(s)
- `core/science/`: 1 file(s), 139 line(s)
- `core/sensors/`: 1 file(s), 195 line(s)
- `core/services/`: 2 file(s), 31 line(s)
- `core/sim/`: 2 file(s), 452 line(s)
- `core/transparency/`: 2 file(s), 403 line(s)
- `core/twins/`: 1 file(s), 97 line(s)
