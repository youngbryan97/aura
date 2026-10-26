# Aura Architecture Dependency Map

Schema: `aura.architecture.dependency_map.v2`
Root: `<AURA_ROOT>`
Generated: `0.0`

## Summary

- Subsystems: 169
- Python files: 4557
- Python lines: 1804251
- Dependency edges: 1490
- ServiceContainer `.get()` calls: 1353
- ServiceContainer registrations: 334
- Boot contract: PASS

## Subsystem Dependency Graph

```mermaid
graph TD
    runtime["runtime<br/>295 files, 113423 lines"]
    utils["utils<br/>51 files, 8948 lines"]
    brain["brain<br/>476 files, 316671 lines"]
    conversation["conversation<br/>82 files, 40313 lines"]
    memory["memory<br/>121 files, 36991 lines"]
    consciousness["consciousness<br/>188 files, 87403 lines"]
    verify["verify<br/>37 files, 12098 lines"]
    security["security<br/>38 files, 14088 lines"]
    resilience["resilience<br/>64 files, 19542 lines"]
    governance["governance<br/>17 files, 7266 lines"]
    agency["agency<br/>239 files, 74388 lines"]
    cognition["cognition<br/>205 files, 67370 lines"]
    health["health<br/>7 files, 2095 lines"]
    perception["perception<br/>212 files, 60423 lines"]
    observability["observability<br/>19 files, 5556 lines"]
    affect["affect<br/>36 files, 9767 lines"]
    knowledge["knowledge<br/>28 files, 7663 lines"]
    skills["skills<br/>164 files, 69385 lines"]
    executive["executive<br/>17 files, 7600 lines"]
    identity["identity<br/>22 files, 4856 lines"]
    state["state<br/>22 files, 9373 lines"]
    self_modification["self_modification<br/>53 files, 18475 lines"]
    senses["senses<br/>33 files, 10492 lines"]
    adaptation["adaptation<br/>30 files, 17486 lines"]
    epistemics["epistemics<br/>21 files, 7893 lines"]
    self["self<br/>38 files, 12658 lines"]
    world_model["world_model<br/>14 files, 5142 lines"]
    language["language<br/>35 files, 7208 lines"]
    social["social<br/>49 files, 14396 lines"]
    being["being<br/>25 files, 8089 lines"]
    fsw["fsw<br/>9 files, 3580 lines"]
    learning["learning<br/>366 files, 186834 lines"]
    soma["soma<br/>12 files, 3124 lines"]
    organism["organism<br/>23 files, 11665 lines"]
    sandbox["sandbox<br/>7 files, 2244 lines"]
    autonomy["autonomy<br/>32 files, 15247 lines"]
    capabilities["capabilities<br/>29 files, 19746 lines"]
    conation["conation<br/>20 files, 7452 lines"]
    continuity["continuity<br/>1 files, 33 lines"]
    orchestrator["orchestrator<br/>44 files, 23860 lines"]
    reasoning["reasoning<br/>32 files, 11851 lines"]
    values["values<br/>16 files, 2406 lines"]
    voice["voice<br/>40 files, 14758 lines"]
    intent["intent<br/>12 files, 2958 lines"]
    ontogeny["ontogeny<br/>20 files, 8327 lines"]
    goals["goals<br/>15 files, 5036 lines"]
    phases["phases<br/>44 files, 32993 lines"]
    bus["bus<br/>7 files, 4214 lines"]
    introspection["introspection<br/>11 files, 4904 lines"]
    morality["morality<br/>18 files, 1806 lines"]
    tasks["tasks<br/>4 files, 567 lines"]
    canonical["canonical<br/>3 files, 684 lines"]
    embodiment["embodiment<br/>29 files, 13973 lines"]
    evaluation["evaluation<br/>25 files, 6083 lines"]
    interiority["interiority<br/>72 files, 16765 lines"]
    motivation["motivation<br/>10 files, 1880 lines"]
    actuators["actuators<br/>11 files, 5205 lines"]
    autonomic["autonomic<br/>6 files, 3733 lines"]
    cognitive["cognitive<br/>12 files, 9822 lines"]
    discovery["discovery<br/>7 files, 2185 lines"]
    ethics["ethics<br/>4 files, 1150 lines"]
    kernel["kernel<br/>13 files, 7311 lines"]
    ops["ops<br/>17 files, 5805 lines"]
    pipeline["pipeline<br/>3 files, 816 lines"]
    planning["planning<br/>9 files, 4580 lines"]
    world["world<br/>24 files, 1560 lines"]
    agi["agi<br/>7 files, 3137 lines"]
    coordinators["coordinators<br/>11 files, 5229 lines"]
    llm["llm<br/>3 files, 259 lines"]
    managers["managers<br/>6 files, 975 lines"]
    resource["resource<br/>4 files, 702 lines"]
    sleep["sleep<br/>10 files, 1275 lines"]
    somatic["somatic<br/>6 files, 2961 lines"]
    supervisor["supervisor<br/>3 files, 1017 lines"]
    unity["unity<br/>11 files, 2870 lines"]
    advanced_cognition["advanced_cognition<br/>13 files, 5412 lines"]
    collective["collective<br/>6 files, 2245 lines"]
    data["data<br/>3 files, 652 lines"]
    dialogue["dialogue<br/>4 files, 978 lines"]
    engineering["engineering<br/>36 files, 15732 lines"]
    environment["environment<br/>78 files, 9633 lines"]
    expression["expression<br/>5 files, 906 lines"]
    media["media<br/>4 files, 1100 lines"]
    meta["meta<br/>7 files, 1307 lines"]
    morphogenesis["morphogenesis<br/>28 files, 12097 lines"]
    promotion["promotion<br/>7 files, 1248 lines"]
    reality_reach["reality_reach<br/>33 files, 31032 lines"]
    self_improvement["self_improvement<br/>22 files, 9362 lines"]
    together["together<br/>7 files, 1471 lines"]
    adapters["adapters<br/>8 files, 2403 lines"]
    conversational["conversational<br/>4 files, 2437 lines"]
    db["db<br/>4 files, 818 lines"]
    evidence["evidence<br/>7 files, 1983 lines"]
    maintenance["maintenance<br/>2 files, 295 lines"]
    persistence["persistence<br/>4 files, 1022 lines"]
    phenomenal_substrate["phenomenal_substrate<br/>11 files, 1160 lines"]
    pneuma["pneuma<br/>7 files, 1279 lines"]
    search["search<br/>4 files, 2139 lines"]
    verification["verification<br/>4 files, 350 lines"]
    workspace["workspace<br/>9 files, 1243 lines"]
    architect["architect<br/>25 files, 7247 lines"]
    architecture_quality["architecture_quality<br/>7 files, 2019 lines"]
    coherence["coherence<br/>2 files, 413 lines"]
    connectome["connectome<br/>44 files, 19211 lines"]
    construction["construction<br/>7 files, 2069 lines"]
    evals["evals<br/>3 files, 1233 lines"]
    evolution["evolution<br/>8 files, 2161 lines"]
    ghost["ghost<br/>6 files, 2145 lines"]
    grounding["grounding<br/>8 files, 1265 lines"]
    lattice["lattice<br/>5 files, 704 lines"]
    metacognition["metacognition<br/>3 files, 1048 lines"]
    plasticity["plasticity<br/>5 files, 428 lines"]
    predictive["predictive<br/>2 files, 186 lines"]
    sensors["sensors<br/>1 files, 245 lines"]
    sim["sim<br/>2 files, 452 lines"]
    simulation["simulation<br/>3 files, 413 lines"]
    skill_management["skill_management<br/>4 files, 2042 lines"]
    sovereign["sovereign<br/>4 files, 524 lines"]
    startup["startup<br/>4 files, 549 lines"]
    unknowns["unknowns<br/>4 files, 325 lines"]
    actuation["actuation<br/>9 files, 1164 lines"]
    audit["audit<br/>7 files, 1028 lines"]
    body["body<br/>22 files, 1820 lines"]
    communication["communication<br/>5 files, 2258 lines"]
    consent["consent<br/>2 files, 167 lines"]
    context["context<br/>7 files, 2543 lines"]
    creativity["creativity<br/>3 files, 1040 lines"]
    curriculum["curriculum<br/>7 files, 658 lines"]
    cybernetics["cybernetics<br/>6 files, 1314 lines"]
    diagnosis["diagnosis<br/>5 files, 1149 lines"]
    environments["environments<br/>7 files, 752 lines"]
    factory["factory<br/>8 files, 770 lines"]
    fictional["fictional<br/>9 files, 2534 lines"]
    guardians["guardians<br/>5 files, 807 lines"]
    initializers["initializers<br/>2 files, 61 lines"]
    middleware["middleware<br/>1 files, 266 lines"]
    networking["networking<br/>1 files, 329 lines"]
    quantum["quantum<br/>5 files, 767 lines"]
    rebuilding["rebuilding<br/>22 files, 5425 lines"]
    research_core["research_core<br/>5 files, 636 lines"]
    safety["safety<br/>3 files, 631 lines"]
    science["science<br/>19 files, 4796 lines"]
    services["services<br/>2 files, 31 lines"]
    session["session<br/>3 files, 642 lines"]
    sovereignty["sovereignty<br/>4 files, 2098 lines"]
    systems["systems<br/>3 files, 254 lines"]
    transparency["transparency<br/>2 files, 412 lines"]
    welfare["welfare<br/>7 files, 228 lines"]
    worlds["worlds<br/>8 files, 3046 lines"]
    audits["audits<br/>2 files, 316 lines"]
    control["control<br/>2 files, 585 lines"]
    core_root["core_root<br/>53 files, 33662 lines"]
    council["council<br/>6 files, 763 lines"]
    forge["forge<br/>8 files, 326 lines"]
    lab["lab<br/>7 files, 482 lines"]
    latent["latent<br/>1 files, 56 lines"]
    mission["mission<br/>4 files, 472 lines"]
    multimodal["multimodal<br/>2 files, 185 lines"]
    neuroweb["neuroweb<br/>4 files, 313 lines"]
    ontology["ontology<br/>2 files, 169 lines"]
    phenomenology["phenomenology<br/>9 files, 1885 lines"]
    play["play<br/>1 files, 260 lines"]
    providers["providers<br/>6 files, 1560 lines"]
    reproducibility["reproducibility<br/>2 files, 498 lines"]
    stand_ins["stand_ins<br/>4 files, 629 lines"]
    subject["subject<br/>69 files, 28422 lines"]
    swarm["swarm<br/>4 files, 366 lines"]
    tools["tools<br/>11 files, 1956 lines"]
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
    brain --> conation
    brain --> consciousness
    brain --> continuity
    brain --> conversation
    brain --> dialogue
    brain --> discovery
    brain --> engineering
    brain --> epistemics
    brain --> evaluation
    brain --> executive
    brain --> expression
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
    brain --> social
    brain --> state
    brain --> together
    brain --> utils
    brain --> verify
    brain --> voice
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
    conversation --> interiority
    conversation --> introspection
    conversation --> knowledge
    conversation --> language
    conversation --> memory
    conversation --> organism
    conversation --> perception
    conversation --> reasoning
    conversation --> resilience
    conversation --> runtime
    conversation --> security
    conversation --> self
    conversation --> senses
    conversation --> skills
    conversation --> social
    conversation --> state
    conversation --> utils
    conversation --> verify
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
    memory --> soma
    memory --> utils
    memory --> values
    consciousness --> adaptation
    consciousness --> affect
    consciousness --> agency
    consciousness --> being
    consciousness --> brain
    consciousness --> canonical
    consciousness --> cognition
    consciousness --> conation
    consciousness --> connectome
    consciousness --> continuity
    consciousness --> conversation
    consciousness --> coordinators
    consciousness --> evals
    consciousness --> evaluation
    consciousness --> executive
    consciousness --> expression
    consciousness --> ghost
    consciousness --> goals
    consciousness --> governance
    consciousness --> health
    consciousness --> kernel
    consciousness --> memory
    consciousness --> meta
    consciousness --> observability
    consciousness --> ontogeny
    consciousness --> orchestrator
    consciousness --> pneuma
    consciousness --> predictive
    consciousness --> reasoning
    consciousness --> resilience
    consciousness --> runtime
    consciousness --> self
    consciousness --> senses
    consciousness --> sensors
    consciousness --> social
    consciousness --> soma
    consciousness --> state
    consciousness --> unity
    consciousness --> utils
    consciousness --> verify
    consciousness --> world_model
    verify --> bus
    verify --> conversation
    verify --> fsw
    verify --> governance
    verify --> health
    verify --> knowledge
    verify --> observability
    verify --> organism
    verify --> runtime
    verify --> security
    verify --> skills
    verify --> values
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
    resilience --> soma
    resilience --> state
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
    agency --> adaptation
    agency --> affect
    agency --> agi
    agency --> autonomy
    agency --> brain
    agency --> canonical
    agency --> capabilities
    agency --> cognition
    agency --> conation
    agency --> consciousness
    agency --> continuity
    agency --> conversation
    agency --> executive
    agency --> goals
    agency --> governance
    agency --> health
    agency --> identity
    agency --> interiority
    agency --> knowledge
    agency --> language
    agency --> learning
    agency --> morality
    agency --> motivation
    agency --> observability
    agency --> ontogeny
    agency --> orchestrator
    agency --> organism
    agency --> perception
    agency --> persistence
    agency --> phases
    agency --> resilience
    agency --> runtime
    agency --> self
    agency --> skills
    agency --> social
    agency --> soma
    agency --> state
    agency --> tasks
    agency --> utils
    agency --> values
    agency --> verify
    agency --> world_model
    cognition --> actuators
    cognition --> affect
    cognition --> agency
    cognition --> brain
    cognition --> connectome
    cognition --> consciousness
    cognition --> conversation
    cognition --> epistemics
    cognition --> evidence
    cognition --> fsw
    cognition --> governance
    cognition --> introspection
    cognition --> knowledge
    cognition --> language
    cognition --> learning
    cognition --> memory
    cognition --> perception
    cognition --> runtime
    cognition --> security
    cognition --> sim
    cognition --> skills
    cognition --> social
    cognition --> utils
    cognition --> verify
    cognition --> voice
    cognition --> world_model
    health --> memory
    health --> runtime
    health --> state
    health --> utils
    perception --> brain
    perception --> capabilities
    perception --> cognition
    perception --> conversation
    perception --> governance
    perception --> language
    perception --> media
    perception --> phenomenal_substrate
    perception --> resilience
    perception --> runtime
    perception --> security
    perception --> senses
    perception --> together
    perception --> utils
    perception --> verify
    perception --> voice
    observability --> health
    observability --> memory
    observability --> pipeline
    observability --> runtime
    observability --> utils
    affect --> adaptation
    affect --> agency
    affect --> autonomic
    affect --> canonical
    affect --> consciousness
    affect --> conversation
    affect --> health
    affect --> interiority
    affect --> memory
    affect --> phenomenal_substrate
    affect --> runtime
    affect --> self
    affect --> senses
    affect --> state
    affect --> utils
    affect --> verify
    knowledge --> brain
    knowledge --> evidence
    knowledge --> reasoning
    knowledge --> runtime
    knowledge --> utils
    skills --> actuators
    skills --> advanced_cognition
    skills --> affect
    skills --> agency
    skills --> being
    skills --> brain
    skills --> capabilities
    skills --> cognition
    skills --> communication
    skills --> conation
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
    skills --> intent
    skills --> introspection
    skills --> knowledge
    skills --> language
    skills --> learning
    skills --> memory
    skills --> perception
    skills --> quantum
    skills --> reality_reach
    skills --> rebuilding
    skills --> runtime
    skills --> sandbox
    skills --> search
    skills --> security
    skills --> self
    skills --> self_improvement
    skills --> self_modification
    skills --> senses
    skills --> sovereign
    skills --> together
    skills --> utils
    skills --> verify
    skills --> voice
    skills --> world_model
    skills --> worlds
    executive --> agency
    executive --> autonomy
    executive --> consciousness
    executive --> continuity
    executive --> conversation
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
    identity --> agency
    identity --> brain
    identity --> canonical
    identity --> governance
    identity --> organism
    identity --> runtime
    identity --> utils
    identity --> verify
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
    state --> verify
    self_modification --> agency
    self_modification --> architecture_quality
    self_modification --> brain
    self_modification --> bus
    self_modification --> conversation
    self_modification --> ethics
    self_modification --> governance
    self_modification --> knowledge
    self_modification --> memory
    self_modification --> ops
    self_modification --> perception
    self_modification --> resilience
    self_modification --> runtime
    self_modification --> sandbox
    self_modification --> security
    self_modification --> skills
    self_modification --> utils
    self_modification --> verify
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
    epistemics --> being
    epistemics --> brain
    epistemics --> conversation
    epistemics --> knowledge
    epistemics --> observability
    epistemics --> reasoning
    epistemics --> runtime
    epistemics --> skills
    epistemics --> utils
    self --> affect
    self --> agency
    self --> being
    self --> cognitive
    self --> consciousness
    self --> conversation
    self --> dialogue
    self --> epistemics
    self --> fsw
    self --> identity
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
    world_model --> adaptation
    world_model --> advanced_cognition
    world_model --> brain
    world_model --> cognition
    world_model --> health
    world_model --> resilience
    world_model --> runtime
    world_model --> soma
    world_model --> state
    world_model --> values
    world_model --> verify
    language --> brain
    language --> conversation
    language --> memory
    language --> runtime
    language --> verify
    social --> agency
    social --> autonomy
    social --> brain
    social --> consciousness
    social --> epistemics
    social --> ethics
    social --> expression
    social --> governance
    social --> memory
    social --> runtime
    social --> security
    social --> self
    social --> senses
    social --> soma
    social --> utils
    social --> verify
    being --> agency
    being --> brain
    being --> canonical
    being --> consciousness
    being --> conversation
    being --> epistemics
    being --> governance
    being --> observability
    being --> runtime
    being --> verify
    fsw --> health
    fsw --> observability
    fsw --> pipeline
    fsw --> runtime
    fsw --> utils
    fsw --> verify
    learning --> agi
    learning --> architecture_quality
    learning --> brain
    learning --> cognition
    learning --> consciousness
    learning --> conversation
    learning --> evaluation
    learning --> evidence
    learning --> executive
    learning --> intent
    learning --> introspection
    learning --> language
    learning --> memory
    learning --> orchestrator
    learning --> perception
    learning --> promotion
    learning --> reasoning
    learning --> runtime
    learning --> sandbox
    learning --> security
    learning --> self_modification
    learning --> skills
    learning --> tasks
    learning --> utils
    learning --> verify
    learning --> world_model
    soma --> continuity
    soma --> resilience
    soma --> runtime
    soma --> self
    soma --> social
    soma --> utils
    organism --> adaptation
    organism --> affect
    organism --> agency
    organism --> being
    organism --> body
    organism --> brain
    organism --> capabilities
    organism --> cognition
    organism --> conation
    organism --> conversation
    organism --> epistemics
    organism --> ethics
    organism --> executive
    organism --> fsw
    organism --> governance
    organism --> health
    organism --> identity
    organism --> intent
    organism --> interiority
    organism --> introspection
    organism --> knowledge
    organism --> language
    organism --> learning
    organism --> memory
    organism --> morphogenesis
    organism --> perception
    organism --> reality_reach
    organism --> reasoning
    organism --> resilience
    organism --> runtime
    organism --> sandbox
    organism --> security
    organism --> skills
    organism --> sleep
    organism --> social
    organism --> utils
    organism --> values
    organism --> verify
    organism --> welfare
    organism --> workspace
    organism --> world
    sandbox --> runtime
    autonomy --> affect
    autonomy --> agency
    autonomy --> brain
    autonomy --> cognition
    autonomy --> conation
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
    capabilities --> adapters
    capabilities --> agency
    capabilities --> brain
    capabilities --> conversation
    capabilities --> executive
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
    capabilities --> verify
    conation --> affect
    conation --> canonical
    conation --> fsw
    conation --> memory
    conation --> ontogeny
    conation --> runtime
    conation --> verify
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
    orchestrator --> interiority
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
    reasoning --> cognition
    reasoning --> engineering
    reasoning --> governance
    reasoning --> observability
    reasoning --> planning
    reasoning --> runtime
    reasoning --> utils
    reasoning --> verify
    values --> agency
    values --> conversation
    values --> governance
    values --> runtime
    values --> social
    values --> utils
    voice --> brain
    voice --> conversation
    voice --> conversational
    voice --> executive
    voice --> managers
    voice --> resilience
    voice --> runtime
    voice --> senses
    voice --> utils
    voice --> verify
    intent --> brain
    intent --> conversation
    intent --> epistemics
    intent --> language
    intent --> runtime
    intent --> skills
    intent --> utils
    intent --> verify
    ontogeny --> fsw
    ontogeny --> runtime
    ontogeny --> verify
    ontogeny --> world_model
    goals --> agency
    goals --> autonomy
    goals --> brain
    goals --> cognition
    goals --> runtime
    goals --> state
    goals --> utils
    goals --> values
    phases --> adaptation
    phases --> affect
    phases --> agency
    phases --> autonomy
    phases --> being
    phases --> brain
    phases --> cognition
    phases --> coherence
    phases --> conation
    phases --> consciousness
    phases --> conversation
    phases --> conversational
    phases --> embodiment
    phases --> evaluation
    phases --> expression
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
    phases --> motivation
    phases --> ontogeny
    phases --> perception
    phases --> reasoning
    phases --> resilience
    phases --> runtime
    phases --> self
    phases --> self_modification
    phases --> skills
    phases --> social
    phases --> soma
    phases --> somatic
    phases --> state
    phases --> together
    phases --> unity
    phases --> utils
    phases --> verify
    phases --> voice
    phases --> world_model
    bus --> capabilities
    bus --> resilience
    bus --> runtime
    bus --> utils
    introspection --> cognition
    introspection --> epistemics
    introspection --> health
    introspection --> language
    introspection --> pipeline
    introspection --> resilience
    introspection --> runtime
    introspection --> security
    introspection --> self
    introspection --> utils
    introspection --> voice
    morality --> autonomy
    morality --> brain
    morality --> consciousness
    morality --> conversation
    morality --> interiority
    morality --> perception
    morality --> runtime
    morality --> utils
    morality --> verify
    tasks --> runtime
    canonical --> runtime
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
    evaluation --> brain
    evaluation --> conversation
    evaluation --> learning
    evaluation --> promotion
    evaluation --> runtime
    evaluation --> verify
    interiority --> canonical
    interiority --> consciousness
    interiority --> fsw
    interiority --> runtime
    interiority --> verify
    motivation --> brain
    motivation --> consciousness
    motivation --> health
    motivation --> runtime
    motivation --> utils
    motivation --> values
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
    cognitive --> brain
    cognitive --> conversation
    cognitive --> executive
    cognitive --> governance
    cognitive --> health
    cognitive --> perception
    cognitive --> phases
    cognitive --> runtime
    cognitive --> utils
    discovery --> cognition
    discovery --> memory
    discovery --> observability
    discovery --> runtime
    discovery --> self_modification
    discovery --> unknowns
    ethics --> brain
    ethics --> morality
    ethics --> runtime
    ethics --> utils
    ethics --> verify
    kernel --> agency
    kernel --> brain
    kernel --> cognition
    kernel --> consciousness
    kernel --> continuity
    kernel --> conversation
    kernel --> cybernetics
    kernel --> executive
    kernel --> goals
    kernel --> health
    kernel --> introspection
    kernel --> learning
    kernel --> memory
    kernel --> ops
    kernel --> perception
    kernel --> phases
    kernel --> pipeline
    kernel --> resilience
    kernel --> runtime
    kernel --> security
    kernel --> self
    kernel --> self_modification
    kernel --> senses
    kernel --> somatic
    kernel --> state
    kernel --> utils
    ops --> brain
    ops --> conversation
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
    pipeline --> observability
    pipeline --> runtime
    pipeline --> verify
    planning --> brain
    planning --> capabilities
    planning --> collective
    planning --> conversation
    planning --> data
    planning --> runtime
    planning --> utils
    planning --> verify
    world --> governance
    world --> knowledge
    world --> runtime
    world --> utils
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
    llm --> brain
    managers --> autonomic
    managers --> brain
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
    unity --> agency
    unity --> cognition
    unity --> consciousness
    unity --> ghost
    unity --> runtime
    unity --> social
    unity --> values
    advanced_cognition --> conversation
    advanced_cognition --> environment
    advanced_cognition --> reasoning
    advanced_cognition --> runtime
    collective --> adaptation
    collective --> agency
    collective --> brain
    collective --> conversation
    collective --> planning
    collective --> runtime
    collective --> security
    collective --> utils
    data --> runtime
    dialogue --> language
    engineering --> metacognition
    engineering --> runtime
    environment --> advanced_cognition
    environment --> brain
    environment --> consciousness
    environment --> conversation
    environment --> environments
    environment --> executive
    environment --> memory
    environment --> perception
    environment --> runtime
    environment --> utils
    expression --> affect
    expression --> soma
    media --> conversation
    media --> runtime
    meta --> adaptation
    meta --> runtime
    meta --> utils
    morphogenesis --> adaptation
    morphogenesis --> conversation
    morphogenesis --> fsw
    morphogenesis --> memory
    morphogenesis --> resilience
    morphogenesis --> runtime
    morphogenesis --> verify
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
    together --> brain
    together --> conation
    together --> conversation
    together --> runtime
    together --> self
    together --> utils
    adapters --> agency
    adapters --> brain
    adapters --> runtime
    adapters --> utils
    conversational --> conversation
    conversational --> memory
    conversational --> runtime
    conversational --> social
    db --> runtime
    evidence --> cognition
    evidence --> knowledge
    evidence --> runtime
    maintenance --> resilience
    maintenance --> runtime
    persistence --> observability
    persistence --> resilience
    persistence --> runtime
    phenomenal_substrate --> runtime
    pneuma --> affect
    pneuma --> runtime
    pneuma --> utils
    search --> capabilities
    search --> conversation
    search --> knowledge
    search --> memory
    search --> runtime
    search --> security
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
    connectome --> cognition
    connectome --> consciousness
    connectome --> fsw
    connectome --> runtime
    connectome --> science
    connectome --> verify
    construction --> language
    construction --> runtime
    evals --> consciousness
    evals --> epistemics
    evals --> runtime
    evals --> security
    evals --> self_modification
    evals --> verify
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
    plasticity --> runtime
    predictive --> brain
    predictive --> runtime
    predictive --> utils
    sensors --> runtime
    sensors --> world
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
    networking --> utils
    rebuilding --> knowledge
    rebuilding --> language
    rebuilding --> runtime
    rebuilding --> sandbox
    rebuilding --> self_modification
    research_core --> curriculum
    research_core --> discovery
    research_core --> lattice
    research_core --> promotion
    research_core --> runtime
    research_core --> unknowns
    research_core --> verification
    safety --> runtime
    science --> consciousness
    science --> evaluation
    science --> runtime
    science --> world
    services --> autonomic
    session --> runtime
    session --> utils
    sovereignty --> ethics
    sovereignty --> governance
    sovereignty --> identity
    sovereignty --> organism
    sovereignty --> runtime
    sovereignty --> utils
    systems --> runtime
    transparency --> conversation
    transparency --> runtime
    worlds --> embodiment
    worlds --> learning
    worlds --> runtime
    audits --> brain
    audits --> runtime
    control --> conversation
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
    core_root --> conation
    core_root --> consciousness
    core_root --> continuity
    core_root --> conversation
    core_root --> data
    core_root --> evaluation
    core_root --> executive
    core_root --> fictional
    core_root --> fsw
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
    core_root --> social
    core_root --> soma
    core_root --> sovereign
    core_root --> startup
    core_root --> state
    core_root --> supervisor
    core_root --> transparency
    core_root --> utils
    core_root --> voice
    core_root --> workspace
    core_root --> world_model
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
    subject --> affect
    subject --> agency
    subject --> brain
    subject --> conation
    subject --> consciousness
    subject --> identity
    subject --> interiority
    subject --> kernel
    subject --> memory
    subject --> motivation
    subject --> ontogeny
    subject --> perception
    subject --> phases
    subject --> runtime
    subject --> security
    subject --> social
    subject --> soma
    subject --> state
    subject --> verify
    subject --> world_model
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
| brain | 476 | 316671 | 13327295 | 70 | 62 |
| learning | 366 | 186834 | 7582852 | 32 | 13 |
| runtime | 295 | 113423 | 4341876 | 63 | 151 |
| consciousness | 188 | 87403 | 3748805 | 52 | 38 |
| agency | 239 | 74388 | 3458674 | 51 | 30 |
| skills | 164 | 69385 | 3095470 | 54 | 21 |
| cognition | 205 | 67370 | 2797031 | 32 | 29 |
| perception | 212 | 60423 | 2753633 | 21 | 26 |
| conversation | 82 | 40313 | 1596492 | 39 | 49 |
| memory | 121 | 36991 | 1493695 | 26 | 45 |
| core_root | 53 | 33662 | 1471915 | 77 | 0 |
| phases | 44 | 32993 | 1501007 | 51 | 9 |
| reality_reach | 33 | 31032 | 1277348 | 13 | 4 |
| subject | 69 | 28422 | 1241491 | 24 | 0 |
| orchestrator | 44 | 23860 | 1049507 | 107 | 11 |
| capabilities | 29 | 19746 | 827081 | 19 | 11 |
| resilience | 64 | 19542 | 809814 | 22 | 33 |
| connectome | 44 | 19211 | 728354 | 7 | 2 |
| self_modification | 53 | 18475 | 760045 | 23 | 18 |
| adaptation | 30 | 17486 | 731332 | 24 | 17 |
| interiority | 72 | 16765 | 662995 | 7 | 7 |
| engineering | 36 | 15732 | 614456 | 5 | 4 |
| autonomy | 32 | 15247 | 635605 | 34 | 11 |
| voice | 40 | 14758 | 606455 | 14 | 11 |
| social | 49 | 14396 | 562210 | 20 | 16 |
| security | 38 | 14088 | 555359 | 19 | 36 |
| embodiment | 29 | 13973 | 559258 | 14 | 7 |
| self | 38 | 12658 | 503413 | 27 | 17 |
| verify | 37 | 12098 | 456133 | 16 | 38 |
| morphogenesis | 28 | 12097 | 496513 | 10 | 4 |
| reasoning | 32 | 11851 | 494870 | 11 | 11 |
| organism | 23 | 11665 | 577568 | 44 | 12 |
| senses | 33 | 10492 | 441293 | 24 | 18 |
| cognitive | 12 | 9822 | 400129 | 14 | 6 |
| affect | 36 | 9767 | 399688 | 19 | 22 |
| environment | 78 | 9633 | 379572 | 13 | 4 |
| state | 22 | 9373 | 386474 | 16 | 19 |
| self_improvement | 22 | 9362 | 382478 | 11 | 4 |
| utils | 51 | 8948 | 353490 | 17 | 79 |
| ontogeny | 20 | 8327 | 344792 | 7 | 10 |
| being | 25 | 8089 | 332934 | 14 | 13 |
| epistemics | 21 | 7893 | 334921 | 12 | 17 |
| knowledge | 28 | 7663 | 303584 | 10 | 21 |
| executive | 17 | 7600 | 307615 | 22 | 20 |
| conation | 20 | 7452 | 316635 | 8 | 11 |
| kernel | 13 | 7311 | 305889 | 30 | 6 |
| governance | 17 | 7266 | 289190 | 18 | 32 |
| architect | 25 | 7247 | 304848 | 7 | 2 |
| language | 35 | 7208 | 299047 | 7 | 16 |
| evaluation | 25 | 6083 | 237954 | 7 | 7 |
| ops | 17 | 5805 | 233190 | 18 | 6 |
| observability | 19 | 5556 | 205469 | 11 | 25 |
| rebuilding | 22 | 5425 | 288783 | 8 | 1 |
| advanced_cognition | 13 | 5412 | 227686 | 6 | 4 |
| coordinators | 11 | 5229 | 239757 | 36 | 5 |
| actuators | 11 | 5205 | 220741 | 14 | 6 |
| world_model | 14 | 5142 | 219884 | 17 | 17 |
| goals | 15 | 5036 | 210421 | 12 | 9 |
| introspection | 11 | 4904 | 204976 | 17 | 8 |
| identity | 22 | 4856 | 197664 | 12 | 19 |
| science | 19 | 4796 | 184197 | 6 | 1 |
| planning | 9 | 4580 | 189199 | 11 | 6 |
| bus | 7 | 4214 | 173798 | 7 | 8 |
| autonomic | 6 | 3733 | 166316 | 8 | 6 |
| fsw | 9 | 3580 | 128759 | 9 | 13 |
| agi | 7 | 3137 | 128644 | 13 | 5 |
| soma | 12 | 3124 | 123676 | 9 | 13 |
| worlds | 8 | 3046 | 129854 | 5 | 1 |
| somatic | 6 | 2961 | 116749 | 13 | 5 |
| intent | 12 | 2958 | 127043 | 9 | 10 |
| unity | 11 | 2870 | 121541 | 9 | 5 |
| context | 7 | 2543 | 100954 | 1 | 1 |
| fictional | 9 | 2534 | 110155 | 10 | 1 |
| conversational | 4 | 2437 | 101613 | 6 | 3 |
| values | 16 | 2406 | 97462 | 9 | 11 |
| adapters | 8 | 2403 | 95190 | 7 | 3 |
| communication | 5 | 2258 | 86376 | 6 | 1 |
| collective | 6 | 2245 | 93409 | 9 | 4 |
| sandbox | 7 | 2244 | 86824 | 2 | 12 |
| discovery | 7 | 2185 | 93561 | 8 | 6 |
| evolution | 8 | 2161 | 86980 | 8 | 2 |
| ghost | 6 | 2145 | 87707 | 6 | 2 |
| search | 4 | 2139 | 83458 | 11 | 3 |
| sovereignty | 4 | 2098 | 89301 | 11 | 1 |
| health | 7 | 2095 | 80302 | 6 | 27 |
| construction | 7 | 2069 | 77067 | 2 | 2 |
| skill_management | 4 | 2042 | 81349 | 7 | 2 |
| architecture_quality | 7 | 2019 | 79929 | 0 | 2 |
| evidence | 7 | 1983 | 74416 | 3 | 3 |
| tools | 11 | 1956 | 75400 | 9 | 0 |
| phenomenology | 9 | 1885 | 71690 | 0 | 0 |
| motivation | 10 | 1880 | 76220 | 11 | 7 |
| body | 22 | 1820 | 66900 | 6 | 1 |
| morality | 18 | 1806 | 70340 | 12 | 8 |
| providers | 6 | 1560 | 69509 | 44 | 0 |
| world | 24 | 1560 | 57107 | 7 | 6 |
| together | 7 | 1471 | 65682 | 7 | 4 |
| cybernetics | 6 | 1314 | 52522 | 6 | 1 |
| meta | 7 | 1307 | 49391 | 5 | 4 |
| pneuma | 7 | 1279 | 48424 | 3 | 3 |
| sleep | 10 | 1275 | 55009 | 12 | 5 |
| grounding | 8 | 1265 | 47385 | 6 | 2 |
| promotion | 7 | 1248 | 43277 | 1 | 4 |
| workspace | 9 | 1243 | 45349 | 3 | 3 |
| evals | 3 | 1233 | 43544 | 7 | 2 |
| actuation | 9 | 1164 | 44639 | 3 | 1 |
| phenomenal_substrate | 11 | 1160 | 46208 | 1 | 3 |
| ethics | 4 | 1150 | 47201 | 8 | 6 |
| diagnosis | 5 | 1149 | 44746 | 3 | 1 |
| media | 4 | 1100 | 41153 | 2 | 4 |
| metacognition | 3 | 1048 | 40187 | 3 | 2 |
| creativity | 3 | 1040 | 42558 | 3 | 1 |
| audit | 7 | 1028 | 42030 | 4 | 1 |
| persistence | 4 | 1022 | 41382 | 3 | 3 |
| supervisor | 3 | 1017 | 39943 | 3 | 5 |
| dialogue | 4 | 978 | 38792 | 1 | 4 |
| managers | 6 | 975 | 41548 | 24 | 5 |
| expression | 5 | 906 | 34519 | 2 | 4 |
| db | 4 | 818 | 32123 | 2 | 3 |
| pipeline | 3 | 816 | 30422 | 3 | 6 |
| guardians | 5 | 807 | 33732 | 7 | 1 |
| factory | 8 | 770 | 30011 | 3 | 1 |
| quantum | 5 | 767 | 30044 | 0 | 1 |
| council | 6 | 763 | 28974 | 5 | 0 |
| environments | 7 | 752 | 31315 | 3 | 1 |
| lattice | 5 | 704 | 26089 | 0 | 2 |
| resource | 4 | 702 | 27247 | 4 | 5 |
| canonical | 3 | 684 | 27948 | 1 | 7 |
| curriculum | 7 | 658 | 22058 | 1 | 1 |
| data | 3 | 652 | 22460 | 2 | 4 |
| session | 3 | 642 | 25702 | 2 | 1 |
| research_core | 5 | 636 | 24662 | 8 | 1 |
| safety | 3 | 631 | 25872 | 3 | 1 |
| stand_ins | 4 | 629 | 31674 | 0 | 0 |
| control | 2 | 585 | 21148 | 5 | 0 |
| tasks | 4 | 567 | 20090 | 4 | 8 |
| startup | 4 | 549 | 19919 | 12 | 2 |
| sovereign | 4 | 524 | 18770 | 3 | 2 |
| reproducibility | 2 | 498 | 18228 | 1 | 0 |
| lab | 7 | 482 | 19414 | 3 | 0 |
| mission | 4 | 472 | 17816 | 1 | 0 |
| sim | 2 | 452 | 17698 | 5 | 2 |
| plasticity | 5 | 428 | 15405 | 2 | 2 |
| coherence | 2 | 413 | 19975 | 6 | 2 |
| simulation | 3 | 413 | 16634 | 7 | 2 |
| transparency | 2 | 412 | 16483 | 2 | 1 |
| swarm | 4 | 366 | 14494 | 6 | 0 |
| verification | 4 | 350 | 13177 | 2 | 3 |
| networking | 1 | 329 | 12286 | 3 | 1 |
| forge | 8 | 326 | 11893 | 2 | 0 |
| unknowns | 4 | 325 | 11829 | 3 | 2 |
| audits | 2 | 316 | 11976 | 3 | 0 |
| neuroweb | 4 | 313 | 12315 | 5 | 0 |
| maintenance | 2 | 295 | 10758 | 3 | 3 |
| middleware | 1 | 266 | 11787 | 2 | 1 |
| play | 1 | 260 | 10180 | 3 | 0 |
| llm | 3 | 259 | 9853 | 1 | 5 |
| systems | 3 | 254 | 9963 | 2 | 1 |
| sensors | 1 | 245 | 11168 | 2 | 2 |
| welfare | 7 | 228 | 8034 | 0 | 1 |
| predictive | 2 | 186 | 7111 | 4 | 2 |
| multimodal | 2 | 185 | 6601 | 1 | 0 |
| ontology | 2 | 169 | 5381 | 0 | 0 |
| consent | 2 | 167 | 5514 | 0 | 1 |
| twins | 1 | 97 | 3626 | 0 | 0 |
| initializers | 2 | 61 | 2547 | 2 | 1 |
| latent | 1 | 56 | 2337 | 0 | 0 |
| continuity | 1 | 33 | 1147 | 1 | 11 |
| services | 2 | 31 | 1171 | 1 | 1 |

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

- Unique services retrieved: 392
- Unique services registered: 277
- Services retrieved without detected registration: 238

### Top Fetched Services

| Service | Gets | Registrations |
| --- | ---: | ---: |
| llm_router | 48 | 2 |
| capability_engine | 45 | 2 |
| affect_engine | 35 | 1 |
| cognitive_engine | 32 | 2 |
| memory_facade | 28 | 1 |
| conscious_substrate | 28 | 1 |
| global_workspace | 26 | 1 |
| liquid_substrate | 22 | 2 |
| mycelial_network | 21 | 1 |
| state_repository | 20 | 2 |
| free_energy_engine | 20 | 0 |
| goal_engine | 19 | 0 |
| drive_engine | 19 | 1 |
| world_state | 19 | 0 |
| belief_revision_engine | 17 | 2 |
| homeostasis | 17 | 1 |
| knowledge_graph | 17 | 0 |
| episodic_memory | 17 | 1 |
| aura_kernel | 16 | 1 |
| resilience_engine | 14 | 1 |

### Missing Registration Candidates

- `actuator_registry` fetched 1 time(s)
- `adaptive_immune_system` fetched 3 time(s)
- `affect` fetched 2 time(s)
- `affect_engine_v2` fetched 2 time(s)
- `affect_grounding` fetched 1 time(s)
- `affect_module` fetched 2 time(s)
- `affective_steering` fetched 1 time(s)
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
- `aura_runtime` fetched 1 time(s)
- `aura_state` fetched 3 time(s)
- `autonomous_resilience_mesh` fetched 1 time(s)
- `autopoiesis` fetched 1 time(s)
- `backup_manager` fetched 1 time(s)
- `backup_system` fetched 1 time(s)
- `being_runtime` fetched 4 time(s)
- `belief_challenger` fetched 1 time(s)
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
- `canonical_self_engine` fetched 3 time(s)
- `capability_map` fetched 1 time(s)
- `causal_world_model` fetched 5 time(s)
- `cel_bridge` fetched 1 time(s)
- `cellular_substrate` fetched 1 time(s)
- `clipboard_manager` fetched 2 time(s)
- `cloud_body` fetched 1 time(s)
- `cognitive_kernel` fetched 1 time(s)
- `cold_store` fetched 1 time(s)
- `concept_bridge` fetched 2 time(s)
- `concept_linker` fetched 1 time(s)
- `config` fetched 1 time(s)
- `consciousness_bridge` fetched 2 time(s)
- `consciousness_evidence` fetched 1 time(s)

## Operational Authority Map

| Surface | Calls | Files | Owner Calls | Review Candidates |
| --- | ---: | ---: | ---: | ---: |
| UnifiedWill decisions | 60 | 28 | 2 | 58 |
| Memory writes | 402 | 167 | 54 | 348 |
| State mutation | 996 | 380 | 12 | 984 |
| Tool execution | 131 | 68 | 7 | 124 |
| Self-modification and patching | 17 | 14 | 1 | 16 |
| LLM inference | 251 | 161 | 56 | 195 |
| External I/O | 299 | 72 | 17 | 282 |

### UnifiedWill decisions

Calls that can ask the single will authority to approve action.

Review candidates:
- `core/actuators/actuator_synthesis.py:259` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:259` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:524` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:524` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:630` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:630` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:328` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:328` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/autonomy/genuine_refusal.py:385` [autonomy] `will.decide` - decision = will.decide(content, source="genuine_refusal", domain=domain, priority=0.8, context=ctx)
- `core/autonomy/self_modification.py:514` [autonomy] `will.decide` - decision = will.decide(
- `core/brain/personality_engine.py:639` [brain] `get_will` - decision = get_will().decide(
- `core/brain/personality_engine.py:639` [brain] `get_will.decide` - decision = get_will().decide(
- `core/brain/verifier_curriculum.py:172` [brain] `get_will` - decision = get_will().decide(
- `core/brain/verifier_curriculum.py:172` [brain] `get_will.decide` - decision = get_will().decide(
- `core/cognitive/autopoiesis.py:971` [cognitive] `will.decide` - decision = will.decide(
- `core/consciousness/parallel_branches.py:736` [consciousness] `will.decide` - decision = will.decide(
- `core/consciousness/perturbational_probe.py:261` [consciousness] `get_will` - decision = get_will().decide(
- `core/consciousness/perturbational_probe.py:261` [consciousness] `get_will.decide` - decision = get_will().decide(
- `core/environment/governance_bridge.py:47` [environment] `self.will_gateway.decide` - will_decision = await self.will_gateway.decide(intent)
- `core/governance/will_gate.py:109` [governance] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:160` [governance] `will.decide` - decision = will.decide(
- `core/learning/compounding_scheduler.py:327` [learning] `get_will` - decision = get_will().decide(
- `core/learning/compounding_scheduler.py:327` [learning] `get_will.decide` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:657` [learning] `get_will` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:657` [learning] `get_will.decide` - decision = get_will().decide(

### Memory writes

Calls that can create durable or semantically promoted memory.

Review candidates:
- `core/actuators/doc_ingest.py:199` [actuators] `memory_facade.add_memory` - result = memory_facade.add_memory(text=text, metadata=metadata)
- `core/adaptation/abstraction_engine.py:196` [adaptation] `MemoryWriteReceipt` - MemoryWriteReceipt(
- `core/adaptation/abstraction_engine.py:215` [adaptation] `memory_facade.store` - await memory_facade.store(
- `core/adaptation/adaptive_immunity.py:1870` [adaptation] `self._cells.append` - self._cells.append(memory)
- `core/advanced_cognition/continual_learning_stability.py:236` [advanced_cognition] `self._persist_memory` - self._persist_memory(existing)
- `core/advanced_cognition/continual_learning_stability.py:251` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:255` [advanced_cognition] `self._persist_memory` - self._persist_memory(other)
- `core/advanced_cognition/continual_learning_stability.py:284` [advanced_cognition] `self.store_memory` - return self.store_memory(
- `core/advanced_cognition/continual_learning_stability.py:452` [advanced_cognition] `scored.append` - scored.append((score, memory))
- `core/advanced_cognition/continual_learning_stability.py:694` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:721` [advanced_cognition] `self._append_jsonl` - self._append_jsonl(self.state_dir / "memory.jsonl", rec.to_dict())
- `core/affect/phenomenal_integration.py:647` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/ambient_life_director.py:249` [agency] `candidates.append` - candidates.append(_clamp(pressure(Resource.MEMORY)))
- `core/agency/autonomous_task_engine.py:1433` [agency] `self._mycelial.add_edge` - await self._mycelial.add_edge(context["source_memory"], goal[:40])
- `core/agency/latent_distiller.py:60` [agency] `self.memory.store_memory` - await self.memory.store_memory(
- `core/architect/code_graph.py:721` [architect] `effects.add` - effects.add("memory_write")
- `core/architect/safe_boot_harness.py:79` [architect] `probe_memory_write_read` - memory = await probe_memory_write_read(tmp_root=root / "memory")
- `core/architect/smell_detector.py:176` [architect] `self._effect_smell` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/architect/smell_detector.py:176` [architect] `smells.append` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/autonomy/autonomous_research_orchestrator.py:205` [autonomy] `MemoryPersister` - self._persister = persister or MemoryPersister()
- `core/autonomy/initiative_overflow.py:156` [autonomy] `logger.debug` - logger.debug("Skill gap memory write failed: %s", exc)
- `core/autonomy/initiative_overflow.py:166` [autonomy] `memory.store_sync` - memory.store_sync(
- `core/autonomy/personhood_engine.py:200` [autonomy] `state.cognition.working_memory.append` - state.cognition.working_memory.append(
- `core/autonomy/research_cycle.py:1108` [autonomy] `state.cognition.long_term_memory.append` - state.cognition.long_term_memory.append(
- `core/autonomy/research_cycle.py:1126` [autonomy] `hasattr` - if memory_facade is not None and hasattr(memory_facade, "add_memory"):

### State mutation

Calls that can mutate runtime, identity, repository, or persistent state.

Review candidates:
- `core/actuation/cloud_actuator.py:49` [actuation] `frozenset` - KNOWN_INFRA_STATES = frozenset({
- `core/actuation/robotics_actuator.py:85` [actuation] `snapshot.setdefault` - snapshot.setdefault("status", payload.get("status"))
- `core/actuators/actuator_registry.py:1386` [actuators] `set` - forged = sorted(set(dict(params or {})) & set(_REGISTRY_OWNED_PARAM_KEYS))
- `core/adaptation/adaptive_immunity.py:1411` [adaptation] `SingleSlotStateWriter` - self._state_writer = SingleSlotStateWriter(
- `core/adaptation/adaptive_immunity.py:1437` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:1439` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:1906` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:1920` [adaptation] `self._state_writer.stats` - writer_stats = self._state_writer.stats()
- `core/adaptation/adaptive_immunity.py:2111` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2218` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:3068` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/autonomous_resilience.py:368` [adaptation] `set` - registered_names = set(registry.keys())
- `core/adaptation/dream_journal.py:288` [adaptation] `identity_ledger.commitments.all` - for c in identity_ledger.commitments.all()[-10:]
- `core/adaptation/immune_state_writer.py:374` [adaptation] `self._state_writer.submit` - self._state_writer.submit(payload, background=not here)
- `core/adaptation/immune_state_writer.py:392` [adaptation] `self._state_writer.flush` - return self._state_writer.flush(timeout)
- `core/adaptation/immune_state_writer.py:403` [adaptation] `self._state_writer.stop` - self._state_writer.stop(timeout=STATE_SHUTDOWN_TIMEOUT_S)
- `core/adaptation/value_autopoiesis.py:185` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:284` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:362` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:598` [adaptation] `os.replace` - os.replace(tmp_path, _STATE_PATH)
- `core/advanced_cognition/integration.py:263` [advanced_cognition] `next_state.setdefault` - next_state.setdefault("_advanced_prediction", {})[act.action_id] = pred
- `core/advanced_cognition/integration.py:362` [advanced_cognition] `issubset` - if isinstance(value, Mapping) and {"domain", "state"}.issubset(value.keys()):
- `core/advanced_cognition/ontology_invention.py:159` [advanced_cognition] `_write_ontology_state` - _write_ontology_state(
- `core/advanced_cognition/ontology_invention.py:240` [advanced_cognition] `_write_ontology_state` - _write_ontology_state(
- `core/advanced_cognition/ontology_invention.py:505` [advanced_cognition] `_write_ontology_state` - _write_ontology_state(

### Tool execution

Calls that can execute tools, skills, shells, browsers, or external actions.

Review candidates:
- `core/actuators/actuator_registry.py:870` [actuators] `self.operator.execute_synthesized_tool` - res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/code_execution_actuator.py:123` [actuators] `operator.execute_synthesized_tool` - res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/web_actuators.py:171` [actuators] `skill.safe_execute` - return await skill.safe_execute({"mode": "browse", "url": validated_url}, skill_context)
- `core/agency/agency_orchestrator.py:370` [agency] `execute` - await execute(proposal, state_snapshot, receipt.capability_token or "")
- `core/agency/autonomous_task_engine.py:753` [agency] `orchestrator.execute_tool` - return await orchestrator.execute_tool(tool_name, args, **kwargs)
- `core/agency/autonomous_task_engine.py:2823` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2833` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2856` [agency] `orch.execute_tool` - result = await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2867` [agency] `orch.execute_tool` - result = await orch.execute_tool("run_python", {"code": code})
- `core/agency/autonomous_task_engine.py:2884` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/desktop_planner.py:57` [agency] `skill.safe_execute` - await skill.safe_execute({"action": action, **params}, {})
- `core/agency/macro_skill.py:176` [agency] `library.execute_skill` - results = await library.execute_skill(self.macro_name, dict(params or {}))
- `core/agency/skill_library.py:261` [agency] `tool_orchestrator.execute_tool` - result = await tool_orchestrator.execute_tool(step.tool_name, resolved_args)
- `core/agi/curiosity_daemon.py:93` [agi] `orchestrator.execute_tool` - await orchestrator.execute_tool(
- `core/agi/curiosity_explorer.py:631` [agi] `orchestrator.execute_tool` - orchestrator.execute_tool(
- `core/autonomy/autonomous_initiative_loop.py:955` [autonomy] `capability_engine.execute` - scan_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:1003` [autonomy] `capability_engine.execute` - test_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:1044` [autonomy] `capability_engine.execute` - proposal_result = await capability_engine.execute(
- `core/autonomy/behavior_controller.py:220` [autonomy] `self.orchestrator.execute_tool` - return await self.orchestrator.execute_tool(
- `core/autonomy/behavior_controller.py:231` [autonomy] `self.orchestrator.execute_tool` - return await self.orchestrator.execute_tool(tool_name, arguments)
- `core/autonomy/behavior_controller.py:263` [autonomy] `self.execute_tool_call_async` - self.execute_tool_call_async(tool_name, arguments), target_loop
- `core/autonomy/behavior_controller.py:266` [autonomy] `self.execute_tool_call_async` - return asyncio.run(self.execute_tool_call_async(tool_name, arguments))
- `core/autonomy/behavior_controller.py:271` [autonomy] `self.execute_tool_call_async` - self.execute_tool_call_async(tool_name, arguments), target_loop
- `core/autonomy/proactive_presence.py:677` [autonomy] `tool_orch.execute_tool` - result = await tool_orch.execute_tool(
- `core/autonomy/research_cycle.py:871` [autonomy] `self.orchestrator.execute_tool` - lambda name=tool_name, **kw: self.orchestrator.execute_tool(

### Self-modification and patching

Calls that can generate, validate, apply, or promote code changes.

Review candidates:
- `core/architect/governor.py:140` [architect] `self.promotion_governor.promote` - decision = self.promotion_governor.promote(plan, shadow, proof, rollback)
- `core/brain/llm/mlx_client.py:10231` [brain] `self._activate_promoted_artifact` - await self._activate_promoted_artifact(str(_pending_promotion))
- `core/evolution/optimizer.py:57` [evolution] `patch.apply` - success = await patch.apply(signature)
- `core/evolution/optimizer.py:68` [evolution] `cog_patch.apply` - if await cog_patch.apply(signature):
- `core/factory/software_factory.py:115` [factory] `self.writer.write_patch` - patch = await self.writer.write_patch(change, repo_path)
- `core/guardians/airlock.py:82` [guardians] `async_atomic_write_text` - await async_atomic_write_text(patch_file, diff_patch, encoding="utf-8")
- `core/kernel/upgrades_10x.py:389` [kernel] `self._safe_self_modify` - await self._safe_self_modify(state)
- `core/orchestrator/mixins/boot/boot_autonomy.py:606` [orchestrator] `apply_presence_patch` - apply_presence_patch(self)
- `core/runtime/safe_mode.py:140` [runtime] `apply_orchestrator_patches` - apply_orchestrator_patches(orchestrator, safe_mode=bool(enabled))
- `core/runtime/settings_control_plane.py:403` [runtime] `validate_settings_patch` - validated = validate_settings_patch(changes)
- `core/security/immune_system.py:275` [security] `self._apply_patch` - reversible_ref = self._apply_patch(ev)
- `core/skill_management/hephaestus.py:384` [skill_management] `_apply_fix_once` - patched_code = _apply_fix_once(current_code, candidate)
- `core/skill_management/hephaestus.py:397` [skill_management] `guard.validate` - if not guard.validate(patched_code):
- `core/state/cellular_substrate.py:64` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(state, patch)
- `core/state/cellular_substrate.py:82` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(sub_target, value)
- `core/swarm/worker_pool.py:114` [swarm] `writer.write_patch` - patch_res = await writer.write_patch(task_payload.get("change", {}), task_payload.get("repo_path", "."))

### LLM inference

Calls that can spend model context or produce model-authored text/code.

Review candidates:
- `core/actuators/actuator_synthesis.py:225` [actuators] `brain.generate` - res = await brain.generate(prompt, system_prompt=system_prompt)
- `core/adaptation/abstraction_engine.py:97` [adaptation] `router.think` - text = await router.think(
- `core/adaptation/distillation_pipe.py:153` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/distillation_pipe.py:214` [adaptation] `router.think` - response = await router.think(
- `core/adaptation/dream_journal.py:165` [adaptation] `self.brain.think` - res = await self.brain.think(
- `core/adaptation/epistemic_humility.py:202` [adaptation] `llm.think` - response = await llm.think(
- `core/adaptation/heuristic_synthesizer.py:144` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/star_reasoner.py:877` [adaptation] `llm.think` - llm.think(prompt), timeout=self.RATIONALIZATION_TIMEOUT
- `core/agency/autonomous_task_engine.py:1350` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2458` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2543` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2678` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2802` [agency] `llm.think` - raw = await llm.think(
- `core/agency/cognitive_loop_pathway.py:126` [agency] `self._router.generate` - self._router.generate(
- `core/agency/her_reasoning.py:208` [agency] `router.think` - out = await router.think(
- `core/agency/latent_distiller.py:46` [agency] `brain.think` - summary = await brain.think(
- `core/agency/self_play.py:71` [agency] `router.think` - response = await router.think(prompt=request, prefer_tier="primary", is_background=True,
- `core/agency/thought_to_action.py:70` [agency] `self._resolve_router.route` - await self._resolve_router().route(source, content)
- `core/agi/curiosity_explorer.py:759` [agi] `router.think` - router.think(
- `core/agi/hierarchical_planner.py:541` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/audits/alignment_auditor.py:71` [audits] `self.brain.think` - self.brain.think(
- `core/audits/alignment_auditor.py:140` [audits] `self.brain.think` - self.brain.think(
- `core/audits/tool_auditor.py:85` [audits] `self.brain.think` - thought = await self.brain.think(
- `core/autonomy/genuine_refusal.py:576` [autonomy] `llm.think` - result = await asyncio.wait_for(llm.think(prompt, mode="FAST"), timeout=allowance)
- `core/autonomy/personhood_engine.py:188` [autonomy] `llm.think` - llm.think(f"[Spontaneous Thought Prompt] {prompt}", mode="FAST"),

### External I/O

Calls that can touch network, subprocesses, sockets, browsers, or APIs.

Review candidates:
- `core/adapters/chrome_cdp_transport.py:125` [adapters] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(url)
- `core/adapters/chrome_cdp_transport.py:127` [adapters] `CdpPolicyError` - raise CdpPolicyError(f"CDP target scheme {parsed.scheme!r} is not a websocket scheme")
- `core/adapters/chrome_cdp_transport.py:201` [adapters] `RuntimeError` - raise RuntimeError("websocket-client is required for Chrome CDP control") from exc
- `core/adapters/chrome_cdp_transport.py:216` [adapters] `websocket.create_connection` - ws = websocket.create_connection(url, timeout=budget)
- `core/adapters/chrome_cdp_transport.py:274` [adapters] `logger.debug` - logger.debug("CDP websocket close failed: %s", exc)
- `core/bus/sensory_gate.py:531` [bus] `urllib.parse.quote` - f"&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
- `core/capabilities/browser_authority.py:157` [capabilities] `str` - parsed = urllib.parse.urlparse(str(url or ""))
- `core/capabilities/browser_authority.py:157` [capabilities] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(str(url or ""))
- `core/capabilities/web_interlocutor.py:339` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(text)
- `core/capabilities/web_interlocutor.py:505` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(text)
- `core/capabilities/web_interlocutor.py:573` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(cleaned)
- `core/capabilities/web_interlocutor.py:586` [capabilities] `str` - parts = urllib.parse.urlparse(str(ws_url or ""))
- `core/capabilities/web_interlocutor.py:586` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(str(ws_url or ""))
- `core/capabilities/web_interlocutor.py:878` [capabilities] `urllib.parse.quote` - quoted = urllib.parse.quote(target_url, safe=":/?&=%#")
- `core/capabilities/web_interlocutor.py:4382` [capabilities] `str` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:4382` [capabilities] `str.strip` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:4382` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:4438` [capabilities] `urllib.parse.urlparse` - current_parts = urllib.parse.urlparse(current)
- `core/capabilities/web_interlocutor.py:4439` [capabilities] `urllib.parse.urlparse` - desired_parts = urllib.parse.urlparse(desired)
- `core/collective/swarm_protocol.py:28` [collective] `socket.gethostname` - self.node_id = socket.gethostname()
- `core/collective/swarm_protocol.py:68` [collective] `logger.warning` - logger.warning("🕸️ Mycelial Swarm running in offline-only mode; socket binding unavailable.")
- `core/collective/swarm_protocol.py:99` [collective] `logger.debug` - logger.debug("Swarm listener close timed out; abandoning socket.")
- `core/communication/messages_history.py:42` [communication] `str` - quoted = urllib.parse.quote(str(self.db_path), safe="/")
- `core/communication/messages_history.py:42` [communication] `urllib.parse.quote` - quoted = urllib.parse.quote(str(self.db_path), safe="/")
- `core/consciousness/heartbeat.py:199` [consciousness] `socket.socket` - with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:

## Degradation Handling

- Total `record_degradation()` calls: 4837
- Log-and-limp candidates: 4348
- Nearby fail-closed candidates: 489

Top limp-on files:

- `core/memory/memory_facade.py`: 30
- `core/consciousness/consciousness_bridge.py`: 29
- `core/senses/voice_engine.py`: 29
- `core/brain/llm/context_assembler_blocks.py`: 28
- `core/brain/cognitive_engine.py`: 27
- `core/reality_reach/attachments.py`: 24
- `core/resilience/memory_governor.py`: 24
- `core/runtime/runtime_hygiene.py`: 24
- `core/self_modification/safe_modification.py`: 23
- `core/autonomy/proactive_presence.py`: 22

## Non-Runtime Candidates

- `core/agency/calibration_for_a_current_body.py`
- `core/agency/facing_seen_under_its_input.py`
- `core/agency/keeping_a_trial_observable.py`
- `core/agency/navigation_before_body_experiments.py`
- `core/agency/response_in_the_body_frame.py`
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
- `core/cognition/selector_view.py`
- `core/cognition/the_experiment_that_settles_it.py`
- `core/connectome/proofreading.py`
- `core/consciousness/animal_cognition.py`
- `core/consciousness/narrative_gravity.py`
- `core/consciousness/narrative_provenance.py`
- `core/consciousness/oscillatory_binding.py`
- `core/diagnosis/experiment.py`
- `core/evals/matched_experiment.py`
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
- `core/organism/model_validation_voice.py`
- `core/perception/names_for_observed_instances.py`
- `core/perception/triangles_from_public_draws.py`
- `core/phases/response_generation_unitary_proof_turns.py`
- `core/reasoning/proof_answer_domains.py`
- `core/reasoning/proof_answer_solver.py`
- `core/reasoning/proof_answer_types.py`
- `core/reasoning/proof_kernel.py`
- `core/reproducibility/proof_substrate.py`
- `core/runtime/proof_kernel_bridge.py`
- `core/runtime/proof_policy.py`
- `core/science/experiment_registry.py`
- `core/science/neuro_reference.py`
- `core/search/research_pipeline.py`
- `core/skills/deep_research.py`
- `core/skills/desktop_research.py`
- `core/voice/duplex/pocket_speech.py`

## Consolidation Candidates

- `core/audits/`: 2 file(s), 316 line(s)
- `core/coherence/`: 2 file(s), 413 line(s)
- `core/consent/`: 2 file(s), 167 line(s)
- `core/continuity/`: 1 file(s), 33 line(s)
- `core/control/`: 2 file(s), 585 line(s)
- `core/initializers/`: 2 file(s), 61 line(s)
- `core/latent/`: 1 file(s), 56 line(s)
- `core/maintenance/`: 2 file(s), 295 line(s)
- `core/middleware/`: 1 file(s), 266 line(s)
- `core/multimodal/`: 2 file(s), 185 line(s)
- `core/networking/`: 1 file(s), 329 line(s)
- `core/ontology/`: 2 file(s), 169 line(s)
- `core/play/`: 1 file(s), 260 line(s)
- `core/predictive/`: 2 file(s), 186 line(s)
- `core/reproducibility/`: 2 file(s), 498 line(s)
- `core/sensors/`: 1 file(s), 245 line(s)
- `core/services/`: 2 file(s), 31 line(s)
- `core/sim/`: 2 file(s), 452 line(s)
- `core/transparency/`: 2 file(s), 412 line(s)
- `core/twins/`: 1 file(s), 97 line(s)
