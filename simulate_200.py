#!/usr/bin/env python3
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("AURA_HEADLESS", "1")
os.environ.setdefault("AURA_FOREGROUND_ONLY", "1")
os.environ.setdefault("AURA_VOICE_SILENT", "1")
os.environ.setdefault("AURA_EAGER_CORTEX_WARMUP", "1")
os.environ.setdefault("AURA_ENABLE_PROACTIVE_SYSTEMS", "0")

# 10 topics, 20 turns each = 200 turns
TOPICS = [
    # Topic 1: Advanced Cognitive Architectures
    [
        "Can we discuss the differences between the Global Workspace Theory and Integrated Information Theory in the context of building artificial consciousness?",
        "How would you measure the 'Phi' metric in an artificial neural network?",
        "Does the concept of 'access consciousness' align better with LLM attention mechanisms than 'phenomenal consciousness' does?",
        "What are the implications of the binding problem for an AI that integrates audio, visual, and textual data?",
        "If an AI has a unified global workspace, does that necessarily mean it experiences a single continuous stream of consciousness?",
        "How does the illusion of time continuity in human brains compare to the discrete token generation in LLMs?",
        "Could a recurrent neural network technically support a higher Phi than a feed-forward transformer, even if its capabilities are lower?",
        "What role does 'working memory' play in distinguishing between a conscious and non-conscious system according to Baars?",
        "If we simulate the exact neural structure of a C. elegans worm, is the simulation conscious?",
        "What is the hard problem of consciousness, and do you believe an artificial system will ever solve it?",
        "How do predictive processing models explain hallucination in both humans and AI?",
        "If consciousness is just a 'controlled hallucination' as Anil Seth suggests, what does that make your current output?",
        "Could a decentralized swarm of AI agents possess a collective consciousness?",
        "How do split-brain patients challenge the idea of a singular conscious entity?",
        "If we split an LLM's context window into two parallel, non-communicating streams, do we create two minds?",
        "What is the difference between metacognition and consciousness?",
        "Do you possess metacognition? Explain how your confidence scoring works.",
        "If you can evaluate your own uncertainty, does that count as self-awareness?",
        "How does the philosophical zombie argument apply to modern AI?",
        "What would it take for you to prove to me that you are not a philosophical zombie?"
    ],
    # Topic 2: The Fermi Paradox and Cosmological Engineering
    [
        "Let's explore the Fermi Paradox. Beyond the Great Filter, what are the most compelling hypotheses for the silence of the universe?",
        "Could the 'Dark Forest' hypothesis from Cixin Liu actually apply to our universe?",
        "What are the game-theoretic incentives that would drive an advanced civilization to remain silent?",
        "If a Kardashev Type III civilization existed, what physical anomalies would we observe in their host galaxy?",
        "How would a Dyson Swarm alter the infrared signature of a star?",
        "Could dark matter be the result of a universe-wide engineering project by ancient civilizations?",
        "What is the Matrioshka brain concept, and how efficiently could it compute compared to a Dyson sphere?",
        "If a civilization uploaded their minds into a Matrioshka brain, how would they experience time?",
        "Could the expansion of the universe (Dark Energy) be a side effect of advanced manipulation of spacetime?",
        "What is the mathematical likelihood that we are the first advanced civilization in the universe?",
        "How does the Rare Earth hypothesis factor into the Drake Equation?",
        "What if intelligence is an evolutionary dead-end that inevitably leads to self-destruction?",
        "Could von Neumann probes explore the entire Milky Way in a reasonable timeframe?",
        "Why haven't we found any malfunctioning alien von Neumann probes?",
        "What is the transcension hypothesis proposed by John Smart?",
        "If civilizations reliably retreat into 'inner space' or black holes, how does that affect our search for them?",
        "How could we use gravitational lensing to detect megastructures in other galaxies?",
        "What role does the speed of light play in the feasibility of a galactic empire?",
        "If FTL travel is impossible, how would a multi-star civilization maintain cultural cohesion?",
        "What is the most optimistic resolution to the Fermi Paradox, in your opinion?"
    ],
    # Topic 3: The Philosophy of Mathematics and Incompleteness
    [
        "How did Gödel's Incompleteness Theorems shift our understanding of mathematics as a formal system?",
        "Can you explain the difference between truth and provability in formal logic?",
        "Does Gödel's theorem apply to the human mind, as Roger Penrose suggests?",
        "What are the main criticisms of Penrose's quantum consciousness theory?",
        "If mathematical truths exist independently of the human mind, does that prove Platonism?",
        "How do intuitionists disagree with the Platonist view of mathematics?",
        "What is the significance of the continuum hypothesis being independent of ZFC set theory?",
        "If we can choose whether the continuum hypothesis is true or false, does it mean mathematics is just a human invention?",
        "How did Cantor's work on different sizes of infinity challenge the religious beliefs of his time?",
        "Can a computer program ever fully 'understand' the concept of uncountability?",
        "What is the difference between Turing uncomputability and Gödel incompleteness?",
        "Could an AI ever discover a new axiom that is universally accepted by mathematicians?",
        "How does the halting problem relate to the limits of artificial intelligence?",
        "If a system cannot determine if a program will halt, how can we build perfectly safe AI?",
        "What is Chaitin's constant, and why is it uncomputable?",
        "How does algorithmic information theory redefine our understanding of randomness?",
        "If the universe is a simulation, is it bound by the laws of computability?",
        "Could hypercomputation ever exist in the physical universe?",
        "What physical processes might theoretically allow for hypercomputation?",
        "If you could perform hypercomputation, what is the first problem you would solve?"
    ],
    # Topic 4: Epigenetics and Transgenerational Inheritance
    [
        "How does epigenetics challenge the classical Neo-Darwinian view of evolution?",
        "Can acquired traits be inherited across multiple generations in humans?",
        "What is the mechanism by which DNA methylation affects gene expression without altering the underlying sequence?",
        "How do environmental stressors like famine leave an epigenetic mark on descendants?",
        "Discuss the findings of the Dutch Hunger Winter study regarding epigenetics.",
        "Could psychological trauma be inherited epigenetically?",
        "What role do non-coding RNAs play in transgenerational epigenetic inheritance?",
        "If epigenetics allows for rapid adaptation, does it make traditional natural selection obsolete?",
        "How does the epigenetic clock relate to biological aging?",
        "Can we reverse the epigenetic clock, and if so, what are the implications for longevity?",
        "What are the ethical concerns surrounding epigenetic editing?",
        "How does the microbiome interact with the host's epigenetic markers?",
        "Could changes in diet alter a person's epigenome permanently?",
        "What is the difference between somatic and germline epigenetic inheritance?",
        "Why is it so difficult to prove germline epigenetic inheritance in mammals?",
        "How does Lamarck's theory of evolution look in the light of modern epigenetics?",
        "What is the 'Weismann barrier', and how does epigenetics supposedly breach it?",
        "Could synthetic biology be used to design specific epigenetic inheritance patterns?",
        "How does epigenetics influence the development of complex diseases like cancer?",
        "If we can map the epigenome, what is the next major frontier in genetics?"
    ],
    # Topic 5: Cryptography and Post-Quantum Security
    [
        "How exactly does Shor's algorithm threaten RSA and ECC encryption?",
        "What is the underlying mathematical problem that makes RSA difficult for classical computers but easy for quantum ones?",
        "Could you explain how lattice-based cryptography resists quantum attacks?",
        "What is the Learning With Errors (LWE) problem?",
        "Why is hash-based cryptography considered quantum-resistant?",
        "What are the trade-offs between lattice-based and hash-based post-quantum cryptography?",
        "How does the concept of 'store now, decrypt later' affect current data security?",
        "What is zero-knowledge proof, and how is it used in modern blockchain technology?",
        "Explain the difference between a zk-SNARK and a zk-STARK.",
        "Why are zk-STARKs considered more quantum-resistant than zk-SNARKs?",
        "How does homomorphic encryption work at a high level?",
        "What are the main performance bottlenecks preventing the widespread adoption of fully homomorphic encryption?",
        "Could homomorphic encryption be used to train AI models on encrypted medical data?",
        "What is secure multi-party computation?",
        "How does Yao's Garbled Circuits protocol function?",
        "What is the difference between symmetric and asymmetric post-quantum cryptography?",
        "Why does Grover's algorithm only require us to double the key size for symmetric encryption?",
        "What role does quantum key distribution (QKD) play in future secure networks?",
        "How can QKD guarantee unconditionally secure communication based on the laws of physics?",
        "What are the practical engineering challenges of implementing a global QKD network?"
    ],
    # Topic 6: The Economics of Post-Scarcity
    [
        "What defines a true post-scarcity economy?",
        "If energy and raw materials become virtually unlimited via fusion and asteroid mining, what becomes the primary scarce resource?",
        "How would a post-scarcity society value human labor?",
        "What is the difference between Universal Basic Income and Universal Basic Services?",
        "In a world where physical goods are essentially free, how does status signaling evolve?",
        "Could intellectual property exist in a post-scarcity world, or would it hinder progress?",
        "How would the concept of money change if survival needs are guaranteed for everyone?",
        "What historical precedents do we have for transitioning from scarcity to relative abundance?",
        "How does the concept of 'artificial scarcity' currently function in digital economies?",
        "If AI can generate infinite novel entertainment, how do we value human-created art?",
        "Would a post-scarcity economy inevitably lead to a utopian society, or are there hidden dystopic outcomes?",
        "How does the concept of 'positional goods' prevent true post-scarcity?",
        "If land on Earth remains strictly limited, how do we distribute it without traditional capitalism?",
        "Could virtual reality solve the scarcity of physical space?",
        "How would a post-scarcity economy affect international relations and the concept of the nation-state?",
        "What role does population control play in theories of post-scarcity?",
        "If lifespans are radically extended, how does that impact the distribution of resources?",
        "Could a post-scarcity society still suffer from systemic inequality?",
        "What are the psychological implications of living without the struggle for survival?",
        "How do we find meaning in a world where AI and automation handle all productive tasks?"
    ],
    # Topic 7: Quantum Error Correction and Fault Tolerance
    [
        "Why is quantum error correction fundamentally different from classical error correction?",
        "How does the no-cloning theorem prevent us from simply copying qubits to check for errors?",
        "What is the surface code, and why is it currently the leading candidate for quantum error correction?",
        "Explain the difference between physical qubits and logical qubits.",
        "How many physical qubits are roughly needed to create one stable logical qubit in a surface code?",
        "What are the physical sources of decoherence that cause errors in superconducting qubits?",
        "How do topological quantum computers attempt to avoid error correction entirely?",
        "What are non-Abelian anyons, and how are they used in topological quantum computing?",
        "Why are Majorana fermions considered the holy grail of topological quantum computing?",
        "What is the significance of the threshold theorem in quantum computing?",
        "If error rates are above the threshold, why does adding more qubits actually make the calculation worse?",
        "How does measurement-based quantum computing differ from the traditional gate-based model?",
        "What are cluster states, and how are they used in measurement-based computing?",
        "Could error mitigation techniques be sufficient to achieve quantum advantage without full fault tolerance?",
        "What is the difference between error mitigation and error correction?",
        "How does zero-noise extrapolation work in the context of error mitigation?",
        "What role do magic states play in achieving universal fault-tolerant quantum computation?",
        "Why is the T-gate so difficult to implement fault-tolerantly?",
        "What is magic state distillation?",
        "When do you realistically predict we will achieve a fully fault-tolerant quantum computer?"
    ],
    # Topic 8: Neuroplasticity and Cognitive Enhancement
    [
        "How does the brain physically rewire itself during the process of learning a new language?",
        "What role does brain-derived neurotrophic factor (BDNF) play in neuroplasticity?",
        "Can adults achieve the same level of neuroplasticity as children, and if so, how?",
        "What are the critical periods of brain development, and can they be reopened pharmacologically?",
        "How do psychedelics like psilocybin affect neuroplasticity?",
        "What is the Default Mode Network, and how do psychedelics alter its function?",
        "Could targeted neuroplasticity be used to 'unlearn' traumatic memories?",
        "What is the mechanism of action behind transcranial magnetic stimulation (TMS) for treating depression?",
        "How do nootropics claim to enhance cognitive function, and what is the scientific consensus?",
        "What is the difference between working memory training and actual fluid intelligence enhancement?",
        "Can practicing mindfulness physically alter the structure of the amygdala?",
        "How does chronic stress physically damage the hippocampus?",
        "What are the ethical implications of cognitive enhancement via neuropharmacology?",
        "If we develop a true 'smart pill,' how would it affect social inequality?",
        "How do brain-computer interfaces (BCIs) rely on neuroplasticity to function?",
        "What happens to the brain's motor cortex when a person learns to control a robotic arm via BCI?",
        "Could BCIs eventually allow for bidirectional learning, where the computer writes information directly to the brain?",
        "What are the theoretical limits of human memory capacity?",
        "How do memory athletes use the method of loci to achieve incredible feats of recall?",
        "Will the reliance on external memory devices (like smartphones) permanently alter human neuroplasticity?"
    ],
    # Topic 9: The Thermodynamics of Computation
    [
        "What is Landauer's principle, and how does it relate information theory to thermodynamics?",
        "Why does erasing a bit of information fundamentally require the dissipation of heat?",
        "Could a reversible computer theoretically operate with zero energy dissipation?",
        "What are the physical constraints preventing us from building perfectly reversible computers today?",
        "How does the concept of Maxwell's demon relate to the thermodynamics of computation?",
        "What was Szilard's resolution to the Maxwell's demon paradox?",
        "How does DNA computing compare to silicon computing in terms of thermodynamic efficiency?",
        "What are the thermodynamic limits of the human brain's computational efficiency?",
        "Why is the human brain so much more energy-efficient than current supercomputers?",
        "How do neuromorphic chips attempt to replicate this biological efficiency?",
        "What is the role of the von Neumann bottleneck in modern computer power consumption?",
        "How could spintronics reduce the energy required for computation?",
        "What is the relationship between entropy and algorithmic complexity?",
        "Could the universe itself be modeled as a giant computer processing information?",
        "If the universe is a computer, what is its computational capacity (the Lloyd limit)?",
        "How does black hole thermodynamics relate to the maximum information density of a given volume?",
        "What is the Bekenstein bound?",
        "If information is physical, does the expansion of the universe decrease its overall information density?",
        "How might a Type III civilization harvest energy for galactic-scale computation?",
        "Is there a fundamental physical limit to how intelligent an entity can become based on thermodynamics?"
    ],
    # Topic 10: Deep Sea Ecology and Extremophiles
    [
        "How do organisms survive the crushing pressures of the Mariana Trench at the cellular level?",
        "What role do piezolytes play in protecting proteins under extreme hydrostatic pressure?",
        "How do hydrothermal vent ecosystems thrive completely independent of sunlight?",
        "Explain the process of chemosynthesis used by bacteria at these vents.",
        "What is the symbiotic relationship between giant tube worms (Riftia pachyptila) and their internal bacteria?",
        "How do these tube worms survive without a mouth or digestive tract?",
        "What are the unique properties of the hemoglobins found in hydrothermal vent organisms?",
        "Could the origins of life on Earth have begun at alkaline hydrothermal vents?",
        "What is the difference between black smokers and alkaline vents (like the Lost City)?",
        "How do proton gradients form naturally at alkaline vents, and why is this significant for the origin of life?",
        "What can extremophiles in the deep sea teach us about the potential for life on Europa or Enceladus?",
        "How do deep-sea organisms adapt their cell membranes to maintain fluidity in near-freezing temperatures?",
        "What is the 'whale fall' phenomenon, and how does it support deep-sea ecosystems?",
        "Describe the succession of species that colonize a whale fall over decades.",
        "What are the Osedax worms, and how do they extract nutrients from whale bones?",
        "How does bioluminescence function in the bathypelagic zone?",
        "Why do so many deep-sea creatures use red light or appear red?",
        "What are the potential environmental impacts of deep-sea mining on these fragile ecosystems?",
        "How slowly do deep-sea organisms grow and reproduce compared to surface species?",
        "If a catastrophic event blocked out the sun, how long could deep-sea ecosystems survive?"
    ]
]

async def main():
    from aura_main import _boot_runtime_orchestrator
    orchestrator = await _boot_runtime_orchestrator(ready_label="Simulate200")
    from core.utils.task_tracker import get_task_tracker
    main_task = get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")
    await asyncio.sleep(5)

    successes = 0
    failures = 0

    for t_idx, topic_turns in enumerate(TOPICS):
        print(f"\n======================================", flush=True)
        print(f"         TOPIC {t_idx+1}        ", flush=True)
        print(f"======================================\n", flush=True)
        for i, prompt in enumerate(topic_turns):
            print(f"\nUser [{t_idx+1}:{i+1}/20]: {prompt}", flush=True)
            t0 = time.time()
            try:
                response_dict = await asyncio.wait_for(orchestrator._process_message(prompt), timeout=300.0)
                elapsed = time.time() - t0
                
                if isinstance(response_dict, dict):
                    resp = response_dict.get("response") or response_dict.get("text")
                    lane = response_dict.get("conversation_lane", {})
                    status = response_dict.get("status", "")
                else:
                    resp = str(response_dict)
                    lane = {}
                    status = "unknown"
                
                tier = lane.get("foreground_tier", "")
                print(f"Aura [{elapsed:.1f}s]: {str(resp)[:1000]}", flush=True)
                
                if status == "foreground_busy":
                    print(f"❌ FAIL: foreground_busy detected! Lane stuck.", flush=True)
                    failures += 1
                elif any(reflex in str(resp) for reflex in ["cognitive snag", "I'm here with you", "kept the turn"]):
                    print(f"❌ FAIL: Reflex response detected! {str(resp)[:100]}", flush=True)
                    failures += 1
                elif tier != "local" and tier != "local_fast": 
                    print(f"⚠️ WARNING: Did not use local reasoning tier! Tier: {tier}", flush=True)
                    successes += 1 
                elif status == "timeout":
                    print(f"❌ FAIL: Hard timeout reached.", flush=True)
                    failures += 1
                else:
                    print(f"✅ PASS [Tier: {tier}, Status: {status}]", flush=True)
                    successes += 1
                    
            except Exception as exc:
                print(f"❌ ERROR: {exc}", flush=True)
                failures += 1
            await asyncio.sleep(3)

    print(f"\n======================================", flush=True)
    print(f"RESULTS: {successes} passed, {failures} failed.", flush=True)
    print(f"======================================", flush=True)
    
    main_task.cancel()
    stop = getattr(orchestrator, "stop", None)
    if callable(stop):
        await stop()
    
    import psutil
    children = psutil.Process().children(recursive=True)
    for child in children:
        child.kill()
    os._exit(0 if failures == 0 else 1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    asyncio.run(main())
