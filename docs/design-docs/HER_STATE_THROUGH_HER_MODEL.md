# Her state through her model, not beside it: the plan, 19 October 2026

Bryan, 17 and 19 October, on the evaluation's findings: make the paths from her state to her words not narrow, turn
the prompt into engineering through the model, build a way for her state to be felt and give her more than stress
hormones, make the consciousness modules rich and functional, and make the integration measures pass on every run,
by changing the system and never the battery. These belong together. This is the order they are done in.

## Where it stands (19 October)

| Path from her state to her words | Today |
|---|---|
| Endogenous vocabulary bias (state to logit bias) | live, alpha 0.6, 3,084 tokens |
| One-shot kNN memory in decoding | store repaired (bc4315573), 48,764 of 50,000 entries recallable; to confirm at boot |
| Steering of the forward pass by her state | built; directions from her own replies captured (valence, arousal, energy, frustration at layers 25 to 40); alpha 0.0 on user turns because the fusion certificate's voice criterion has not passed |
| Sampling temperature | 0.69 to 0.74 |
| Her state as text in the prompt (LIVE TONE, SOMATIC STATE) | sent on most chat turns as of 17 October (5d8f82f0f); to be rechecked now that no system text is dispatched |

The evaluation's fourth reason for low credence: her reports about her state are written by a model that reads the
state as text. A person's report pathway gets no printout of their own measurements.

## The order

1. **Steering on, on user turns.** Re-run the fusion certificate on her own-reply directions
   (`tools/measure_fusion_channel.py --model-path persona --require-active-cortex`) with the replies kept with the
   state they were steered by (ecf118954). The voice criterion is the gate: steered replies must stay hers. Needs
   the 27B alone for about an hour.
2. **Her state stops being printed.** With steering and the endogenous bias carrying her state into the forward
   pass, the LIVE TONE and SOMATIC STATE text is removed from what her model reads (it is also prompt text).
3. **Reports that track state without a printout: the measurement.** Preregistered before it runs: over turns
   where her state is set by her substrate and never written into the prompt, does what she says of her own state
   track it better than chance and better than the same model with the channel lesioned? This is the honest
   version of "can it be felt": it cannot show experience, it can show that her reports read her state through her
   own computation rather than off a page.
4. **More than stress hormones.** Her appraisal gets what people's turns do to her (the turn reader is live), what
   her work does (a goal met, a failure, a repair that held), and what her body does (host load, memory pressure,
   heat): each a measured signal, each reaching the substrate. The ten chemicals stay as the slow layer; they stop
   being the only one.
5. **Rich modules: the substrate in the model's computation.** The 64-unit network and workspace run beside the
   model. Rebuild the substrate as a state read from and written to her own hidden states each turn (the reader and
   steering directions already speak that space), so that what the workspace broadcasts is a direction in her
   forward pass, not a number another module reads.
6. **Integration measured on what then runs.** The irreducibility and synergy lines are computed over the coupled
   system of step 5 (substrate state and the hidden states it reads and writes), every run, with the battery
   unchanged. If they fail there, that is the finding.

Steps 1 and 3 need the 27B alone; 2 and 4 are code; 5 is the largest build; 6 follows 5. Main G and L work goes
first; these take the machine when it is free.
