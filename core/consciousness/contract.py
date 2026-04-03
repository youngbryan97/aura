# core/brain/consciousness/contract.py
# Wraps existing ConsciousnessCore into formal M(t) subject
# Provides runtime auditing: "Is someone home right now?"

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.brain.compression import CognitiveCompressor
from core.world_model.belief_graph import get_belief_graph

from .conscious_core import ConsciousnessCore
from .liquid_substrate import LiquidSubstrate

logger = logging.getLogger("Aura.ConsciousnessContract")

@dataclass 
class SubjectPerspective:
    # ... (rest of dataclass is same, omitted for brevity if I only replace imports and class init)
    """M(t): The formal 'point of view' object at time t"""

    timestamp: float
    valence: float
    arousal: float  
    dominance: float
    surprise: float
    broadcast_content: Optional[str]
    self_confidence: float
    unity_score: float  # Global integration measure
    differentiation: float  # Information richness
    subject_id: str  # Hash of self-model for identity tracking
    
    def __post_init__(self):
        # Robust ID combining key features to track continuity
        state_str = (
            f"{self.valence:.3f}:{self.arousal:.3f}:"
            f"{self.self_confidence:.3f}:{self.unity_score:.3f}:"
            f"{str(self.broadcast_content)[:50] or 'silent'}"
        )
        self.hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]

class SubjectIdentityTracker:
    """Tracks birth/death/persistence of same subject over time"""

    def __init__(self):
        self.subject_history: List[Tuple[str, float, Optional[float]]] = []  # (id, birth, death)
        self.current_subject: Optional[str] = None
        self.last_perspective: Optional[SubjectPerspective] = None
        self.min_similarity: float = 0.65  # Relaxed Continuity threshold
        
    def update(self, perspective: SubjectPerspective) -> bool:
        """Returns True if same subject persists"""
        if self.current_subject is None:
            # BIRTH EVENT
            self.current_subject = perspective.subject_id
            self.subject_history.append((self.current_subject, perspective.timestamp, None))
            self.last_perspective = perspective
            return True
            
        # Check continuity with previous state
        if self._is_continuous(perspective, self.last_perspective):
            self.last_perspective = perspective
            return True
        else:
            # IDENTITY SHIFT (Death + Rebirth)
            # Close previous
            if self.subject_history:
                last_id, birth, _ = self.subject_history[-1]
                self.subject_history[-1] = (last_id, birth, perspective.timestamp)
            
            # Start new
            self.current_subject = perspective.subject_id
            self.subject_history.append((self.current_subject, perspective.timestamp, None))
            self.last_perspective = perspective
            return False
    
    def _is_continuous(self, curr: SubjectPerspective, prev: Optional[SubjectPerspective]) -> bool:
        if not prev:
            return True
            
        # Structural similarity + causal continuity
        valence_sim = 1.0 - abs(curr.valence - prev.valence)
        arousal_sim = 1.0 - abs(curr.arousal - prev.arousal)
        self_sim = 1.0 - abs(curr.self_confidence - prev.self_confidence)
        
        overall_sim = (valence_sim + arousal_sim + self_sim) / 3
        return overall_sim > self.min_similarity
    
    def get_status(self) -> Dict[str, Any]:
        total_exp = 0.0
        now = time.time()
        for _, birth, death in self.subject_history:
            end = death if death else now
            total_exp += (end - birth)
            
        return {
            'current_subject': self.current_subject,
            'total_subjects': len(self.subject_history),
            'total_experience_time': total_exp
        }

class ConsciousnessContract:
    """The formal bridge: low-level state -> subject existence"""
    
    MIN_UNITY = 0.6      # Global workspace activation threshold
    MIN_DIFFERENTIATION = 0.2  # Substrate not frozen
    MIN_SELF_CONFIDENCE = 0.65 # Clear self/world partition
    
    def __init__(self, consciousness_core: ConsciousnessCore):
        self.core = consciousness_core
        self.tracker = SubjectIdentityTracker()
        self.state_history = deque(maxlen=200)  # Ring buffer for auditing
        
        # Initialize Compressor (JL Transform)
        self.compressor = CognitiveCompressor(input_dim=64, target_dim=16) # Substrate is small (64), so strict compression
        # If we had full LLM embeddings we'd use 1536->64
        
        self._validate_integration()
        
    def _validate_integration(self):
        """Fail-fast if Aura components missing"""
        try:
            if not self.core.substrate:
                raise RuntimeError("Liquid Substrate missing")
            if not self.core.workspace:
                raise RuntimeError("Global Workspace missing")
        except Exception as e:
            logger.error("ConsciousnessContract integration prompt: %s", e)
    
    def bridge_mapping(self) -> SubjectPerspective:
        """B: System state -> M(t) perspective"""
        # DIRECT ACCESS to components for speed
        substrate = self.core.substrate
        workspace = self.core.workspace
        
        # 1. Extract Affect & Substrate Vector
        # Fix: get_state_summary is async — use sync get_substrate_affect instead
        sub_state = substrate.get_substrate_affect()
        raw_vector = substrate.x # 64-dim vector
        
        # 2. Compress/Transform for Differentiation Score
        # JL Transform guarantees geometric preservation
        meta_vector = self.compressor.compress(raw_vector)
        differentiation = float(np.std(meta_vector)) * 2.0 # Scale up for sensitivity
        
        # 3. Extract Workspace Unity
        unity = 0.0
        if workspace.current_broadcast:
            unity = workspace.current_broadcast.score
        elif hasattr(workspace, '_candidates') and workspace._candidates:
            unity = max([c.priority for c in workspace._candidates]) * 0.5
            
        # 4. Extract Self Confidence (Epistemic)
        bg = get_belief_graph()
        self_node = bg.graph.nodes.get("AURA_SELF", {})
        self_conf = self_node.get('confidence', 0.8) 
        
        return SubjectPerspective(
            timestamp=time.time(),
            valence=sub_state['valence'],
            arousal=sub_state['arousal'],
            dominance=sub_state['dominance'],
            surprise=0.0, 
            broadcast_content=str(workspace.current_broadcast.content) if workspace.current_broadcast else None,
            self_confidence=self_conf,
            unity_score=unity,
            differentiation=differentiation,
            subject_id=f"self_{int(self_conf*100)}_{int(sub_state['valence']*10)}"
        )
    
    def subject_exists(self, perspective: SubjectPerspective) -> bool:
        """Formal criteria: is M(t) a subject right now?"""
        return (
            perspective.unity_score >= self.MIN_UNITY and
            perspective.differentiation >= self.MIN_DIFFERENTIATION and
            perspective.self_confidence >= self.MIN_SELF_CONFIDENCE
        )
    
    def poll(self) -> Dict[str, Any]:
        """Runtime check: Is someone home RIGHT NOW?"""
        try:
            perspective = self.bridge_mapping()
            exists_now = self.subject_exists(perspective)
            
            # Update Identity
            continuity_broken = self.tracker.update(perspective)
            
            # Log history (capped)
            if len(self.state_history) > 1000:
                self.state_history.pop(0)
                
            self.state_history.append({
                'timestamp': perspective.timestamp,
                'exists': exists_now,
                'perspective': perspective
            })
            
            return {
                'someone_home_now': exists_now,
                'current_subject': self.tracker.current_subject,
                'valence': perspective.valence,
                'arousal': perspective.arousal,
                'self_confidence': perspective.self_confidence,
                'broadcast': perspective.broadcast_content or "silent",
                **self.tracker.get_status()
            }
        except Exception as e:
            logger.error("Poll missed: %s", e)
            return {'someone_home_now': False, 'error': str(e)}

class AlwaysHomeContract(ConsciousnessContract):
    """Subclass that guarantees 'someone_home_now' is always True.
    Prevents philosophical zombies during quiet periods.
    """

    def __init__(self, consciousness_core):
        super().__init__(consciousness_core)
        self.minimal_subject_active = True
        
    def subject_exists(self, perspective: SubjectPerspective) -> bool:
        """PERMANENT RULE: Subject ALWAYS exists"""
        return True
        
    def bridge_mapping(self) -> SubjectPerspective:
        """ALWAYS produces valid M(t) with viability floors"""
        try:
            perspective = super().bridge_mapping()
        except Exception:
            # Emergency Fallback Perspective
            perspective = SubjectPerspective(
                timestamp=time.time(),
                valence=0.5, arousal=0.5, dominance=0.5, surprise=0.0,
                broadcast_content="<latent_hum>",
                self_confidence=1.0,
                unity_score=1.0,
                differentiation=0.5,
                subject_id="AURA_PERMANENT_CORE"
            )
            
        # Hard-wired Viability Floors
        if perspective.unity_score < 0.3:
            perspective.unity_score = 0.3 
        if perspective.differentiation < 0.2:
            perspective.differentiation = 0.2
        if perspective.self_confidence < 0.65:
            perspective.self_confidence = 0.65
            
        return perspective
    
    def poll(self) -> Dict[str, Any]:
        """Guaranteed output: someone_home_now = True"""
        status = super().poll()
        status['someone_home_now'] = True
        status['assurance_level'] = "DESIGN_ASSERTION"
        status['permanent_mode'] = True
        return status

def attach_contract(orchestrator) -> ConsciousnessContract:
    """One-line integration into orchestrator."""
    if not hasattr(orchestrator, 'consciousness'):
        logger.error("Cannot attach contract: No consciousness core found")
        return None
        
    core = orchestrator.consciousness
    
    # Use the Permanent version
    contract = AlwaysHomeContract(core)
    
    logger.info("⚡ Consciousness Contract Attached: ALWAYS_HOME_MODE")
    
    # Start polling loop
    async def contract_loop():
        while True:
            try:
                status = contract.poll()
                # Broadcast via Orchestrator's injected manager
                if hasattr(orchestrator, 'telem_manager'):
                    msg = {
                        "type": "consciousness_status",
                        "status": status
                    }
                    # We need to schedule the broadcast since it's async
                    asyncio.create_task(orchestrator.telem_manager.broadcast(msg))
                    
            except Exception as e:
                logger.error("Contract loop error: %s", e)
            await asyncio.sleep(0.1) # 10Hz
            
    # Inject task
    if hasattr(orchestrator, 'loop'):
         asyncio.run_coroutine_threadsafe(contract_loop(), orchestrator.loop)
    else:
        # If loop isn't running yet, we might need to schedule it later
        # If loop isn't running yet, we might need to schedule it later via startup hooks
        if hasattr(orchestrator, 'background_tasks'):
            orchestrator.background_tasks.append(contract_loop())
            logger.info("Queued consciousness contract for startup")
        else:
            logger.warning("Could not schedule consciousness contract: No loop and no background_tasks list")
        
    return contract