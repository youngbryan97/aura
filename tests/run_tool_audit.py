################################################################################


import asyncio
import logging
import sys
import os

sys.path.append(os.getcwd())

from core.container import get_container
from core.audits.tool_auditor import ToolAuditor
from core.brain.cognitive_engine import CognitiveEngine

logging.basicConfig(level=logging.INFO)

async def main():
    print("🚀 Initializing Aura Tool Audit...")
    
    # Setup
    brain = CognitiveEngine()
    
    # Setup
    brain = CognitiveEngine()
    
    # Use Local Client (Ollama) since OpenAI Key might be missing in test env
    try:
        from core.brain.llm.ollama_client import RobustOllamaClient
        brain.client = RobustOllamaClient()
        print("✓ Ollama Client Injected (Local)")
    except Exception as e:
        print(f"⚠️ Failed to inject Ollama Client: {e}")
        # Fallback to OpenAI if Ollama fails, just in case
        try:
             from core.brain.llm.openai_client import OpenAIClient
             brain.client = OpenAIClient()
             print("✓ OpenAI Client Injected (Fallback)")
        except Exception:
             pass

    auditor = ToolAuditor(brain)
    
    print("\n🔍 Running Tool Selection Suite...")
    # Add suite run to ToolAuditor
    results = await auditor.run_suite()
    
    print("\n📊 Audit Results:")
    print(f"Score: {results['score']}/{results['total']}")
    for r in results['details']:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        print(f"{status} | Q: {r['query'][:30]}... -> Tool: {r['selected_tool']} (Exp: {r['expected']})")
        
    if results['score'] == results['total']:
        print("\n🎉 ALL TESTS PASSED.")
        raise SystemExit(0)
    else:
        print("\n⚠️ SOME TESTS FAILED.")
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(main())


##
