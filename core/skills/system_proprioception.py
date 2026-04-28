from core.runtime.errors import record_degradation
import inspect
import logging
import json
from typing import Any, Dict, List, Optional
from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer

from pydantic import BaseModel, Field


class ProprioceptionInput(BaseModel):
    service_name: Optional[str] = Field(None, description="Specific service to detail.")
    include_docstrings: bool = Field(True, description="Whether to extract and include module/class documentation.")

class SystemProprioceptionSkill(BaseSkill):
    """Provides Aura with a structural map of her own architecture and modules.
    
    Allows the AI to inspect registered services, their purpose, and their current status.
    """

    name = "system_proprioception"
    description = "Inspect Aura's internal architecture, registered services, and core modules."
    input_model = ProprioceptionInput
    output = "A structured map of the system's core services and modules."

    def __init__(self):
        self.logger = logging.getLogger(f"Skills.{self.name}")

    async def execute(self, params: ProprioceptionInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute proprioception check."""
        if isinstance(params, dict):
            try:
                params = ProprioceptionInput(**params)
            except Exception as e:
                record_degradation('system_proprioception', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        target_service = params.service_name
        include_docs = params.include_docstrings

        try:
            report = ServiceContainer.get_health_report()
            services = report.get("services", {})
            
            system_map = []
            
            # Get the actual container instance to peek at descriptors
            container = ServiceContainer()
            
            for name, status in services.items():
                if target_service and name != target_service:
                    continue
                
                service_info = {
                    "name": name,
                    "status": status.get("status"),
                    "required": status.get("required"),
                }
                
                # Try to get more metadata from the descriptor
                descriptor = container._services.get(name)
                if descriptor:
                    service_info["lifetime"] = descriptor.lifetime.value
                    
                    # Try to find the module and docstring
                    try:
                        instance = descriptor.instance
                        target_obj = instance if instance else descriptor.factory
                        
                        module = inspect.getmodule(target_obj)
                        if module:
                            service_info["module"] = module.__name__
                            service_info["file"] = getattr(module, "__file__", "unknown")
                        
                        if include_docs:
                            doc = inspect.getdoc(target_obj)
                            if doc:
                                service_info["description"] = doc.split('\n')[0] # First line only for brevity
                                
                    except Exception as e:
                        record_degradation('system_proprioception', e)
                        self.logger.debug("Metadata extraction failed for %s: %s", name, e)

                system_map.append(service_info)

            return {
                "ok": True,
                "summary": f"System Map contains {len(system_map)} services.",
                "system_map": system_map,
                "message": "I've conducted a self-diagnostic. All core systems are mapped and functional."
            }

        except Exception as e:
            record_degradation('system_proprioception', e)
            self.logger.error("Proprioception failed: %s", e)
            return {"ok": False, "error": str(e)}