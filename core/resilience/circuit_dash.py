import asyncio
import time
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.text import Text
import psutil

# Import metrics if possible
try:
    from core.resilience.resilience import SmartCircuitBreaker, PROMETHEUS_AVAILABLE
except ImportError:
    PROMETHEUS_AVAILABLE = False

class CircuitDash:
    """
    Rich-based terminal dashboard for Aura Production Hardening.
    """
    def __init__(self, refresh_rate: float = 1.0):
        self.console = Console()
        self.refresh_rate = refresh_rate
        self.running = False
        self._task = None

    async def start(self):
        self.running = True
        self._task = asyncio.create_task(self._draw_loop())

    async def stop(self):
        self.running = False
        if self._task:
            await self._task

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="circuits", ratio=2),
            Layout(name="resources", ratio=1)
        )
        return layout

    def _get_circuit_table(self) -> Table:
        table = Table(title="[bold cyan]Subsystem Circuit Breakers[/bold cyan]", expand=True)
        table.add_column("Subsystem", style="white")
        table.add_column("State", justify="center")
        table.add_column("Failures", justify="right")
        table.add_column("Total Calls", justify="right")
        table.add_column("Last Trip", justify="right")

        # This is a bit hacky - we need access to created circuit breakers.
        # In a real system, we'd have a registry.
        # For now, we'll try to find them in common places or just show placeholder if none.
        
        # Simulated/Placeholder if none registered (actually LLMRouter maintains some)
        from core.container import ServiceContainer
        router = ServiceContainer.get("llm_router", default=None)
        circuits = []
        if router and hasattr(router, 'adapters'):
            for name, adapter in router.adapters.items():
                if hasattr(adapter, 'circuit'):
                    circuits.append(adapter.circuit)
        
        if not circuits:
            table.add_row("No circuits registered", "-", "-", "-", "-")
        else:
            for c in circuits:
                status_color = "green" if c.state == "CLOSED" else ("yellow" if c.state == "HALF_OPEN" else "red")
                last_trip = f"{time.time() - c.last_failure:.1f}s ago" if c.last_failure > 0 else "-"
                table.add_row(
                    c.name,
                    f"[{status_color}]{c.state}[/{status_color}]",
                    str(c.failures),
                    str(c.total_calls),
                    last_trip
                )
        return table

    def _get_resource_panel(self) -> Panel:
        mem = psutil.virtual_memory()
        proc = psutil.Process()
        rss_gb = proc.memory_info().rss / (1024**3)
        
        msg = f"[bold]System Memory:[/bold] {mem.percent}% used\n"
        msg += f"[bold]Aura RSS:[/bold] {rss_gb:.2f} GB\n"
        msg += f"[bold]CPU:[/bold] {psutil.cpu_percent()}%"
        
        # Simple health bar
        health = 100 - (mem.percent * 0.5 + psutil.cpu_percent() * 0.5)
        health_color = "green" if health > 70 else ("yellow" if health > 40 else "red")
        
        return Panel(msg, title=f"[{health_color}]Resource Health: {health:.0f}%[/{health_color}]")

    async def _draw_loop(self):
        layout = self._make_layout()
        layout["header"].update(Panel("[bold white]Aura Zenith Production Node Monitoring[/bold white]", style="bg:blue"))
        layout["footer"].update(Panel(f"Press Ctrl+C to exit dashboard | Refresh: {self.refresh_rate}s", style="dim"))

        with Live(layout, console=self.console, screen=True, refresh_per_second=1/self.refresh_rate) as live:
            while self.running:
                layout["circuits"].update(self._get_circuit_table())
                layout["resources"].update(self._get_resource_panel())
                await asyncio.sleep(self.refresh_rate)

if __name__ == "__main__":
    # Test stub
    dash = CircuitDash()
    asyncio.run(dash.start())
