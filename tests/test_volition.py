################################################################################

import asyncio
import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from core.config import config
from core.volition import VolitionEngine


@pytest.fixture
def mock_orchestrator():
    orchestrator = MagicMock()
    orchestrator.status.running = True
    orchestrator.cognitive_engine = MagicMock()
    orchestrator.project_store = MagicMock()
    orchestrator.strategic_planner = MagicMock()
    
    # Mock soul drives
    mock_soul = MagicMock()
    mock_drive = MagicMock()
    mock_drive.name = "Connection"
    mock_drive.urgency = 0.5
    mock_soul.get_dominant_drive.return_value = mock_drive
    orchestrator.soul = mock_soul
    
    orchestrator.conversation_history = []
    return orchestrator

@pytest.fixture
def engine(mock_orchestrator):
    with patch("core.volition.config") as mock_config:
        # Mock the config paths to prevent AttributeError during __init__
        mock_config.paths = MagicMock()
        mock_config.paths.brain_dir = Path("/tmp/mock_brain")
        mock_config.paths.brain_dir.mkdir(parents=True, exist_ok=True)
        # Ensure it behaves like a Path object for .exists() etc.
        mock_config.paths.data_dir = Path("/tmp/mock_data")
        return VolitionEngine(mock_orchestrator)

def test_init(engine, mock_orchestrator):
    assert engine.orchestrator == mock_orchestrator
    assert engine.brain == mock_orchestrator.cognitive_engine
    assert engine.boredom_threshold == 45
    assert not engine.is_dreaming
    assert engine._consecutive_idle_cycles == 0
    assert "the future of bio-computing" in engine.general_interests


@pytest.mark.asyncio
async def test_tick_skip_not_running(engine):
    engine.orchestrator.status.running = False
    assert await engine.tick(current_goal=None) is None


@pytest.mark.asyncio
async def test_tick_skip_has_goal(engine):
    engine.last_activity_time = 0.0
    engine._consecutive_idle_cycles = 5
    
    assert await engine.tick(current_goal="Working on a user request") is None
    
    assert engine.last_activity_time > 0.0
    assert engine._consecutive_idle_cycles == 0


@pytest.mark.asyncio
async def test_search_for_autonomous_goals_boredom(engine):
    # Fast forward time to trigger boredom
    engine.last_activity_time = time.time() - 100
    engine.boredom_threshold = 45
    
    goals = await engine._search_for_autonomous_goals()
    assert len(goals) > 0
    
    boredom_origin = goals[0].get("origin")
    assert boredom_origin in [
        "intrinsic_duty", 
        "intrinsic_duty_strategic", 
        "intrinsic_reflection", 
        "intrinsic_curiosity", 
        "intrinsic_fun",
        "intrinsic_evolution"
    ]


@patch("core.volition.random.random")
@pytest.mark.asyncio
async def test_generate_duty_goal_strategic(mock_random, engine):
    mock_random.return_value = 0.1  # Force duty goal path
    
    mock_project = MagicMock()
    mock_project.name = "Test Project"
    mock_project.id = "proj_1"
    engine.orchestrator.project_store.get_active_projects.return_value = [mock_project]
    
    mock_task = MagicMock()
    mock_task.description = "Test task description"
    mock_task.id = "task_1"
    engine.orchestrator.strategic_planner.get_next_task.return_value = mock_task
    
    goal = await engine._generate_duty_goal()
    assert goal is not None
    assert goal["origin"] == "intrinsic_duty_strategic"
    assert "Test Project" in goal["objective"]


def test_notify_activity(engine):
    engine.last_activity_time = 0.0
    engine._consecutive_idle_cycles = 10
    engine.unanswered_speak_count = 3
    engine.speak_backoff_multiplier = 4.0
    
    engine.notify_activity()
    
    assert engine.last_activity_time > 0.0
    assert engine._consecutive_idle_cycles == 0
    assert engine.unanswered_speak_count == 0
    assert engine.speak_backoff_multiplier == 1.0


def test_check_soul_drives_connection(engine):
    # Set urgency high enough to trigger connection
    engine.orchestrator.soul.get_dominant_drive.return_value.name = "Connection"
    engine.orchestrator.soul.get_dominant_drive.return_value.urgency = 0.9
    engine.last_speak_time = 0.0  # Force cooldown pass
    
    goal = engine._check_soul_drives()
    assert goal is not None
    assert goal["origin"] == "intrinsic_connection"
    assert goal["speak"] is True
    assert engine.unanswered_speak_count == 1


def test_check_soul_drives_connection_silenced(engine):
    engine.orchestrator.soul.get_dominant_drive.return_value.name = "Connection"
    engine.orchestrator.soul.get_dominant_drive.return_value.urgency = 0.9
    
    engine.unanswered_speak_count = engine.max_unanswered_before_silence
    
    goal = engine._check_soul_drives()
    assert goal is None  # Suppressed due to unanswered messages


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists")
def test_load_interests_empty(mock_exists, mock_file, engine):
    mock_exists.return_value = True
    mock_file.return_value.read.return_value = "{}"  # Empty JSON
    
    engine.general_interests = []
    engine.load_interests()
    
    assert "the future of bio-computing" in engine.general_interests


@patch("core.volition.random.choices")
@patch("core.volition.random.choice")
def test_generate_impulse(mock_choice, mock_choices, engine):
    # Setup mocks to force a specific impulse type and template
    mock_choices.return_value = ["question"]
    mock_choice.side_effect = ["Ask the user what they think about {topic}.", "the future of bio-computing"]
    
    # Needs some general interests to function
    engine.general_interests = ["the future of bio-computing"]
    
    now = time.time()
    impulse = engine._generate_impulse(now)
    
    assert impulse is not None
    assert impulse["origin"] == "impulse_question"
    assert "bio-computing" in impulse["objective"]
    assert impulse["speak"] is True
    assert engine.unanswered_speak_count == 1

@patch("core.volition.random.choices")
def test_generate_impulse_silenced(mock_choices, engine):
    mock_choices.return_value = ["question"]
    engine.unanswered_speak_count = engine.max_unanswered_before_silence
    
    # We don't care about the random result, just that 'speak' is forced false
    impulse = engine._generate_impulse(time.time())
    
    assert impulse is not None
    assert impulse["speak"] is False
    assert "[Internal thought" in impulse["objective"]


@pytest.mark.asyncio
async def test_generate_duty_goal_fallback(engine):
    # Test fall back to task.md when strategic planner is not active
    engine.orchestrator.strategic_planner = None
    
    with patch("builtins.open", mock_open(read_data="- [ ] Refactor Core")) as mock_file:
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("os.path.getmtime") as mock_mtime:
                mock_mtime.return_value = time.time()
                with patch("pathlib.Path.rglob") as mock_rglob:
                    mock_file_path = Path("/tmp/mock_brain/task.md")
                    mock_rglob.return_value = [mock_file_path]
                    
                    goal = await engine._generate_duty_goal()
                    
                    assert goal is not None
                    assert goal["origin"] == "intrinsic_duty"
                    assert "Refactor Core" in goal["objective"]


def test_generate_reflection_goal(engine):
    with patch("core.volition.random.choice", return_value="Reflect on my own thinking patterns"):
        goal = engine._generate_reflection_goal()
        
        assert goal is not None
        assert goal["origin"] == "intrinsic_reflection"
        assert "Reflect on my own thinking patterns" in goal["objective"]


def test_generate_curiosity_goal_educational(engine):
    engine.general_interests = ["robotics"]
    with patch("core.volition.random.choice", side_effect=["robotics", "Research the history of {topic}."]):
        goal = engine._generate_curiosity_goal("educational")
        
        assert goal is not None
        assert goal["origin"] == "intrinsic_curiosity"
        assert "robotics" in goal["objective"]


def test_generate_curiosity_goal_fun(engine):
    engine.fun_interests = ["origami"]
    with patch("core.volition.random.choice", side_effect=["origami", "Spend some time {topic} just for fun."]):
        goal = engine._generate_curiosity_goal("fun")
        
        assert goal is not None
        assert goal["origin"] == "intrinsic_fun"
        assert "origami" in goal["objective"]

def test_check_soul_drives_competence(engine):
    engine.orchestrator.soul.get_dominant_drive.return_value.name = "Competence"
    engine.orchestrator.soul.get_dominant_drive.return_value.urgency = 0.6
    
    goal = engine._check_soul_drives()
    assert goal is not None
    assert goal["origin"] == "intrinsic_competence"

def test_check_soul_drives_curiosity(engine):
    engine.orchestrator.soul.get_dominant_drive.return_value.name = "Curiosity"
    engine.orchestrator.soul.get_dominant_drive.return_value.urgency = 0.7
    
    goal = engine._check_soul_drives()
    assert goal is not None
    assert goal["origin"] == "intrinsic_curiosity"


def test_scan_roadmap(engine):
    engine.brain_base = MagicMock()
    engine.brain_base.exists.return_value = True
    
    mock_file_path = MagicMock()
    engine.brain_base.glob.return_value = [mock_file_path]
    
    with patch("builtins.open", mock_open(read_data="# Phase 1\nSome text")):
        milestones = engine._scan_roadmap()
        assert milestones == ["Phase 1"]


@patch("core.volition.random.random")
def test_check_roadmap(mock_random, engine):
    engine.milestones = ["Phase 1", "Phase 2"]
    mock_random.return_value = 0.01  # Force hit
    
    goal = engine._check_roadmap()
    
    assert goal is not None
    assert goal["origin"] == "intrinsic_evolution"
    assert "Phase 2" in goal["objective"]

def test_select_and_parse_goal(engine):
    goals = [
        {"objective": "Impulse goal", "origin": "impulse_question", "id": "1"},
        {"objective": "Strategic duty", "origin": "intrinsic_duty_strategic", "id": "2"}
    ]
    
    selected = engine._select_and_parse_goal(goals)
    # Strategic duty always overrides
    assert selected["origin"] == "intrinsic_duty_strategic"
    
    # Test selecting the first if no strategic
    goals = [
        {"objective": "Impulse goal", "origin": "impulse_question", "id": "1"},
    ]
    selected = engine._select_and_parse_goal(goals)
    assert selected["origin"] == "impulse_question"


##
