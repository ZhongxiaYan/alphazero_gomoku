import agent, ai_agent
import display

HUMAN_CMD_LINE = agent.CommandLineInputAgent
HUMAN_GUI = agent.GuiInputAgent
AI_REFLEX = ai_agent.ReflexAgent
AI_REFLEX_CACHED = ai_agent.ReflexCachedAgent
AI_MINIMAX = ai_agent.MinimaxAgent
AI_MCTS = ai_agent.MCTSAgent

DISPLAY_COMMAND_LINE = display.CommandLineDisplay
DISPLAY_GUI = display.PyQtDisplay