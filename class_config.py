import agent, aiAgent
import display

HUMAN_CMD_LINE = agent.CommandLineInputAgent
HUMAN_GUI = agent.GuiInputAgent
AI_REFLEX = aiAgent.ReflexAgent
AI_REFLEX_CACHED = aiAgent.ReflexCachedAgent
AI_MINIMAX = aiAgent.MinimaxAgent
AI_MCTS = aiAgent.MCTSAgent

DISPLAY_COMMAND_LINE = display.CommandLineDisplay
DISPLAY_GUI = display.PyQtDisplay