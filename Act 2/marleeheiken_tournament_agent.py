#!/usr/bin/env python
"""
Tournament Agent: Improved Agent
Student: Marlee Heiken
Generated: 2026-02-13 22:51:01

Evolution Details:
- Generations: 100
- Final Fitness: N/A
- Trained against: Pavlov, Always Invest, Always Undercut, Generous Tit-for-Tat, Hard Majority...

Strategy: Cooperates with cooperative opponents but retaliates harshly against defectors
"""

from agents import Agent, INVEST, UNDERCUT
import random


class MarleeHeikenAgent(Agent):
    """
    Improved Agent
    
    Cooperates with cooperative opponents but retaliates harshly against defectors
    
    Evolved Genes: [0.7559765867303226, 0.5401606078484164, 1.0, 1.0, 0.16372069226057695, 0.0]
    """
    
    def __init__(self):
        # These genes were evolved through 100 generations
        self.genes = [0.7559765867303226, 0.5401606078484164, 1.0, 1.0, 0.16372069226057695, 0.0]
        
        # Required for tournament compatibility
        self.student_name = "Marlee Heiken"
        
        super().__init__(
            name="Improved Agent",
            description="Cooperates with cooperative opponents but retaliates harshly against defectors"
        )
    
    def choose_action(self) -> bool:
        """
        IMPROVED decision logic - AGGRESSIVE VERSION
        More likely to retaliate, less exploitable
        """
        
        # First 3 rounds: use initial cooperation gene
        if self.round_num < 3:
            return random.random() < self.genes[0]
        
        # Calculate memory window
        memory_length = max(1, int(self.genes[4] * 10) + 1)
        recent_history = self.history[-memory_length:]
        cooperation_rate = sum(recent_history) / len(recent_history)
        
        # AGGRESSIVE STRATEGY: Higher threshold for cooperation
        
        # Only cooperate if opponent is VERY cooperative (>80%)
        if cooperation_rate > 0.8:
            return random.random() < self.genes[1]  # gene[1] should evolve HIGH
        
        # If somewhat cooperative (50-80%), be cautious
        elif cooperation_rate > 0.5:
            # Mix of cooperation and defection
            coop_prob = self.genes[1] * (cooperation_rate - 0.5) * 2  # Scale 0-1
            return random.random() < coop_prob
        
        # If aggressive (<50%), retaliate hard
        else:
            # Mostly defect, with small chance to forgive
            if random.random() < self.genes[3] * 0.3:  # Reduced forgiveness
                return INVEST
            else:
                return UNDERCUT



# Convenience function for tournament loading
def get_agent():
    """Return an instance of this agent for tournament use"""
    return MarleeHeikenAgent()


if __name__ == "__main__":
    # Test that the agent can be instantiated
    agent = get_agent()
    print(f"✅ Agent loaded successfully: {agent.name}")
    print(f"   Genes: {agent.genes}")
    print(f"   Description: {agent.description}")
