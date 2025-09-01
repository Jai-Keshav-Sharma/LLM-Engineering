import modal
from agents.agent import Agent

class SpecialistAgent(Agent):
    """
    An agent that runs over fine-tuned LLM that's running remotely on Modal
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        """
        Set up this agent by creating an instance of the modal class
        """

        self.log("Specialist Agent initialising - connecting to Modal")
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()
        self.log("Specialist Agent in ready")

    def price(self, description: str) -> float:
        """
        Make a remote call to return the estimate of the price of this item
        """

        self.log("Specialist Agent is calling the remote fine-tuned model")
        result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result