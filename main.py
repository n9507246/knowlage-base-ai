from agents.my_agent.MyAgent import MyAgent
from interface.CLI.runner import CLI

def main():
    CLI.run(MyAgent(knowledge_base_path="./knowledge_base") )

if __name__ == "__main__":
    main()