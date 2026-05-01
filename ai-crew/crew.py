import os

import requests
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI


@tool("create_linear_ticket")
def create_linear_ticket(title: str, description: str) -> str:
    """Create a new issue in Linear for the configured team.

    Args:
        title: The title of the Linear issue.
        description: The markdown body / description for the issue.

    Returns:
        A status string describing the result of the API call.
    """
    api_key = os.getenv("LINEAR_API_KEY")
    team_id = os.getenv("LINEAR_TEAM_ID")

    query = """
    mutation IssueCreate($title: String!, $description: String!, $teamId: String!) {
      issueCreate(input: { title: $title, description: $description, teamId: $teamId }) {
        success
        issue { id identifier url title }
      }
    }
    """

    response = requests.post(
        "https://api.linear.app/graphql",
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "variables": {
                "title": title,
                "description": description,
                "teamId": team_id,
            },
        },
    )

    if response.status_code == 200:
        return f"Successfully created Linear ticket: {title}"
    return response.text


llm = ChatOpenAI(
    model="openai/gpt-4o",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


pm_agent = Agent(
    role="Product Manager",
    goal=(
        "Break down an AR application architecture into actionable Linear "
        "tickets."
    ),
    backstory=(
        "You are a seasoned product manager for the DYAI project. You take "
        "architectural designs and translate them into clear, actionable "
        "engineering tickets in Linear with detailed acceptance criteria."
    ),
    verbose=True,
    llm=llm,
    tools=[create_linear_ticket],
)


planning_task = Task(
    description=(
        "Use your `create_linear_ticket` tool to create exactly 3 Linear "
        "tickets for Phase 1 of our open-source AR guide for assembling an "
        "IKEA LACK table. The tickets must be:\n"
        "1. Setup the Unity AR Foundation project structure.\n"
        "2. Create the JSON schema for the assembly instructions.\n"
        "3. Build a basic C# script to parse that JSON.\n\n"
        "Each ticket description must include detailed acceptance criteria "
        "so an engineer can pick it up and execute without further "
        "clarification."
    ),
    expected_output=(
        "Confirmation that all 3 Linear tickets have been created, including "
        "the title of each ticket."
    ),
    agent=pm_agent,
)


crew = Crew(
    agents=[pm_agent],
    tasks=[planning_task],
    process=Process.sequential,
)


if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
