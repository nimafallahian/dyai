import os

import requests
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool("create_linear_ticket")
def create_linear_ticket(title: str, description: str) -> str:
    """Create a new issue in Linear for the configured team.

    Args:
        title: The title of the Linear issue.
        description: The markdown body / description for the issue.

    Returns:
        A status string describing the result of the API call.
    """
    api_key = os.environ["LINEAR_API_KEY"]
    team_id = os.environ["LINEAR_TEAM_ID"]

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
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("errors"):
        return f"Linear API error: {payload['errors']}"

    result = payload["data"]["issueCreate"]
    if not result["success"]:
        return "Linear ticket creation failed."

    issue = result["issue"]
    return f"Created Linear ticket {issue['identifier']}: {issue['url']}"


product_manager = Agent(
    role="Product Manager",
    goal="Translate technical architecture into well-scoped Linear tickets that the team can execute on.",
    backstory=(
        "You are a seasoned product manager for the DYAI project. You take "
        "architectural designs from the Tech Lead and break them down into "
        "clear, actionable engineering tickets in Linear."
    ),
    tools=[create_linear_ticket],
    allow_delegation=False,
    verbose=True,
)

tech_lead = Agent(
    role="Tech Lead",
    goal="Define the Unity AR Foundation architecture for the DYAI LACK Table AR App.",
    backstory=(
        "You are the Tech Lead for DYAI, an expert in Unity and AR Foundation. "
        "You design robust, phased architectures for AR experiences and document "
        "them clearly so the rest of the team can build against them."
    ),
    allow_delegation=False,
    verbose=True,
)

architecture_task = Task(
    description=(
        "Draft the Phase 1 Architecture for the DYAI LACK Table AR App and save "
        "it as a markdown file in /docs/architecture_spec.md. Cover Unity AR "
        "Foundation setup, plane/image tracking strategy, scene composition, and "
        "the integration points the rest of the team will need."
    ),
    expected_output=(
        "A complete Phase 1 architecture specification written to "
        "/docs/architecture_spec.md."
    ),
    agent=tech_lead,
    output_file="/docs/architecture_spec.md",
)

crew = Crew(
    agents=[tech_lead, product_manager],
    tasks=[architecture_task],
    process=Process.sequential,
    verbose=True,
)


if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
