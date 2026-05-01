import os

import requests
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai_tools import FileWriteTool
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


write_arch_tool = FileWriteTool(file_path='../docs/architecture_spec.md')
write_design_tool = FileWriteTool(file_path='../docs/design_system.md')


tech_lead = Agent(
    role="AR Systems Architect",
    goal=(
        "Design the Unity AR Foundation architecture and JSON schemas for "
        "IKEA furniture assembly."
    ),
    backstory=(
        "You are a senior Mixed Reality developer who writes pristine, "
        "technical markdown documentation."
    ),
    verbose=True,
    llm=llm,
    tools=[write_arch_tool],
)


spatial_designer = Agent(
    role="Spatial UI/UX Designer",
    goal=(
        "Design the holographic interfaces, 3D arrow logic, and user flows."
    ),
    backstory=(
        "You are an expert in Apple Vision Pro and Meta Quest spatial design "
        "patterns."
    ),
    verbose=True,
    llm=llm,
    tools=[write_design_tool],
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


task_1_architecture = Task(
    description=(
        "Draft the system architecture for the DYAI LACK Table AR App. "
        "Include the exact JSON schema for the assembly steps and the "
        "required Unity packages. Save the output using your file write tool."
    ),
    expected_output=(
        "A complete architecture specification in markdown, including the "
        "JSON schema and Unity package list, written to "
        "../docs/architecture_spec.md."
    ),
    agent=tech_lead,
)


task_2_design = Task(
    description=(
        "Based on the LACK table AR concept, design the visual states for "
        "the 3D directional arrows and the holographic 'ghost' furniture "
        "pieces. Save the output using your file write tool."
    ),
    expected_output=(
        "A markdown design system document covering arrow visual states and "
        "holographic ghost furniture, written to ../docs/design_system.md."
    ),
    agent=spatial_designer,
)


task_3_planning = Task(
    description=(
        "Review the project goals. Use your tool to create exactly 3 Linear "
        "tickets for Phase 1: 1. Setup Unity AR Foundation, 2. Implement the "
        "JSON parser, 3. Create the 3D Bezier Arrow Shader. Include detailed "
        "acceptance criteria in each ticket."
    ),
    expected_output=(
        "Confirmation that all 3 Linear tickets have been created, including "
        "the title of each ticket."
    ),
    agent=pm_agent,
)


crew = Crew(
    agents=[tech_lead, spatial_designer, pm_agent],
    tasks=[task_1_architecture, task_2_design, task_3_planning],
    process=Process.sequential,
)


if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
