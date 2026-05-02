import os

import requests
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai_tools import FileWriterTool
from pydantic import BaseModel, Field


class CreateLinearTicketArgs(BaseModel):
    ticket_title: str = Field(..., description="The title of the Linear issue.")
    ticket_description: str = Field(..., description="The markdown body / description for the issue.")


class CreateLinearTicketTool(BaseTool):
    name: str = "create_linear_ticket"
    description: str = "Create a new issue in Linear for the configured team."
    args_schema: type[BaseModel] = CreateLinearTicketArgs

    def _run(self, ticket_title: str, ticket_description: str) -> str:
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
                    "title": ticket_title,
                    "description": ticket_description,
                    "teamId": team_id,
                },
            },
        )

        if response.status_code != 200:
            return f"Linear HTTP {response.status_code}: {response.text}"

        payload = response.json()
        if payload.get("errors"):
            return f"Linear GraphQL error: {payload['errors']}"

        result = (payload.get("data") or {}).get("issueCreate") or {}
        if not result.get("success"):
            return f"Linear ticket creation failed: {payload}"

        issue = result.get("issue") or {}
        return f"Created Linear ticket {issue.get('identifier', '?')}: {issue.get('url', ticket_title)}"


create_linear_ticket = CreateLinearTicketTool()


llm = LLM(
    model="openrouter/openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


write_arch_tool = FileWriterTool(file_path='../docs/architecture_spec.md')
write_design_tool = FileWriterTool(file_path='../docs/design_system.md')


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
        "You are a meticulous Product Manager. You analyze architectural "
        "documents and automatically determine how many engineering tasks "
        "are required. You break complex systems down into perfectly scoped "
        "technical tickets."
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
        "Review the system architecture provided by the Tech Lead. Determine "
        "the logical sequence of engineering tasks needed to build Phase 1 "
        "of this AR app. Use your Linear tool to dynamically create a ticket "
        "for each distinct engineering task you identify. Do not stop until "
        "every part of the architecture is covered by a ticket. Include "
        "detailed acceptance criteria in each ticket based on the "
        "architecture specs."
    ),
    expected_output=(
        "A summary list of all the Linear tickets that were dynamically "
        "created."
    ),
    agent=pm_agent,
    context=[task_1_architecture],
)


crew = Crew(
    agents=[tech_lead, spatial_designer, pm_agent],
    tasks=[task_1_architecture, task_2_design, task_3_planning],
    process=Process.sequential,
)


if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
