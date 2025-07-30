import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled,Runner , ModelSettings, InputGuardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered,RunContextWrapper

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("BASE_URL") 
API_KEY = os.getenv("OPENAI_API_KEY") 
MODEL_NAME = os.getenv("MODEL_NAME") 

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set BASE_URL, API_KEY, and MODEL_NAME."
    )
    

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


# --- Models for structured outputs ---

class Job(BaseModel):
    title: str = Field(description="Job title or position name")
    company: str = Field(description="Name of the company offering the job")
    location: str = Field(description="Location of the job")
    requirements: List[str] = Field(description="List of required qualifications or skills for the job")
    salary_range: Optional[str] = Field(default=None, description="Salary range offered for the job, e.g., '$60k-$80k'")
    link: Optional[str] = Field(default=None, description="URL to the job post")

class SkillGap(BaseModel):
    target_job: str = Field(description="Job role the user is aiming for")
    user_skills: List[str] = Field(description="List of skills the user already has")
    required_skills: List[str] = Field(description="List of skills required for the target job")
    missing_skills: List[str] = Field(description="List of skills the user is missing")

class Course(BaseModel):
    name: str = Field(description="Course title or name")
    provider: str = Field(description="Platform or institution offering the course")
    duration_hours: Optional[float] = Field(default=None, description="Approximate duration of the course in hours")
    skills_covered: List[str] = Field(description="List of skills covered in the course")
    link: Optional[str] = Field(default=None, description="URL link to the course")



@dataclass
class UserContext:  
    user_id: str
    current_skills: List[str] = None
    preferred_location: Optional[str] = None
    session_start: datetime = None
    
    def __post_init__(self):
        if self.current_skills is None:
            self.current_skills = []
        if self.session_start is None:
            self.session_start = datetime.now()



@function_tool
def get_missing_skills(user_skills: list, target_job: str) -> dict:
    """Identify missing skills for a target job role based on the user's current skills."""
    required_skills_map = {
        "data scientist": ["Python", "SQL", "Machine Learning", "Statistics", "Data Visualization"],
        "web developer": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
        "data analyst": ["Excel", "SQL", "Tableau", "Statistics", "Python"],
        "project manager": ["Project Planning", "Communication", "Leadership", "Risk Management", "Agile"]
    }
    
    required_skills = required_skills_map.get(target_job.lower(), [])
    missing_skills = [skill for skill in required_skills if skill not in user_skills]
    
    return {
        "target_job": target_job.title(),
        "user_skills": user_skills,
        "required_skills": required_skills,
        "missing_skills": missing_skills
    }


@function_tool
def find_jobs(user_skills: list, location: str = None) -> list:
    """Find job openings based on user skills and optional location filter."""
    job_listings = [
        {
            "title": "Junior Data Scientist",
            "company": "TechCorp Analytics",
            "location": "New York, NY",
            "requirements": ["Python", "SQL", "Machine Learning"],
            "salary_range": "$70k - $90k",
            "link": "https://techcorp.com/jobs/123"
        },
        {
            "title": "Web Developer",
            "company": "CreativeSoft Solutions",
            "location": "Remote",
            "requirements": ["HTML", "CSS", "JavaScript"],
            "salary_range": "$50k - $65k",
            "link": "https://creativesoft.com/jobs/456"
        },
        {
            "title": "Data Analyst",
            "company": "Insight Analytics",
            "location": "Chicago, IL",
            "requirements": ["Excel", "SQL", "Tableau"],
            "salary_range": "$55k - $75k",
            "link": "https://insightanalytics.com/jobs/789"
        }
    ]
    
    # Filter by location (if provided)
    filtered_jobs = [job for job in job_listings if not location or location.lower() in job["location"].lower()]
    
    # Match skills
    matching_jobs = [
        job for job in filtered_jobs
        if any(skill.lower() in [req.lower() for req in job["requirements"]] for skill in user_skills)
    ]
    
    # Fallback: if no match, return top 2 jobs
    if not matching_jobs:
        matching_jobs = filtered_jobs[:2] if filtered_jobs else job_listings[:2]
    
    return matching_jobs


@function_tool
def recommend_courses(missing_skills: list) -> list:
    """Recommend online courses to learn missing skills."""
    course_recommendations = {
        "Python": [
            {
                "name": "Python for Everybody",
                "provider": "Coursera",
                "duration_hours": 40,
                "skills_covered": ["Python"],
                "link": "https://www.coursera.org/learn/python"
            }
        ],
        "Machine Learning": [
            {
                "name": "Machine Learning by Andrew Ng",
                "provider": "Coursera",
                "duration_hours": 60,
                "skills_covered": ["Machine Learning", "AI Basics"],
                "link": "https://www.coursera.org/learn/machine-learning"
            }
        ],
        "SQL": [
            {
                "name": "SQL for Data Analysis",
                "provider": "Udemy",
                "duration_hours": 20,
                "skills_covered": ["SQL"],
                "link": "https://www.udemy.com/course/sql-for-data-analysis"
            }
        ],
        "Pandas": [
            {
                "name": "Pandas Masterclass",
                "provider": "DataCamp",
                "duration_hours": 25,
                "skills_covered": ["Pandas"],
                "link": "https://www.datacamp.com/courses/pandas"
            }
        ]
    }
    
    recommended_courses = []
    for skill in missing_skills:
        if skill in course_recommendations:
            recommended_courses.extend(course_recommendations[skill])
    
    return recommended_courses


# --- Guardrail ---
async def skill_input_guardrail(ctx, agent, input_data):
    """Ensure the user input is not empty or nonsense."""
    try:
        cleaned = input_data.strip().lower()
        if len(cleaned) == 0:
            return GuardrailFunctionOutput(
                output_info="No query provided.",
                tripwire_triggered=True
            )
        return GuardrailFunctionOutput(
            output_info="Valid input, no guardrail triggered.",
            tripwire_triggered=False
        )
    except Exception as e:
        return GuardrailFunctionOutput(
            output_info=f"Guardrail check failed: {e}",
            tripwire_triggered=False
        )

# --- Skill Gap Agent ---
skill_gap_agent = Agent(
    name="Skill Gap Specialist",
    handoff_description="Handles questions about what skills are required for a job or how to become something.",
    instructions="""
    You specialize in skill gap analysis.
    When users ask about becoming a certain professional or required skills for a job:
    - Use the get_missing_skills tool
    - Compare user skills with required skills
    - Return required and missing skills clearly.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_missing_skills],
    output_type=SkillGap
)

# --- Job Finder Agent ---
job_finder_agent = Agent(
    name="Job Finder Specialist",
    handoff_description="Handles job search queries like 'find jobs' or 'job opportunities'.",
    instructions="""
    You specialize in job search.
    When users ask about finding jobs:
    - Use the find_jobs tool
    - Match jobs to their skills and location if given
    - Return 2-3 job recommendations.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[find_jobs],
    output_type=List[Job]
)

# --- Course Recommender Agent ---
course_recommender_agent = Agent(
    name="Course Recommender Specialist",
    handoff_description="Handles queries about learning skills or finding courses.",
    instructions="""
    You specialize in recommending learning resources.
    When users ask how to learn skills or want course recommendations:
    - Use the recommend_courses tool
    - Provide 2-3 relevant online courses per skill.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[recommend_courses],
    output_type=List[Course]
)

# --- Conversation Agent (Main Controller) ---
conversation_agent = Agent(
    name="CareerMate Main Controller",
    instructions="""
    You are CareerMate, a friendly career guidance assistant.
    Your job:
    - Understand the user's request
    - Handoff to:
        * Skill Gap Specialist → for 'become', 'required skills', 'what skills do I need'
        * Job Finder Specialist → for 'find jobs', 'job opportunities'
        * Course Recommender Specialist → for 'learn', 'courses', 'training'
    If unclear, politely ask for clarification.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    handoffs=[skill_gap_agent, job_finder_agent, course_recommender_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=skill_input_guardrail),
    ]
)


# --- Main Function ---

async def main():
    # Create a user context
    user_context = UserContext(
        user_id="user123",
        current_skills=["programming", "statistics", "machine learning", "data analysis"],
        preferred_location="New York"
    )
    
    # Example queries
    queries = [
        "I want to become a data scientist",                # Skill Gap Agent
        "Can you help me find jobs?",                       # Job Finder Agent
        "How do I learn SQL and Pandas?",                   # Course Recommender Agent
        "What skills do I need to become a project manager",# Another skill gap query
        "Find remote jobs for web development"              # Another job search query
    ]
    
    for query in queries:
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("="*50)

        try:
            # Run Conversation Agent (handles routing + handoffs)
            result = await Runner.run(conversation_agent, query, context=user_context)

            print("\nFINAL RESPONSE:")

            # --- Skill Gap Output ---
            if hasattr(result.final_output, "missing_skills"):
                skill_gap = result.final_output
                print(f"Target Job: {skill_gap.target_job}")
                print(f"User Skills: {skill_gap.user_skills}")
                print(f"Required Skills: {skill_gap.required_skills}")
                print(f"Missing Skills: {skill_gap.missing_skills}")

            # --- Job Recommendations Output ---
            elif isinstance(result.final_output, list) and result.final_output and hasattr(result.final_output[0], "company"):
                print("\n📝 JOB RECOMMENDATIONS 📝")
                for job in result.final_output:
                    print(f"- {job.title} at {job.company} ({job.location})")
                    print(f"  Requirements: {', '.join(job.requirements)}")
                    if job.salary_range:
                        print(f"  Salary: {job.salary_range}")
                    if job.link:
                        print(f"  Link: {job.link}")
                    print()

            # --- Course Recommendations Output ---
            elif isinstance(result.final_output, list) and result.final_output and hasattr(result.final_output[0], "provider"):
                print("\n🎓 COURSE RECOMMENDATIONS 🎓")
                for course in result.final_output:
                    print(f"- {course.name} by {course.provider}")
                    if course.duration_hours:
                        print(f"  Duration: {course.duration_hours} hours")
                    print(f"  Skills Covered: {', '.join(course.skills_covered)}")
                    if course.link:
                        print(f"  Link: {course.link}")
                    print()

            else:
                # Generic fallback
                print(result.final_output)

        except InputGuardrailTripwireTriggered as e:
            print("\n⚠️ GUARDRAIL TRIGGERED ⚠️")
            print(e)

if __name__ == "__main__":
    asyncio.run(main())
