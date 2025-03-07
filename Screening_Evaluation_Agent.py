import os
import asyncio
import google.generativeai as genai
import PyPDF2
import docx
import chardet
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.model_context import BufferedChatCompletionContext

# Configuration
JD_FOLDER = r"D:\Aditya Program files\AI-AGENT\AUTOGEN\Resume_Agent\JD"
RESUME_FOLDER = r"D:\Aditya Program files\AI-AGENT\AUTOGEN\Resume_Agent\Resumes"
API_KEY = "AIzaSyApWYeLpqS7cLKCF3ryY4c2Yq69w2i7f5c"
MODEL_NAME = "gemini-1.5-flash"

# Configure Gemini API
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel(MODEL_NAME)

# Utility Functions
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def read_pdf_file(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def read_docx_file(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        raw_data = file.read(1024)
    return chardet.detect(raw_data)["encoding"]

def read_resume(file_path):
    ext = file_path.lower().split(".")[-1]
    try:
        if ext == "pdf":
            return read_pdf_file(file_path)
        elif ext == "docx":
            return read_docx_file(file_path)
        elif ext == "txt":
            encoding = detect_encoding(file_path)
            return read_text_file(file_path).encode(encoding).decode("utf-8")
        else:
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Loading JDs and Resumes
def load_text_from_folder(folder_path):
    documents = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            try:
                documents[file_name] = read_resume(file_path)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    return documents

async def evaluate_candidate(jd_name, jd_text, resume_name, resume_text):
    """
    Evaluate a candidate's resume against the job description using Gemini API.
    """
    print(f"\nEvaluating Candidate: {resume_name} for Job: {jd_name}")

    prompt = f"""
    Analyze the following resume against this job description and provide a clear assessment.

    Resume:
    {resume_text}

    Job Description:
    {jd_text}

    **Evaluation Criteria:**
    - Skills match: List skills from the resume that match the job description.
    - Experience match: Summarize relevant experience.
    - Qualifications: Assess if the candidate meets the required qualifications.
    - Weaknesses: Identify any missing qualifications.
    - Do not include the full resume or job description in your response.
    - Do not invent information.

    **Provide a structured response with:**
    - Strengths
    - Weaknesses
    - Final Hiring Decision: 'APPROVE' if a strong fit, or 'REJECT' otherwise.
    """

    try:
        response = gemini_model.generate_content(prompt)
        print(f"\n**Evaluation Result for {resume_name}:**\n{response.text}")
    except Exception as e:
        print(f"Error generating response: {e}")

async def main():
    job_descriptions = load_text_from_folder(JD_FOLDER)
    resumes = load_text_from_folder(RESUME_FOLDER)

    evaluator = AssistantAgent(
        name="Evaluator",
        model_client=gemini_model,
        system_message="""
       Analyze the following resume against this job description and provide a clear assessment.

**Evaluation Criteria:**
- **Skills Match:** Identify key skills from the resume that align with the job description.
- **Experience Match:** Summarize relevant work experience.
- **Qualifications:** Assess if the candidate meets the necessary qualifications.
- **Weaknesses:** Highlight any missing skills, experience, or qualifications.

**Response Format:**  
**Strengths:**  
- List specific strengths with bullet points.  
**Weaknesses:**  
- List any concerns or gaps in the resume.  
**Final Hiring Decision:**  
- Provide a clear decision: 'APPROVE' if the candidate is a strong fit, otherwise 'REJECT'.  
- Justify the decision concisely.

**Guidelines:**
- Do **not** include the full resume or job description in your response.
- Do **not** invent information or make assumptions.
        """,
        model_context=BufferedChatCompletionContext(buffer_size=10),
    )

    critic = AssistantAgent(
        name="Critic",
        model_client=gemini_model,
        system_message="""
        Review the evaluation provided by the Evaluator.

**Your Task:**  
- Verify the correctness and completeness of the assessment.
- Ensure consistency with the job description requirements.
- Refine and structure the response clearly.

**Response Format:**  
**Strengths:**  
- Summarize the key strengths in a structured manner.  
**Weaknesses:**  
- Highlight missing skills, experience, or other concerns.  
**Final Hiring Decision:**  
- Provide a **final decision**: 'APPROVE' if the candidate is a strong fit, or 'REJECT' otherwise.  
- Include a **justification** with a concise explanation.

**Guidelines:**
- Do **not** add unnecessary details or assumptions.
- Keep the response structured and professional.
        """,
        model_context=BufferedChatCompletionContext(buffer_size=10),
    )

    termination_condition = TextMentionTermination("APPROVE|REJECT")
    team = RoundRobinGroupChat([evaluator, critic], termination_condition=termination_condition)

    for jd_name, jd_text in job_descriptions.items():
        print(f"\n**Processing Job Description: {jd_name}**")

        for resume_name, resume_text in resumes.items():
            print(f"\nEvaluating Candidate: {resume_name}")
            await evaluate_candidate(jd_name, jd_text, resume_name, resume_text)
            await team.reset()

if __name__ == "__main__":
    asyncio.run(main())
