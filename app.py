
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from recruitment.constants import APP_HOST, APP_PORT
from recruitment.pipeline.prediction_pipeline import RecruitData, RecruitClassifier
from recruitment.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Age: Optional[str] = None
        self.ExperienceYears: Optional[str] = None
        self.PreviousCompanies: Optional[str] = None
        self.DistanceFromCompany: Optional[str] = None
        self.InterviewScore: Optional[str] = None
        self.SkillScore: Optional[str] = None
        self.PersonalityScore: Optional[str] = None
        self.Gender: Optional[str] = None
        self.EducationLevel: Optional[str] = None
        self.RecruitmentStrategy: Optional[str] = None
        
 
    async def get_recruit_data(self):
        form = await self.request.form()
        self.Age = form.get("Age")
        self.ExperienceYears = form.get("ExperienceYears")
        self.PreviousCompanies = form.get("PreviousCompanies")
        self.DistanceFromCompany = form.get("DistanceFromCompany")
        self.InterviewScore = form.get("InterviewScore")
        self.SkillScore = form.get("SkillScore")
        self.PersonalityScore = form.get("PersonalityScore")
        self.Gender = form.get("Gender")
        self.EducationLevel = form.get("EducationLevel")
        self.RecruitmentStrategy = form.get("RecruitmentStrategy")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "recruit.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_recruit_data()
        
        recruit_data = RecruitData(
                                Age= form.Age,
                                ExperienceYears = form.ExperienceYears,
                                PreviousCompanies = form.PreviousCompanies,
                                DistanceFromCompany = form.DistanceFromCompany,
                                InterviewScore= form.InterviewScore,
                                SkillScore= form.SkillScore,
                                PersonalityScore = form.PersonalityScore,
                                Gender= form.Gender,
                                EducationLevel= form.EducationLevel,
                                RecruitmentStrategy= form.RecruitmentStrategy,
                                )
        
        recruit_df = recruit_data.get_recruit_input_dataframe()

        model_predictor = RecruitClassifier()

        value = model_predictor.predict(dataframe=recruit_df)[0]

        status = None
        if value == 1:
            status = "Will be Hired For Job"
        else:
            status = "Will not be Hired For Job"

        return templates.TemplateResponse(
            "recruit.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)