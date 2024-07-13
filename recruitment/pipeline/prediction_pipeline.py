import os
import sys

import numpy as np
import pandas as pd
from recruitment.entity.config_entity import RecruitPredictorConfig
from recruitment.entity.s3estimator import RecruitEstimator
from recruitment.exception import RecruitException
from recruitment.logger import logging
from recruitment.utils.main_utils import read_yaml_file
from pandas import DataFrame


class RecruitData:
    def __init__(self,
                Age,
                ExperienceYears,
                PreviousCompanies,
                DistanceFromCompany,
                InterviewScore,
                SkillScore,
                PersonalityScore,
                Gender,
                EducationLevel,
                RecruitmentStrategy
                ):
        """
        Recruit Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Age = Age
            self.ExperienceYears = ExperienceYears
            self.PreviousCompanies = PreviousCompanies
            self.DistanceFromCompany = DistanceFromCompany
            self.InterviewScore = InterviewScore
            self.SkillScore = SkillScore
            self.PersonalityScore = PersonalityScore
            self.Gender = Gender
            self.EducationLevel = EducationLevel
            self.RecruitmentStrategy = RecruitmentStrategy


        except Exception as e:
            raise RecruitException(e, sys) from e

    def get_recruit_input_dataframe(self)-> DataFrame:
        """
        This function returns a DataFrame from RecruitData class input
        """
        try:
            
            recruit_input_dict = self.get_recruit_data_as_dict()
            return DataFrame(recruit_input_dict)
        
        except Exception as e:
            raise RecruitException(e, sys) from e


    def get_recruit_data_as_dict(self):
        """
        This function returns a dictionary from RecruitData class input 
        """
        logging.info("Entered get_recruit_data_as_dict method as RecruitData class")

        try:
            input_data = {
                "Age": [self.Age],
                "ExperienceYears": [self.ExperienceYears],
                "PreviousCompanies": [self.PreviousCompanies],
                "DistanceFromCompany": [self.DistanceFromCompany],
                "InterviewScore": [self.InterviewScore],
                "SkillScore": [self.SkillScore],
                "PersonalityScore": [self.PersonalityScore],
                "Gender": [self.Gender],
                "EducationLevel": [self.EducationLevel],
                "RecruitmentStrategy": [self.RecruitmentStrategy],
            }

            logging.info("Created recruit data dict")

            logging.info("Exited get_recruit_data_as_dict method as RecruitData class")

            return input_data

        except Exception as e:
            raise RecruitException(e, sys) from e

class RecruitClassifier:
    def __init__(self,prediction_pipeline_config: RecruitPredictorConfig = RecruitPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise RecruitException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of RecruitClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of RecruitClassifier class")
            model = RecruitEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise RecruitException(e, sys)