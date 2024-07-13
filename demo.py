from recruitment.logger import logging
from recruitment.exception import RecruitException
import sys
from recruitment.pipeline.training_pipeline import TrainPipeline

def main():
    pipeline = TrainPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()